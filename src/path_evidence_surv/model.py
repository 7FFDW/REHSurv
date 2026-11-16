from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .spatial import build_spatial_index


def masked_attention(logits, mask, topk=None):
    """Return zero for an unavailable modality, never NaN or uniform fake evidence."""
    mask = mask.expand_as(logits)
    values = logits.masked_fill(~mask, -1e4)
    if topk and topk < values.shape[-1]:
        selected = torch.zeros_like(mask)
        selected.scatter_(-1, values.topk(topk, dim=-1).indices, True)
        mask = mask & selected
        values = values.masked_fill(~mask, -1e4)
    weights = values.softmax(dim=-1) * mask
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def spatial_neighbors(x, coords, patch_mask, slide_ids, k, neighbor_index=None):
    """Exact intra-slide kNN pooling using O(N*k) sparse storage, not N*N."""
    if k <= 0:
        return x
    batch_size, n, d = x.shape
    if neighbor_index is None:
        # Convenience for direct model calls. The dataset normally computes this
        # once on CPU and reuses it across all graph samples in this batch.
        import numpy as np
        slides = slide_ids.detach().cpu().numpy().copy()
        slides[~patch_mask.detach().cpu().numpy()] = -1
        neighbor_index = torch.from_numpy(np.stack([
            build_spatial_index(c, s, k) for c, s in zip(coords.detach().cpu().numpy(), slides)
        ])).to(x.device)
    if neighbor_index.shape != (batch_size, n, k):
        raise ValueError("Spatial neighbor shape differs from model spatial_k")
    valid = (neighbor_index >= 0) & (neighbor_index < n) & patch_mask[..., None]
    safe = neighbor_index.clamp(0, n - 1)
    b = torch.arange(batch_size, device=x.device)[:, None, None]
    valid &= patch_mask[b, safe] & (slide_ids[b, safe] == slide_ids[..., None])
    degree = valid.sum(-1).clamp_min(1)
    rows = torch.arange(batch_size * n, device=x.device).reshape(batch_size, n, 1).expand_as(safe)
    columns = safe + b * n
    weights = degree.reciprocal().to(x.dtype)[..., None].expand_as(safe)[valid]
    adjacency = torch.sparse_coo_tensor(torch.stack([rows[valid], columns[valid]]), weights,
                                        (batch_size * n, batch_size * n), device=x.device,
                                        check_invariants=False).coalesce()
    pooled = torch.sparse.mm(adjacency, x.reshape(-1, d)).reshape(batch_size, n, d)
    return (x + pooled) * 0.5 * patch_mask[..., None]


class EvidenceHypergraph(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=128, num_pathways=186,
                 num_concepts=12, num_bins=8, model_type="evidence_hypergraph",
                 spatial_k=8, image_topk=0, pathway_topk=12, temperature=0.67,
                 dropout=0.1, image_chunk_size=2048, checkpoint_image=True):
        super().__init__()
        allowed = {"evidence_hypergraph"}
        if model_type not in allowed:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.model_type, self.temperature = model_type, temperature
        self.spatial_k, self.image_topk, self.pathway_topk = spatial_k, image_topk, pathway_topk
        self.hidden_dim, self.num_concepts = hidden_dim, num_concepts
        if image_chunk_size < 1:
            raise ValueError("image_chunk_size must be positive")
        self.image_chunk_size, self.checkpoint_image = image_chunk_size, checkpoint_image
        self.image_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.pathway_value = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.pathway_id = nn.Embedding(num_pathways, hidden_dim)
        self.concept_id = nn.Embedding(num_concepts, hidden_dim)
        self.state_embedding = nn.Embedding(4, hidden_dim)
        self.score_encoder = nn.Linear(1, hidden_dim, bias=False)
        self.image_pool_query = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.gene_pool_query = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.edge_posterior = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.triad_image, self.triad_gene, self.triad_report = [nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, num_bins))

    def forward(self, batch, sample_graph=False, return_attention=False):
        encoded = []
        for chunk in batch["x"].split(self.image_chunk_size, dim=1):
            if self.training and self.checkpoint_image and torch.is_grad_enabled():
                encoded.append(checkpoint(self.image_encoder, chunk, use_reentrant=False,
                                          preserve_rng_state=False))
            else:
                encoded.append(self.image_encoder(chunk))
        x = torch.cat(encoded, dim=1)
        x = spatial_neighbors(x, batch["coords"], batch["patch_mask"], batch["slide_ids"], self.spatial_k,
                              batch.get("neighbor_index"))
        pathway_mask = batch["pathway_mask"]
        p = self.pathway_value(batch["pathways"][..., None])
        p = p + self.pathway_id.weight[None, :]
        p = p * pathway_mask[..., None]
        states, scores = batch["states"], batch["scores"]
        concept = self.concept_id.weight[None, :, :].expand(len(x), -1, -1)
        report = self.state_embedding(states) + self.score_encoder(scores[..., None])
        query = concept + report
        ai = masked_attention(torch.einsum("bed,bnd->ben", query, x) / math.sqrt(self.hidden_dim),
                              batch["patch_mask"][:, None, :], self.image_topk)
        ag = masked_attention(torch.einsum("bed,bpd->bep", query, p) / math.sqrt(self.hidden_dim),
                              pathway_mask[:, None, :], self.pathway_topk)
        ix, gp = ai @ x, ag @ p
        posterior_logits = self.edge_posterior(torch.cat([ix, gp, query], dim=-1)).squeeze(-1)
        probability = posterior_logits.sigmoid()
        available = batch["patch_mask"].any(-1) & pathway_mask.any(-1)
        if sample_graph and self.model_type == "evidence_hypergraph":
            u = torch.rand_like(probability).clamp(1e-6, 1 - 1e-6)
            gates = ((posterior_logits + torch.logit(u)) / self.temperature).sigmoid()
        else:
            gates = probability
        gates = gates * available[:, None]
        interaction = self.triad_image(ix) * self.triad_gene(gp) * torch.tanh(self.triad_report(query))
        edge_values = self.edge_norm(interaction)
        # Fixed denominator preserves the effect of gate strength (no cancellation).
        fused = (edge_values * gates[..., None]).mean(1)
        image_weights = masked_attention(x @ self.image_pool_query, batch["patch_mask"])
        gene_weights = masked_attention(p @ self.gene_pool_query, pathway_mask)
        image_private = torch.einsum("bn,bnd->bd", image_weights, x)
        gene_private = torch.einsum("bp,bpd->bd", gene_weights, p)
        report_private = report.mean(1)
        logits = self.head(torch.cat([image_private, gene_private, report_private, fused], dim=-1))
        # Evidence priors constrain the phenotype interaction, not pathway absence.
        asserted = (states == 1) | (states == 2)
        direction = torch.where(states == 1, 1.0, -1.0)
        prior = (0.5 + 0.3 * direction * scores).clamp(0.05, 0.95)
        q = probability.clamp(1e-6, 1 - 1e-6)
        kl = q * (q / prior).log() + (1 - q) * ((1 - q) / (1 - prior)).log()
        weight = asserted * scores * available[:, None]
        evidence_kl = (kl * weight).sum() / weight.sum().clamp_min(1)
        result = {"logits": logits, "edge_probability": probability,
                  "edge_gates": gates, "evidence_kl": evidence_kl,
                  "edge_entropy": -(q * q.log() + (1-q) * (1-q).log()).mean(-1)}
        if return_attention:
            result.update(image_membership=ai, pathway_membership=ag)
        return result


def survival_curves(logits):
    return torch.exp(F.logsigmoid(-logits).cumsum(dim=-1))


def perturb_assertions(batch, kind: str, concept: int | None = None):
    result = dict(batch)
    result["states"], result["scores"] = batch["states"].clone(), batch["scores"].clone()
    if kind == "style":
        result["states"], result["scores"] = batch["style_states"], batch["style_scores"]
    elif kind == "missing":
        if concept is None:
            result["states"].zero_(); result["scores"].zero_()
        else:
            result["states"][:, concept] = 0; result["scores"][:, concept] = 0
    elif kind == "conflict":
        if concept is None:
            raise ValueError("Choose one concept for a localized conflict probe")
        state = result["states"][:, concept]
        result["states"][:, concept] = torch.where(state == 1, 2, torch.where(state == 2, 1, state))
    else:
        raise ValueError(kind)
    return result
