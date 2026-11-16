from pathlib import Path
import numpy as np
import torch
from .common import digest
from .data import ExpressionStore
from .evidence import CONCEPT_NAMES
from .model import EvidenceHypergraph


def _make_model(config, num_pathways, num_bins):
    return EvidenceHypergraph(feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"],
                              num_pathways=num_pathways, num_concepts=len(CONCEPT_NAMES), num_bins=num_bins,
                              model_type=config["model_type"], spatial_k=config["spatial_k"],
                              image_topk=config["image_topk"], pathway_topk=config["pathway_topk"],
                              temperature=config["temperature"], dropout=config["dropout"],
                              image_chunk_size=config.get("image_chunk_size", 2048),
                              checkpoint_image=config.get("checkpoint_image", True))


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


def load_run(checkpoint_path, data_dir=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("concept_names") != CONCEPT_NAMES:
        raise ValueError("Checkpoint uses an older concept vocabulary; use its original code or retrain")
    config = checkpoint["config"]
    if data_dir:
        config["data_dir"] = data_dir
    if digest(Path(config["data_dir"])/"patients.jsonl") != checkpoint["manifest_sha256"]:
        raise ValueError("Prepared patient manifest differs from checkpoint; refuse silent split/data changes")
    store = ExpressionStore(config["data_dir"])
    for name, scaler in checkpoint["scalers"].items():
        expected_hash = checkpoint.get("archive_sha256", {}).get(name)
        if expected_hash and digest(Path(config["data_dir"])/name) != expected_hash:
            raise ValueError("Expression cache changed since training")
        actual = store.archive(name)
        if actual["genes"].tolist() != checkpoint["expression_genes"][name]:
            raise ValueError("Expression gene order differs from training checkpoint")
        if actual["pathways"].tolist() != checkpoint["pathway_names"]:
            raise ValueError("Pathway order differs from checkpoint")
        store.scalers[name] = {**scaler, "mean": np.asarray(scaler["mean"], dtype=np.float32),
                               "scale": np.asarray(scaler["scale"], dtype=np.float32)}
    device = _resolve_device(device)
    model = _make_model(config, checkpoint["num_pathways"], len(checkpoint["cutpoints"])+1).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model, store, device
