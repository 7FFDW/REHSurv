"""Unified entry: edit DEFAULTS/get_args, data loading, training, validation and test here.

Portable release of the user-supplied latest main.py. Runs directly on this machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

# ===== 1. EDIT EXPERIMENT SETTINGS HERE (CLI arguments override these) =====
DEFAULTS = {
    'data_dir': None,  # None selects the matching COHORT_PROFILES cache automatically.
    'output_dir': 'runs',
    'cohort': 'LUAD',
    'fold': 0,
    'seed': 42,
    'device': 'auto',
    'cpu_threads': 4,
    'model_type': 'evidence_hypergraph',
    'feature_dim': 1536,
    'hidden_dim': 128,
    'max_patches': None,
    'image_chunk_size': 2048,
    'checkpoint_image': True,
    'feature_cache_size': 0,
    'spatial_k': 8,
    'image_topk': 0,
    'pathway_topk': 12,
    'temperature': 0.67,
    'dropout': 0.1,
    'num_bins': 8,
    'batch_size': 1,
    'epochs': 40,
    'patience': 10,
    'lr_patience': 3,
    'lr_factor': 0.5,
    'min_lr': 1e-6,
    'min_delta': 0.001,
    'learning_rate': 0.0002,
    'log_every_steps': 1,
    'weight_decay': 0.0001,
    'evidence_kl_weight': 0.02,
    'style_weight': 0.0,
    'report_dropout': 0.15,
    'mc_samples': 16,
    'bootstrap_repeats': 0,
    'tss_holdout': [],
    'confirmed_time_unit': None,
    'time_unit_evidence': None,
    'physical_gpu': '0',
    'fold_list': [0, 1, 2, 3, 4],
    'test_only': False,
    'checkpoint': None,
}

# Per-cohort caches keep each cohort's reports/expression and provenance separate.
# Shared training defaults above apply to every cohort; edit overrides here as needed.
COHORT_PROFILES = {name: {'data_dir': 'data/prepared' if name == 'LUAD' else f'data/prepared_{name}'}
                   for name in ['LUAD', 'BLCA', 'BRCA', 'UCEC']}


import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from path_evidence_surv.common import digest, read_json, read_jsonl, seed_everything, write_json, write_jsonl
from path_evidence_surv.data import ExpressionStore, PatientDataset, collate_patients, split_patients, to_device
from path_evidence_surv.evidence import CONCEPT_NAMES
from path_evidence_surv.model import EvidenceHypergraph, perturb_assertions, survival_curves
from path_evidence_surv.survival import (bootstrap_cindex, concordance_index, discrete_survival_nll,
                       fit_cutpoints, ipcw_scores, survival_at)


# ===== 2. ARGUMENTS: same style as the supplied original project =====
def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Optional resolved JSON; explicit CLI arguments take priority')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dataset', '--cohort', dest='cohort', choices=list(COHORT_PROFILES))
    parser.add_argument('--fold_list', '--fold-list', type=int, nargs='+')
    parser.add_argument('--data_dir', '--data-dir', dest='data_dir')
    parser.add_argument('--ckpt_dir', '--output-dir', dest='output_dir')
    parser.add_argument('--device_ids', '--gpu', dest='physical_gpu')
    parser.add_argument('--device', help='Logical device; CUDA_VISIBLE_DEVICES maps the selected GPU to cuda:0')
    parser.add_argument('--model', dest='model_type', choices=['evidence_hypergraph'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--lr', dest='learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--input_dim', dest='feature_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--num_bins', type=int)
    parser.add_argument('--max_patches', type=int, help='0 keeps every existing patch; positive value explicitly samples per patient')
    parser.add_argument('--image_chunk_size', type=int)
    parser.add_argument('--image_topk', type=int)
    parser.add_argument('--pathway_topk', type=int)
    parser.add_argument('--spatial_k', type=int)
    parser.add_argument('--mc_samples', type=int)
    parser.add_argument('--cpu_threads', type=int)
    parser.add_argument('--log_every_steps', type=int)
    parser.add_argument('--report_dropout', type=float)
    parser.add_argument('--evidence_kl_weight', type=float)
    parser.add_argument('--style_weight', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--test_only', action='store_true', help='Only test an existing best.pt; no training')
    parser.add_argument('--checkpoint', help='Local checkpoint path for test_only')
    parser.add_argument('--print_config', action='store_true', help='Print effective settings without connecting or training')
    preliminary, _ = parser.parse_known_args(argv)
    defaults = copy.deepcopy(DEFAULTS)
    overrides = {}
    if preliminary.config:
        with Path(preliminary.config).open(encoding='utf-8-sig') as stream:
            overrides = json.load(stream)
        if 'base_config' in overrides:
            parser.error('Use a resolved JSON or edit DEFAULTS in main.py; inherited JSON is not required here.')
    cohort = preliminary.cohort or overrides.get('cohort', defaults['cohort'])
    if cohort not in COHORT_PROFILES:
        parser.error(f'Unsupported cohort: {cohort}')
    defaults.update(COHORT_PROFILES[cohort])
    if cohort == DEFAULTS['cohort'] and DEFAULTS['data_dir'] is not None:
        defaults['data_dir'] = DEFAULTS['data_dir']
    defaults.update(overrides)
    if (preliminary.test_only or overrides.get('test_only')) and preliminary.data_dir is None and 'data_dir' not in overrides:
        defaults['data_dir'] = None  # Reuse the checkpoint's training location.
    parser.set_defaults(**defaults)
    parsed = parser.parse_args(argv)
    if parsed.max_patches == 0:
        parsed.max_patches = None
    if parsed.max_patches is not None and parsed.max_patches < 1:
        parser.error('max_patches must be 0 (all) or positive')
    if not parsed.fold_list or len(set(parsed.fold_list)) != len(parsed.fold_list) or any(f not in range(5) for f in parsed.fold_list):
        parser.error('fold_list must contain distinct fold numbers from 0 to 4')
    if any(getattr(parsed, key) < 1 for key in ['epochs', 'patience', 'batch_size', 'mc_samples', 'image_chunk_size', 'log_every_steps', 'cpu_threads']):
        parser.error('epochs/patience/batch_size/mc_samples/chunk/log_every_steps/cpu_threads must be positive')
    if parsed.learning_rate <= 0 or not 0 <= parsed.report_dropout <= 1:
        parser.error('lr must be positive; report_dropout must be in [0,1]')
    if parsed.test_only and not parsed.checkpoint:
        parser.error('test_only requires --checkpoint')
    return parsed


# ===== 3. DATASETS AND DATALOADERS =====
def build_loader(rows, store, config, training=False, style=False):
    dataset = PatientDataset(rows, store, config.get("max_patches"), config["seed"],
                             training=training, augment_style=style,
                             cache_size=int(config.get("feature_cache_size", 0)),
                             spatial_k=int(config["spatial_k"]))
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=0, collate_fn=collate_patients, generator=generator)


@torch.no_grad()
def predict(model, data_loader, device, mc_samples=1, perturb=None, concept=None,
            max_batches=None, explain=False, cutpoints=None):
    if cutpoints is not None and mc_samples != 1:
        raise ValueError("Validation likelihood logging requires a deterministic prediction")
    model.eval()
    records = []
    for step, batch in enumerate(data_loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(batch, device)
        if perturb:
            batch = perturb_assertions(batch, perturb, concept)
        curves, probs, gates = [], [], []
        for _ in range(mc_samples):
            output = model(batch, sample_graph=mc_samples > 1, return_attention=explain)
            curves.append(survival_curves(output["logits"]))
            probs.append(output["edge_probability"])
            gates.append(output["edge_gates"])
        curves = torch.stack(curves)
        mean, sd = curves.mean(0), curves.std(0, unbiased=False)
        probability = torch.stack(probs).mean(0)
        gate_sd = torch.stack(gates).std(0, unbiased=False)
        for i, patient in enumerate(batch["patient_id"]):
            record = {"patient_id": patient, "time": float(batch["time"][i]),
                      "patch_count": int(batch["patch_mask"][i].sum()),
                      "patch_selection": "all" if batch["all_patches"][i] else "sampled",
                      "event": int(batch["event"][i]), "survival": mean[i].cpu().tolist(),
                      "survival_mc_sd": sd[i].cpu().tolist(),
                      "risk": float(-mean[i, :-1].mean()),
                      "edge_probability": probability[i].cpu().tolist(),
                      "edge_gate_mc_sd": gate_sd[i].cpu().tolist()}
            if cutpoints is not None:
                record["nll"] = float(discrete_survival_nll(output["logits"][i:i+1],
                                       batch["time"][i:i+1], batch["event"][i:i+1], cutpoints))
            if explain:
                image = output["image_membership"][i]
                gene = output["pathway_membership"][i]
                valid_patches = int(batch["patch_mask"][i].sum())
                record["connections"] = [
                    {"concept": name, "probability": float(probability[i, j]),
                     "patches": [
                         {"input_index": index,
                          "feature_path": batch["feature_paths"][i][int(batch["slide_ids"][i, index])],
                          "feature_row": int(batch["patch_rows"][i, index]),
                          "membership_weight": float(image[j, index])}
                         for index in image[j, :valid_patches].topk(min(5, valid_patches)).indices.cpu().tolist()],
                     "pathway_indices": gene[j].topk(min(5, gene.shape[-1])).indices.cpu().tolist()}
                    for j, name in enumerate(CONCEPT_NAMES)]
            records.append(record)
    return records


def evaluate_records(records, train_rows, cutpoints, bootstrap=0, seed=42):
    if not records:
        raise ValueError("No evaluation patients")
    t = np.asarray([r["time"] for r in records])
    e = np.asarray([r["event"] for r in records])
    risk = np.asarray([r["risk"] for r in records])
    curves = np.asarray([r["survival"] for r in records])
    tt = np.asarray([r["time"] for r in train_rows])
    te = np.asarray([r["event"] for r in train_rows])
    scores = ipcw_scores(tt, te, t, e, survival_at(curves, cutpoints, cutpoints), cutpoints)
    scores.update(patients=len(records), events=int(e.sum()),
                  c_index=concordance_index(t, e, risk),
                  c_index_bootstrap=bootstrap_cindex(t, e, risk, bootstrap, seed),
                  uncertainty_note="Graph-sampling standard deviation is uncalibrated; not a clinical confidence interval.")
    return scores


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


def _metric_text(value):
    return "NA" if value is None else f"{value:.6f}"


def _gpu_memory(device):
    if device.type != "cuda":
        return {}
    return {"gpu_allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
            "gpu_reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
            "gpu_peak_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)}


def _append_step(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record)+"\n")


# ===== 4. PER-FOLD DATA -> TRAIN -> VALIDATE -> BEST CHECKPOINT -> TEST =====
def run_fold(config: dict):
    config = copy.deepcopy(config)
    seed_everything(config["seed"])
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    data_root = Path(config["data_dir"]).resolve()
    rows = read_jsonl(data_root / "patients.jsonl")
    train_rows, val_rows, test_rows = split_patients(rows, config["cohort"], config["fold"], config.get("tss_holdout"))
    validate_comparable_validation(val_rows)
    units = {r["time_unit"] for r in train_rows + val_rows + test_rows}
    confirmed = config.get("confirmed_time_unit")
    if any("unverified" in unit for unit in units) and not (confirmed and config.get("time_unit_evidence")):
        raise ValueError("Unverified survival time unit. Set confirmed_time_unit AND time_unit_evidence after source verification.")
    if confirmed:
        if confirmed not in {"days", "months", "years"}:
            raise ValueError("confirmed_time_unit must be days, months, or years")
        units = {confirmed}
    if len(units) != 1:
        raise ValueError(f"Mixed time units: {units}")
    store = ExpressionStore(data_root)
    store.fit(train_rows)
    archive = store.archive(train_rows[0]["expression_archive"])
    cutpoints = fit_cutpoints([r["time"] for r in train_rows], [r["event"] for r in train_rows], config["num_bins"])
    if len(cutpoints) < 1:
        raise ValueError("Training event times do not provide usable time bins")
    device = _resolve_device(config["device"])
    model = _make_model(config, len(archive["pathways"]), len(cutpoints)+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    lr_patience = int(config.get("lr_patience", 3))
    lr_factor = float(config.get("lr_factor", 0.5))
    min_lr = float(config.get("min_lr", 1e-6))
    min_delta = float(config.get("min_delta", 0.001))
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=lr_patience,
        threshold=min_delta, threshold_mode='abs', min_lr=min_lr
    )
    # These three loaders are defined here; normalization is fitted ONLY on train_rows.
    train_loader = build_loader(train_rows, store, config, True, style=config["style_weight"] > 0)
    val_loader = build_loader(val_rows, store, config)
    test_loader = build_loader(test_rows, store, config, style=True)
    # Flatten: directly save checkpoint and records inside fold directory
    run_dir = Path(config["output_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "split.json", {"train": [r["patient_id"] for r in train_rows],
                                       "validation": [r["patient_id"] for r in val_rows],
                                       "test": [r["patient_id"] for r in test_rows],
                                       "manifest_sha256": digest(data_root / "patients.jsonl")})
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"[START] cohort={config['cohort']} fold={config['fold']} model={config['model_type']} "
          f"device={device} patches={'all' if not config.get('max_patches') else config['max_patches']} "
          f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
          f"epochs={config['epochs']} lr={config['learning_rate']:.3e} lr_schedule=constant", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[GPU] {torch.cuda.get_device_name(device)}", flush=True)
    best_c_index, stale, history = float("-inf"), 0, []
    best = float("inf")
    training_patch_usage = {}
    total_steps = 0
    for epoch in range(config["epochs"]):
        epoch_started = time.monotonic()
        model.train()
        train_loader.dataset.epoch = epoch
        losses, nll_losses, kl_losses, batch_sizes = [], [], [], []
        expected_steps = min(len(train_loader), config.get("max_train_batches") or len(train_loader))
        best_c_str = f"{best_c_index:.4f}" if best_c_index != float("-inf") else "NA"
        best_nll_str = f"{best:.4f}" if best != float("inf") else "NA"
        pbar_desc = f"[Fold {config['fold']}] Epoch {epoch+1}/{config['epochs']} (Best C-Idx:{best_c_str} NLL:{best_nll_str})"
        pbar = tqdm(enumerate(train_loader), total=expected_steps, desc=pbar_desc, dynamic_ncols=True, file=sys.stdout)
        for step, raw_batch in pbar:
            if config.get("max_train_batches") and step >= config["max_train_batches"]:
                break
            batch = to_device(raw_batch, device)
            step_started = time.monotonic()
            for i, patient in enumerate(batch["patient_id"]):
                training_patch_usage[patient] = {
                    "patch_count": int(batch["patch_mask"][i].sum()),
                    "patch_selection": "all" if batch["all_patches"][i] else "sampled"}
            # Report dropout trains a usable fallback; missing evidence is never made negative.
            drop = torch.rand(len(batch["x"]), device=device) < config["report_dropout"]
            if drop.any():
                for key in ["states", "scores", "style_states", "style_scores"]:
                    batch[key] = batch[key].clone(); batch[key][drop] = 0
            output = model(batch, sample_graph=True)
            nll = discrete_survival_nll(output["logits"], batch["time"], batch["event"], cutpoints)
            objective = nll + config["evidence_kl_weight"] * output["evidence_kl"]
            if config["style_weight"] > 0:
                # Compare deterministic graphs with dropout off; do not fit consistency to MC noise.
                model.eval()
                reference = model(batch, sample_graph=False)
                styled = model(perturb_assertions(batch, "style"), sample_graph=False)
                objective = objective + config["style_weight"] * (
                    F.mse_loss(survival_curves(reference["logits"]), survival_curves(styled["logits"])) +
                    F.mse_loss(reference["edge_probability"], styled["edge_probability"]))
                model.train()
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_sizes.append(len(batch["patient_id"]))
            losses.append(float(objective.detach()))
            nll_losses.append(float(nll.detach())); kl_losses.append(float(output['evidence_kl'].detach()))
            total_steps += 1
            step_row = {"epoch": epoch+1, "step": step+1, "steps_in_epoch": expected_steps,
                        "global_step": total_steps, "loss": losses[-1], "nll": nll_losses[-1],
                        "evidence_kl": kl_losses[-1], "learning_rate": optimizer.param_groups[0]['lr'],
                        "patch_count": int(batch['patch_mask'].sum()),
                        "step_seconds": time.monotonic()-step_started, **_gpu_memory(device)}
            _append_step(run_dir / 'steps.jsonl', step_row)
            pbar.set_postfix({
                "loss": f"{losses[-1]:.4f}",
                "avg_loss": f"{np.average(losses, weights=batch_sizes):.4f}",
                "nll": f"{nll_losses[-1]:.4f}",
                "lr": f"{step_row['learning_rate']:.1e}",
            })
        print(f"[VALIDATE] [Fold {config['fold']}] epoch={epoch+1} patients={len(val_rows)}", flush=True)
        # Obtain NLL and survival metrics in one pass; only validation selects checkpoints.
        records = predict(model, val_loader, device, mc_samples=1, cutpoints=cutpoints)
        val_loss = float(np.mean([record['nll'] for record in records]))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        val_metrics = evaluate_records(records, train_rows, cutpoints)
        supported_auc = [h['auc'] for h in val_metrics['horizon_metrics'] if h['auc'] is not None]
        row = {"epoch": epoch+1, "train_loss": float(np.average(losses, weights=batch_sizes)),
               "train_nll": float(np.average(nll_losses, weights=batch_sizes)), "train_evidence_kl": float(np.average(kl_losses, weights=batch_sizes)),
               "learning_rate": optimizer.param_groups[0]['lr'], "validation_nll": val_loss,
               "validation_c_index": val_metrics['c_index'], "validation_ibs": val_metrics['ibs'],
               "validation_auc_mean_supported": float(np.mean(supported_auc)) if supported_auc else None,
               "epoch_seconds": time.monotonic()-epoch_started, **_gpu_memory(device)}
        history.append(row); write_json(run_dir / "history.json", history)
        write_json(run_dir / "training_patch_usage.json", training_patch_usage)
        write_json(run_dir / f"validation_epoch_{epoch+1:03d}.json", val_metrics)
        print(f"[EPOCH] epoch={epoch+1}/{config['epochs']} train_loss={row['train_loss']:.6f} "
              f"lr={row['learning_rate']:.3e} val_nll={val_loss:.6f} "
              f"val_c_index={_metric_text(row['validation_c_index'])} val_ibs={_metric_text(row['validation_ibs'])} "
              f"val_auc_mean={_metric_text(row['validation_auc_mean_supported'])} "
              f"elapsed_s={row['epoch_seconds']:.2f}", flush=True)
        print(json.dumps(row), flush=True)
        curr_c_index = val_metrics['c_index']
        if curr_c_index is None or not np.isfinite(curr_c_index):
            raise ValueError('Validation C-index is undefined; no checkpoint can be selected under the paper protocol.')
        prev_lr = optimizer.param_groups[0]['lr']
        # Step ReduceLROnPlateau using validation C-index (mode='max')
        scheduler.step(curr_c_index)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"[LR_REDUCE] [Fold {config['fold']}] Validation C-index stagnation detected. Learning rate reduced from {prev_lr:.3e} -> {new_lr:.3e}", flush=True)

        # Check effective improvement (threshold min_delta=0.001) for saving best model and early stopping
        if curr_c_index > best_c_index + min_delta:
            best_c_index = curr_c_index
            best = val_loss
            stale = 0
            checkpoint = {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "config": config,
                          "cutpoints": cutpoints.tolist(), "num_pathways": len(archive["pathways"]),
                          "pathway_names": archive["pathways"].tolist(), "concept_names": CONCEPT_NAMES,
                          "scalers": {name: {"mean": s["mean"].tolist(), "scale": s["scale"].tolist(),
                                             "training_patient_ids": s["training_patient_ids"]}
                                      for name, s in store.scalers.items()},
                          "expression_genes": {name: store.archive(name)["genes"].tolist() for name in store.scalers},
                          "train_rows": train_rows, "validation_rows": val_rows,
                          "test_rows": test_rows,
                          "manifest_sha256": digest(data_root / "patients.jsonl"),
                          "archive_sha256": {name: digest(data_root / name) for name in store.scalers},
                          "time_unit": next(iter(units)), "smoke_only": False,
                          "best_epoch": epoch+1, "torch_version": str(torch.__version__),
                          "best_c_index": best_c_index, "best_val_nll": val_loss}
            torch.save(checkpoint, run_dir / "best.pt")
            print(f"[BEST] [Fold {config['fold']}] epoch={epoch+1} New Best C-Index={best_c_index:.6f} (NLL={val_loss:.6f}) saved={run_dir/'best.pt'}", flush=True)
        else:
            stale += 1
            print(f"[STALE] [Fold {config['fold']}] No effective C-index improvement (>={min_delta}) for {stale}/{config['patience']} epochs.", flush=True)
        if stale >= config["patience"]:
            print(f"[EARLY_STOP] epoch={epoch+1} patience={config['patience']}", flush=True)
            break
    loaded = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(loaded["state_dict"])
    final_records = predict(model, test_loader, device,
                            mc_samples=config["mc_samples"])
    metrics = evaluate_records(final_records, train_rows, cutpoints, config.get("bootstrap_repeats", 0), config["seed"])
    metrics.update(smoke_only=False, split="test",
                   time_unit=next(iter(units)), best_epoch=loaded["best_epoch"],
                   model_parameters=sum(p.numel() for p in model.parameters()))
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "predictions.jsonl", final_records)
    print(f"[FINAL] split={metrics['split']} c_index={_metric_text(metrics['c_index'])} "
          f"ibs={_metric_text(metrics['ibs'])} metrics={run_dir/'metrics.json'}", flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    return str(run_dir)


# ===== 5. TEST AN EXISTING CHECKPOINT WITHOUT REFITTING ITS SCALERS =====
def test_checkpoint(config):
    from path_evidence_surv.engine import load_run
    seed_everything(config['seed'])
    checkpoint, model, store, device = load_run(config['checkpoint'], data_dir=config['data_dir'], device=config['device'])
    if checkpoint['smoke_only']:
        raise ValueError('A smoke checkpoint cannot be used as a formal test model.')
    trained = checkpoint['config']
    records = predict(model, build_loader(checkpoint['test_rows'], store, trained, style=True), device,
                      mc_samples=config['mc_samples'])
    metrics = evaluate_records(records, checkpoint['train_rows'], checkpoint['cutpoints'],
                               config['bootstrap_repeats'], trained['seed'])
    metrics.update(split='test', smoke_only=False, best_epoch=checkpoint['best_epoch'])
    output = Path(config['output_dir']) / ('TEST_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', metrics)
    write_jsonl(output / 'predictions.jsonl', records)
    print(f'[TEST_DONE] c_index={_metric_text(metrics["c_index"])} ibs={_metric_text(metrics["ibs"])} output={output}', flush=True)
    return str(output)


# ===== 6. FOLD LOOP AND SUMMARY (no re-splitting, no test-based selection) =====
def validate_comparable_validation(rows):
    times = np.asarray([r['time'] for r in rows], dtype=float)
    events = np.asarray([r['event'] for r in rows], dtype=int)
    if not np.isfinite(times).all() or not np.isin(events, [0, 1]).all():
        raise ValueError('Validation survival times/events are invalid.')
    if not len(times) or not np.any(events == 1) or not np.any(times[events == 1] < times.max()):
        raise ValueError('Validation C-index is undefined: no comparable patient pairs. Check the predefined validation fold; training has not started.')


def validate_cohort_data(config):
    manifest = Path(config['data_dir']) / 'patients.jsonl'
    if not manifest.is_file():
        raise ValueError(f'{config["cohort"]}: formal trimodal cache missing: {manifest}. Prepare this cohort before training.')
    rows = read_jsonl(manifest)
    selected = [r for r in rows if r['cohort'] == config['cohort']]
    if not selected or any(not r.get('has_report') or not r.get('report', '').strip() for r in selected):
        raise ValueError('Formal multimodal entry requires a nonempty cohort with reports for every patient.')
    units = {r['time_unit'] for r in selected}
    confirmed = config.get('confirmed_time_unit')
    if any('unverified' in unit for unit in units) and not (confirmed and config.get('time_unit_evidence')):
        raise ValueError(f'{config["cohort"]}: survival time unit is unverified. Verify its source, then set confirmed_time_unit and time_unit_evidence; do not assume days.')
    if confirmed and confirmed not in {'days', 'months', 'years'}:
        raise ValueError('confirmed_time_unit must be days, months or years')
    if not confirmed and len(units) != 1:
        raise ValueError(f'Mixed time units: {units}')
    for fold in config['fold_list']:
        _, validation, _ = split_patients(rows, config['cohort'], fold, config.get('tss_holdout'))
        validate_comparable_validation(validation)
    return rows


def run_experiment(config):
    try:
        validate_cohort_data(config)
    except ValueError as exc:
        raise SystemExit(f'[DATA_NOT_READY] {exc}') from None
    
    resume_target = config.get('resume_dir')
    if resume_target:
        output = Path(resume_target).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'summary.json').is_file():
            summary = read_json(output / 'summary.json')
            summary['state'] = 'running'
        else:
            summary = {'state': 'running', 'cohort': config['cohort'], 'fold_order': config['fold_list'], 'folds': [],
                       'protocol': 'test=k, validation=(k+1)%5, train=remaining; validation C-index selects best.pt',
                       'note': 'Sample SD is not a confidence interval; IBS/AUC use fold-specific training time grids. No pooled risk C-index.'}
    else:
        output = Path(config['output_dir']).resolve() / (config['cohort'] + '_MAIN_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        output.mkdir(parents=True, exist_ok=False)
        summary = {'state': 'running', 'cohort': config['cohort'], 'fold_order': config['fold_list'], 'folds': [],
                   'protocol': 'test=k, validation=(k+1)%5, train=remaining; validation C-index selects best.pt',
                   'note': 'Sample SD is not a confidence interval; IBS/AUC use fold-specific training time grids. No pooled risk C-index.'}
    
    write_json(output / 'config.json', config)
    write_json(output / 'summary.json', summary)
    print(f'[MAIN_START] folds={config["fold_list"]} output={output}', flush=True)
    from path_evidence_surv.cross_validation import fold_record, aggregate
    try:
        for fold in config['fold_list']:
            folder = output / f'fold_{fold}'
            # Check if this fold is already completed (check flat directory and nested directory)
            finished_run_dir = None
            if (folder / 'metrics.json').is_file() and (folder / 'best.pt').is_file():
                finished_run_dir = folder
            elif (folder / 'result.json').is_file():
                r_dir = Path(read_json(folder / 'result.json').get('run_dir', ''))
                if (r_dir / 'metrics.json').is_file():
                    finished_run_dir = r_dir
            elif folder.is_dir():
                subdirs = [p for p in folder.glob('*') if p.is_dir() and (p / 'metrics.json').is_file()]
                if subdirs:
                    finished_run_dir = subdirs[0]

            if finished_run_dir is not None:
                print(f'[RESUME] Fold {fold} already completed at {finished_run_dir}. Skipping fold training.', flush=True)
                record = fold_record(fold, finished_run_dir)
                summary['folds'] = [f for f in summary['folds'] if f.get('fold') != fold]
                summary['folds'].append(record)
                summary['folds'].sort(key=lambda x: x['fold'])
                summary['aggregate'] = aggregate(summary['folds'])
                write_json(output / 'summary.json', summary)
                with (output / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                    writer.writerows(summary['folds'])
                continue

            folder.mkdir(parents=True, exist_ok=True)
            current = copy.deepcopy(config)
            current.update(fold=fold, output_dir=str(folder))
            path = folder / 'config.json'
            write_json(path, current)
            print(f'[MAIN_FOLD_START] fold={fold}', flush=True)
            subprocess.run([sys.executable, '-u', '-B', str(ROOT / 'main.py'), '--worker',
                            '--config', str(path)], cwd=ROOT, check=True)
            res_path = folder / 'result.json'
            run_dir = Path(read_json(res_path)['run_dir']) if res_path.is_file() else folder
            record = fold_record(fold, run_dir)
            summary['folds'] = [f for f in summary['folds'] if f.get('fold') != fold]
            summary['folds'].append(record)
            summary['folds'].sort(key=lambda x: x['fold'])
            summary['aggregate'] = aggregate(summary['folds'])
            write_json(output / 'summary.json', summary)
            with (output / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(summary['folds'])
            print(f'[MAIN_FOLD_DONE] fold={fold} test_c_index={record["c_index"]} test_ibs={record["ibs"]}', flush=True)
        summary['state'] = 'completed'
        write_json(output / 'summary.json', summary)
        print(f'[MAIN_DONE] summary={output / "summary.json"}', flush=True)
        print('[MAIN_AGGREGATE] ' + json.dumps(summary['aggregate']), flush=True)
        return str(output)
    except BaseException as exc:
        summary.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'summary.json', summary)
        raise


def main(argv=None):
    args = get_args(argv)
    config = vars(args).copy()
    config.pop('config', None)
    if config.pop('print_config', False):
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    worker = config.pop('worker', False)
    if config['device'] == 'auto':
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if str(config['device']).startswith('cuda'):
        if not str(config['physical_gpu']).isdigit():
            raise ValueError('device_ids must identify one physical GPU')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['physical_gpu'])
        config['device'] = 'cuda:0'
    if config.get('test_only'):
        test_checkpoint(config)
    elif worker:
        run_dir = run_fold(config)
        write_json(Path(config['output_dir']) / 'result.json', {'run_dir': run_dir})
    else:
        run_experiment(config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

