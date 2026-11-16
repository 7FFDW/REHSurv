from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .evidence import CONCEPT_NAMES, EVIDENCE_SCHEMA, extract_evidence, rewrite_style, vectorize
from .spatial import build_spatial_index


def split_patients(rows: list[dict], cohort: str, fold: int, tss_holdout: list[str] | None = None):
    selected = [r for r in rows if r["cohort"] == cohort]
    for row in selected:
        if row.get("evidence_schema") != EVIDENCE_SCHEMA or [a["concept"] for a in row.get("assertions", [])] != CONCEPT_NAMES:
            raise ValueError("Concept cache is incompatible; regenerate it with prepare_data.py")
    if len({r["patient_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate patients would leak across splits")
    test = [r for r in selected if (r["tss"] in tss_holdout if tss_holdout else r["fold"] == fold)]
    pool = [r for r in selected if r not in test]
    val = [r for r in pool if r["fold"] == (fold + 1) % 5]
    train = [r for r in pool if r["fold"] != (fold + 1) % 5]
    if not train or not val or not test:
        raise ValueError("Train/validation/test split is empty; adjust fold or held-out TSS")
    if not any(r["event"] for r in train):
        raise ValueError("Training split has no events")
    groups = [set(r["patient_id"] for r in group) for group in (train, val, test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping patient splits")
    return train, val, test


class ExpressionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.archives = {}
        self.scalers = {}

    def archive(self, name: str):
        if name not in self.archives:
            with np.load(self.root / name, allow_pickle=False) as file:
                self.archives[name] = {k: file[k] for k in file.files}
        return self.archives[name]

    def fit(self, train_rows: list[dict]) -> None:
        for name in sorted({r["expression_archive"] for r in train_rows}):
            indices = [r["expression_index"] for r in train_rows if r["expression_archive"] == name]
            a = self.archive(name)
            values = a["values"][indices]
            mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
            scale[scale < 1e-5] = 1.0
            self.scalers[name] = {"mean": mean, "scale": scale,
                                  "training_patient_ids": [r["patient_id"] for r in train_rows
                                                           if r["expression_archive"] == name]}

    def transform(self, row: dict):
        name = row["expression_archive"]
        a = self.archive(name)
        if "patient_ids" in a and str(a["patient_ids"][row["expression_index"]]) != row["patient_id"]:
            raise ValueError("Expression row/patient ID mismatch")
        s = self.scalers[name]
        z = np.clip((a["values"][row["expression_index"]] - s["mean"]) / s["scale"], -8, 8)
        z[~a["measured"]] = 0
        membership = a["membership"] * a["measured"][None, :]
        count = membership.sum(axis=1)
        # Zero denotes a standardized mean, never an imputed measured gene.
        pathway = membership @ z / np.maximum(count, 1)
        return pathway.astype(np.float32), (count > 0)


class PatientDataset(Dataset):
    def __init__(self, rows: list[dict], expression: ExpressionStore, max_patches: int | None = None,
                 seed: int = 42, training: bool = False, augment_style: bool = False,
                 cache_size: int = 4, spatial_k: int = 8):
        self.rows, self.expression = rows, expression
        if max_patches is not None and max_patches < 0:
            raise ValueError("max_patches must be null/0 (all) or a positive integer")
        self.max_patches, self.seed, self.training = max_patches, seed, training
        self.spatial_k = spatial_k
        self.augment_style, self.epoch = augment_style, 0
        self.cache, self.cache_size = OrderedDict(), cache_size

    def __len__(self):
        return len(self.rows)

    def _rng(self, patient: str):
        epoch = self.epoch if self.training else 0
        key = hashlib.blake2b(f"{self.seed}:{epoch}:{patient}".encode(), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(key, "little"))

    def _load(self, path: str, budget: int | None, rng):
        if Path(path).suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as f:
                count = len(f["features"])
                chosen = np.arange(count) if budget is None else np.sort(rng.choice(count, min(count, budget), replace=False))
                result = f["features"][chosen].astype(np.float32), f["coords"][chosen].astype(np.float32)
        else:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("h5py is required for real WSI HDF5 files") from exc
            with h5py.File(path, "r") as f:
                if f["features"].ndim != 2 or f["coords"].shape != (len(f["features"]), 2):
                    raise ValueError(f"Malformed HDF5 arrays: {path}")
                count = len(f["features"])
                chosen = np.arange(count) if budget is None else np.sort(rng.choice(count, min(count, budget), replace=False))
                selection = slice(None) if budget is None else chosen
                # Full mode uses a contiguous read; optional sampling reads selected rows.
                result = np.asarray(f["features"][selection], dtype=np.float32), np.asarray(f["coords"][selection], dtype=np.float32)
        x, coords = result
        if x.ndim != 2 or len(x) == 0 or coords.shape != (len(x), 2):
            raise ValueError(f"Malformed feature/coordinate arrays: {path}")
        if not np.isfinite(x).all() or not np.isfinite(coords).all():
            raise ValueError(f"Nonfinite features: {path}")
        return x, coords, chosen

    def __getitem__(self, index):
        row = self.rows[index]
        rng = self._rng(row["patient_id"])
        paths = [str(Path(p) if Path(p).is_absolute() else self.expression.root / p) for p in row["feature_paths"]]
        # Allocate a fair budget per slide; keep slide identity for spatial neighbors.
        budget = max(1, self.max_patches // len(paths)) if self.max_patches else None
        xs, cs, slide_ids, source_rows = [], [], [], []
        for sid, path in enumerate(paths):
            x, coord, patch_rows = self._load(path, budget, rng)
            coord = (coord - coord.min(0)) / np.maximum(coord.max(0) - coord.min(0), 1)
            xs.append(x); cs.append(coord); slide_ids.extend([sid] * len(x))
            source_rows.append(patch_rows)
        x, coord = np.concatenate(xs), np.concatenate(cs)
        source_rows = np.concatenate(source_rows)
        slide_ids = np.asarray(slide_ids, dtype=np.int64)
        if self.max_patches and len(x) > self.max_patches:
            chosen = np.sort(rng.choice(len(x), self.max_patches, replace=False))
            x, coord, slide_ids = x[chosen], coord[chosen], slide_ids[chosen]
            source_rows = source_rows[chosen]
        neighbors = build_spatial_index(coord, slide_ids, self.spatial_k)
        pathway, mask = self.expression.transform(row)
        if self.augment_style:
            style_states, style_scores = vectorize(extract_evidence(rewrite_style(row["report"])))
        else:
            style_states, style_scores = row["states"], row["scores"]
        return {"patient_id": row["patient_id"], "x": torch.from_numpy(x),
                "feature_paths": paths, "patch_rows": torch.from_numpy(source_rows),
                "all_patches": not bool(self.max_patches),
                "neighbor_index": torch.from_numpy(neighbors),
                "coords": torch.from_numpy(coord), "slide_ids": torch.from_numpy(slide_ids),
                "pathways": torch.from_numpy(pathway), "pathway_mask": torch.from_numpy(mask),
                "states": torch.tensor(row["states"], dtype=torch.long),
                "scores": torch.tensor(row["scores"], dtype=torch.float32),
                "style_states": torch.tensor(style_states, dtype=torch.long),
                "style_scores": torch.tensor(style_scores, dtype=torch.float32),
                "time": torch.tensor(row["time"], dtype=torch.float32),
                "event": torch.tensor(row["event"], dtype=torch.float32)}


def collate_patients(items: list[dict]):
    n = max(len(item["x"]) for item in items)
    d = items[0]["x"].shape[-1]
    batch = {"patient_id": [item["patient_id"] for item in items],
             "feature_paths": [item["feature_paths"] for item in items],
             "all_patches": [item["all_patches"] for item in items],
             "x": torch.zeros(len(items), n, d),
             "coords": torch.zeros(len(items), n, 2),
             "slide_ids": torch.full((len(items), n), -1, dtype=torch.long),
             "patch_rows": torch.full((len(items), n), -1, dtype=torch.long),
             "neighbor_index": torch.full((len(items), n, items[0]["neighbor_index"].shape[-1]), -1, dtype=torch.long),
             "patch_mask": torch.zeros(len(items), n, dtype=torch.bool)}
    for i, item in enumerate(items):
        count = len(item["x"])
        for key in ["x", "coords", "slide_ids", "patch_rows", "neighbor_index"]:
            batch[key][i, :count] = item[key]
        batch["patch_mask"][i, :count] = True
    for key in ["pathways", "pathway_mask", "states", "scores", "style_states", "style_scores", "time", "event"]:
        batch[key] = torch.stack([item[key] for item in items])
    return batch


def to_device(batch: dict, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
