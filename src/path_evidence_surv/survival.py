from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_cutpoints(times, events, num_bins):
    times, events = np.asarray(times), np.asarray(events)
    observed = times[events.astype(bool)]
    if len(observed) < 2 or num_bins < 2:
        raise ValueError("At least two training events and two time bins are required")
    return np.unique(np.quantile(observed, np.arange(1, num_bins) / num_bins)).astype(np.float32)


def discrete_survival_nll(logits, times, events, cutpoints):
    """Events: survival through previous bins, then event in current bin.

    Censoring: only fully observed finite bins count; no forced survival to the
    end of the partially observed censoring bin. Last interval is open-ended.
    """
    cuts = torch.as_tensor(cutpoints, device=times.device, dtype=times.dtype).contiguous()
    event_bin = torch.searchsorted(cuts, times.contiguous(), right=False)
    completed = torch.searchsorted(cuts, times.contiguous(), right=True)
    survived_bins = torch.where(events.bool(), event_bin, completed)
    idx = torch.arange(logits.shape[-1], device=logits.device)[None, :]
    survive_loss = (-F.logsigmoid(-logits) * (idx < survived_bins[:, None])).sum(-1)
    event_loss = -F.logsigmoid(logits).gather(1, event_bin[:, None]).squeeze(1) * events
    return (survive_loss + event_loss).mean()


def concordance_index(times, events, risk):
    t, e, r = map(np.asarray, (times, events, risk))
    comparable = (t[:, None] < t[None, :]) & e[:, None].astype(bool)
    count = int(comparable.sum())
    if not count:
        return None
    score = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
    return float(score[comparable].mean())


class CensoringKM:
    def __init__(self, times, events):
        times, events = np.asarray(times), np.asarray(events)
        self.times = np.unique(times)
        probability = 1.0
        values = []
        for time in self.times:
            at_risk = int((times >= time).sum())
            censor = int(((times == time) & (events == 0)).sum())
            probability *= 1.0 - censor / at_risk
            values.append(probability)
        self.values = np.asarray(values)

    def at(self, time, left=False):
        x = np.asarray(time)
        index = np.searchsorted(self.times, x, side="left" if left else "right") - 1
        return np.where(index >= 0, self.values[np.maximum(index, 0)], 1.0)


def survival_at(survival, cutpoints, horizons):
    # Predictions at finite upper bin boundaries; before first boundary S=1.
    indices = np.searchsorted(cutpoints, horizons, side="right") - 1
    return np.stack([np.ones(len(survival)) if i < 0 else survival[:, i] for i in indices], axis=1)


def ipcw_scores(train_times, train_events, times, events, predictions, horizons, min_g=0.05):
    km = CensoringKM(train_times, train_events)
    times, events = np.asarray(times), np.asarray(events)
    result = []
    for j, horizon in enumerate(horizons):
        g = float(km.at(horizon))
        if g < min_g:
            result.append({"time": float(horizon), "brier": None, "auc": None,
                           "reason": "Insufficient training censoring support"})
            continue
        cases = (times <= horizon) & (events == 1)
        controls = times > horizon
        g_event = km.at(times, left=True)
        # Unsupported event weights are excluded by refusing this horizon.
        if np.any(g_event[cases] < min_g):
            result.append({"time": float(horizon), "brier": None, "auc": None,
                           "reason": "Insufficient event-time censoring support"})
            continue
        w_case = cases / np.maximum(g_event, min_g)
        w_control = controls / g
        predicted_s = predictions[:, j]
        brier = float(np.mean(w_case * predicted_s**2 + w_control * (1-predicted_s)**2))
        auc = None
        if cases.any() and controls.any():
            r = 1 - predicted_s
            pair_score = (r[cases, None] > r[None, controls]) + 0.5 * (r[cases, None] == r[None, controls])
            weights = w_case[cases, None] * w_control[None, controls]
            auc = float((pair_score * weights).sum() / weights.sum())
        calibration = []
        risk = 1 - predicted_s
        for low, high in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
            chosen = (risk >= low) & ((risk <= high) if high == 1 else (risk < high))
            denom = (w_case[chosen] + w_control[chosen]).sum()
            if chosen.any():
                calibration.append({"n": int(chosen.sum()), "predicted_risk": float(risk[chosen].mean()),
                                    "ipcw_observed_risk": float(w_case[chosen].sum()/denom) if denom > 0 else None})
        result.append({"time": float(horizon), "brier": brier, "auc": auc,
                       "case_count": int(cases.sum()), "control_count": int(controls.sum()),
                       "calibration": calibration})
    # Do not bridge unsupported time gaps when integrating Brier scores.
    area, width = 0.0, 0.0
    for left, right in zip(result[:-1], result[1:]):
        if left["brier"] is not None and right["brier"] is not None:
            dt = right["time"]-left["time"]
            area += dt*(left["brier"]+right["brier"])*0.5; width += dt
    return {"horizon_metrics": result, "ibs": area/width if width > 0 else None,
            "ibs_integrated_width": width, "censoring_estimator_source": "training_only"}


def bootstrap_cindex(times, events, risk, repeats=0, seed=42):
    if repeats <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = rng.choice(len(times), len(times), replace=True)
        value = concordance_index(np.asarray(times)[idx], np.asarray(events)[idx], np.asarray(risk)[idx])
        if value is not None:
            values.append(value)
    return {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)),
            "valid_replicates": len(values)} if values else None
