"""Adapt a patient CSV, expression NPZ, and pathway JSON for training. No split is regenerated."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from path_evidence_surv.evidence import extract_evidence, vectorize, EVIDENCE_SCHEMA
from path_evidence_surv.common import write_jsonl, write_json, digest


def prepare(manifest, expression, pathways, output, feature_dim):
    feature_owners = {}
    manifest, expression, pathways = map(lambda p: Path(p).resolve(), (manifest, expression, pathways))
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to avoid overwriting data')
    with np.load(expression, allow_pickle=False) as a:
        ids, genes = a['patient_ids'].astype(str), a['genes'].astype(str)
        values = np.asarray(a['values'], dtype=np.float32)
        measured = np.asarray(a['measured'], dtype=bool) if 'measured' in a else np.ones(len(genes), dtype=bool)
    if values.shape != (len(ids), len(genes)) or measured.shape != (len(genes),):
        raise ValueError('Expression shapes must be values[N,G], patient_ids[N], genes[G], measured[G]')
    if len(set(ids)) != len(ids) or len(set(genes)) != len(genes):
        raise ValueError('Duplicate expression patient IDs or gene names')
    if not np.isfinite(values[:, measured]).all():
        raise ValueError('Measured expression contains NaN/Inf; resolve missing measurements explicitly')
    values[:, ~measured] = 0
    sets = json.loads(pathways.read_text(encoding='utf-8-sig'))
    if not sets or not isinstance(sets, dict) or not all(isinstance(v, list) for v in sets.values()):
        raise ValueError('Pathways JSON must map pathway names to gene-name lists')
    names = list(sets)
    gene_sets = {p: set(sets[p]) for p in names}
    membership = np.asarray([[g in gene_sets[p] for g in genes] for p in names], dtype=np.float32)
    membership *= measured[None, :]
    if not membership.any():
        raise ValueError('No measured genes overlap the pathway definitions; check gene identifiers')
    lookup = {p: i for i, p in enumerate(ids)}
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError('Empty patient CSV')
    required = {'patient_id', 'cohort', 'fold', 'time', 'event', 'time_unit', 'feature_paths', 'report'}
    if not required <= raw[0].keys():
        raise ValueError(f'Missing CSV columns: {sorted(required - raw[0].keys())}')
    rows, seen = [], set()
    for r in raw:
        pid = r['patient_id'].strip()
        if not pid or pid in seen or pid not in lookup:
            raise ValueError(f'Duplicate, empty, or unmatched patient ID: {pid}')
        seen.add(pid)
        fold, event, time = int(r['fold']), int(r['event']), float(r['time'])
        if fold not in range(5) or event not in (0, 1) or not np.isfinite(time) or time < 0:
            raise ValueError(f'Invalid fold/event/time: {pid}')
        if r['time_unit'] not in ('days', 'months', 'years'):
            raise ValueError(f'Explicit verified time_unit required: {pid}')
        paths = json.loads(r['feature_paths'])
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError(f'feature_paths must be a nonempty JSON string list: {pid}')
        resolved = []
        for name in paths:
            p = Path(name)
            p = p.resolve() if p.is_absolute() else (manifest.parent / p).resolve()
            # Prevent the same slide from appearing under multiple patients.
            if str(p) in feature_owners and feature_owners[str(p)] != pid:
                raise ValueError(f'Slide shared by different patients: {p}')
            feature_owners[str(p)] = pid
            if p.suffix.lower() == '.npz':
                with np.load(p, allow_pickle=False) as a:
                    x, coords = a['features'], a['coords']
            elif p.suffix.lower() in ('.h5', '.hdf5'):
                import h5py
                with h5py.File(p, 'r') as a:
                    x, coords = a['features'][:], a['coords'][:]
            else:
                raise ValueError(f'Expected .npz/.h5/.hdf5 feature file: {p}')
            if x.ndim != 2 or not len(x) or x.shape[1] != feature_dim or coords.shape != (len(x), 2):
                raise ValueError(f'Invalid feature/coordinate shape: {p}')
            if not np.isfinite(x).all() or not np.isfinite(coords).all():
                raise ValueError(f'Nonfinite features/coordinates: {p}')
            resolved.append(str(p))
        if len(set(resolved)) != len(resolved):
            raise ValueError(f'Duplicate slide in patient {pid}')
        assertions = extract_evidence(r['report'])
        states, scores = vectorize(assertions)
        rows.append(dict(patient_id=pid, cohort=r['cohort'], fold=fold, event=event, time=time,
                         time_unit=r['time_unit'], tss=r.get('tss', ''), feature_paths=resolved,
                         expression_archive='expression.npz', expression_index=lookup[pid],
                         report=r['report'], has_report=bool(r['report'].strip()), assertions=assertions, states=states, scores=scores,
                         evidence_schema=EVIDENCE_SCHEMA))
    for cohort in {r['cohort'] for r in rows}:
        units = {r['time_unit'] for r in rows if r['cohort'] == cohort}
        if len(units) != 1:
            raise ValueError(f'Mixed time units in {cohort}')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'expression.npz', patient_ids=ids, genes=genes, values=values,
                        pathways=np.asarray(names), membership=membership, measured=measured)
    write_jsonl(output / 'patients.jsonl', rows)
    write_json(output / 'provenance.json', {str(p.name): digest(p) for p in (manifest, expression, pathways)})
    print(f'Prepared {len(rows)} patients, {len(names)} pathways, {len(genes)} genes in {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--expression', required=True)
    p.add_argument('--pathways', required=True)
    p.add_argument('--output', default='data/prepared')
    p.add_argument('--feature-dim', type=int, default=1536)
    a = p.parse_args()
    prepare(a.manifest, a.expression, a.pathways, a.output, a.feature_dim)
