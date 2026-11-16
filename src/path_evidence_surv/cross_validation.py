import json
import math
import statistics


def fold_record(fold, run_dir):
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    if metrics.get('split') != 'test' or metrics.get('smoke_only') is not False:
        raise ValueError(f'Fold {fold} did not produce formal held-out test metrics.')
    auc = [h['auc'] for h in metrics['horizon_metrics'] if h.get('auc') is not None]
    return {'fold': fold, 'run_dir': str(run_dir), 'best_epoch': metrics['best_epoch'],
            'test_patients': metrics['patients'], 'test_events': metrics['events'],
            'c_index': metrics['c_index'], 'ibs': metrics['ibs'],
            'auc_mean_supported': statistics.mean(auc) if auc else None}


def aggregate(records):
    result = {}
    for key in ['c_index', 'ibs', 'auc_mean_supported']:
        values = [r[key] for r in records if r[key] is not None and math.isfinite(r[key])]
        result[key] = {'mean': statistics.mean(values) if values else None,
                       'sample_std': statistics.stdev(values) if len(values) > 1 else None,
                       'valid_folds': len(values)}
    return result
