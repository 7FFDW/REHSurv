import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('release_main', ROOT / 'main.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
from path_evidence_surv.evidence import extract_evidence
from path_evidence_surv.data import PatientDataset


class RegressionTests(unittest.TestCase):
    def states(self, text):
        assertions = extract_evidence(text)
        for a in assertions:
            for span in a['spans']:
                self.assertEqual(text[span['start']:span['end']], span['text'])
        return {a['concept']: a['state'] for a in assertions}

    def test_coordinated_negation(self):
        for text in ['No necrosis, lymphovascular invasion, or calcification.',
                     'No necrosis and lymphovascular invasion and calcification.',
                     'Negative for necrosis, lymphovascular invasion or calcification.']:
            s = self.states(text)
            for name in ['necrosis', 'lymphovascular_invasion', 'calcification']:
                self.assertEqual(s[name], 'absent', (text, name))

    def test_negation_stops_at_new_assertion(self):
        for text in ['No necrosis, but calcification.',
                     'No necrosis and calcification is present.',
                     'No necrosis, calcification present.',
                     'No necrosis. Calcification.']:
            s = self.states(text)
            self.assertEqual(s['necrosis'], 'absent')
            self.assertEqual(s['calcification'], 'present', text)
        self.assertEqual(self.states('No necrosis and possible calcification.')['calcification'], 'uncertain')
        self.assertEqual(self.states('Necrosis present. No necrosis.')['necrosis'], 'uncertain')

    def test_validation_comparability(self):
        for rows in [[], [{'time': 1, 'event': 0}, {'time': 2, 'event': 0}],
                     [{'time': 1, 'event': 1}, {'time': 1, 'event': 1}],
                     [{'time': 1, 'event': 0}, {'time': 2, 'event': 1}]]:
            with self.assertRaisesRegex(ValueError, 'no comparable'):
                main.validate_comparable_validation(rows)
        main.validate_comparable_validation([{'time': 1, 'event': 1}, {'time': 2, 'event': 0}])

    def test_uppercase_npz(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'slide.NPZ'
            with path.open('wb') as stream:
                np.savez(stream, features=np.ones((3, 4)), coords=np.zeros((3, 2)))
            ds = PatientDataset([], None)
            x, coords, indices = ds._load(str(path), None, np.random.default_rng(42))
            self.assertEqual(x.shape, (3, 4))
            self.assertEqual(coords.shape, (3, 2))

    def test_checkpoint_data_directory_defaults(self):
        args = main.get_args(['--test_only', '--checkpoint', 'best.pt'])
        self.assertIsNone(args.data_dir)
        args = main.get_args(['--test_only', '--checkpoint', 'best.pt', '--data-dir', 'custom'])
        self.assertEqual(args.data_dir, 'custom')
        self.assertIsNotNone(main.get_args([]).data_dir)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps({'data_dir': 'from_config'}))
            args = main.get_args(['--test_only', '--checkpoint', 'best.pt', '--config', str(path)])
            self.assertEqual(args.data_dir, 'from_config')


if __name__ == '__main__':
    unittest.main()
