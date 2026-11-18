"""Local packaging tests; no GPU, torch, or research run required."""
import ast
import json
from pathlib import Path
import unittest
from protocol import jobs, recipe_name, sha, validate_config

ROOT = Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_complete_balanced_matrix(self):
        c = json.loads((ROOT / 'config.json').read_text())
        validate_config(c)
        matrix = jobs(c)
        self.assertEqual(len(matrix), 36)
        self.assertEqual(len({j['id'] for j in matrix}), 36)
        self.assertEqual(len({recipe_name(j) for j in matrix}), 18)
        for j in matrix:
            pair = [x for x in matrix if recipe_name(x) == recipe_name(j)]
            self.assertEqual({x['objective'] for x in pair}, {'ce', 'kl'})

    def test_pinned_inputs_and_vendor_source(self):
        manifest = json.loads((ROOT / 'UPSTREAM.json').read_text())
        for name, expected in manifest['files'].items():
            self.assertEqual(sha(ROOT / name), expected, name)

    def test_notebooks_and_python_parse(self):
        notebooks = sorted(ROOT.glob('*.ipynb'))
        self.assertEqual(len(notebooks), 10)
        for path in notebooks:
            n = json.loads(path.read_text())
            self.assertEqual(n['nbformat'], 4)
            for cell in n['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']), filename=path.name)
                    self.assertEqual(cell['outputs'], [])
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
