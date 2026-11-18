"""CPU validation only. Never creates a DGX run or downloads model weights."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'vendor'))
import core
import torch
import transformers
import nbformat
from protocol import atomic_json


def report_fixture(root):
    """Synthetic renderer smoke test; all fixture outputs stay in a temp directory."""
    from protocol import jobs, sha
    from report import build
    config = json.loads((PACKAGE / 'config.json').read_text())
    for name in ['metrics', 'recipes', 'reports', 'models', 'logs', 'state']:
        (root / name).mkdir(exist_ok=True)
    atomic_json(root / 'state/protocol.json', {'scope': 'SYNTHETIC LOCAL RENDER TEST ONLY'})
    for name in ['tests', 'smoke']:
        atomic_json(root / f'state/{name}.json', {'status': 'PASS', 'scope': 'SYNTHETIC'})
    atomic_json(root / 'metrics/teacher.json', {'parameters': 3000, 'test': {'perplexity': 10.}, 'piqa': {'accuracy': .8}})
    atomic_json(root / 'metrics/pruning_scores.json', {'single_layer_delta_nll': [.1] * 36,
                'layer_squared_l2': [1.] * 36, 'channel_quantiles': [[float(i) for i in range(11)]] * 36})
    for i, job in enumerate(jobs(config)):
        model = root / 'models' / job['id']
        model.mkdir()
        artifact = model / 'SYNTHETIC.txt'
        artifact.write_text('Not model weights')
        atomic_json(model / 'EXPORT_COMPLETE.json', {artifact.name: sha(artifact)})
        atomic_json(root / f'metrics/{job["id"]}.json', {
            'parameters': 2500 if job['level'] == 'moderate' else 2000,
            'test': {'perplexity': 11. + i / 10, 'nll': 2.4 + i / 100}, 'piqa': {'accuracy': .7},
            'recovery': [{'tokens': 0, 'validation': {'nll': 3.}}, {'tokens': 1966080, 'validation': {'nll': 2.5}}],
            'input_tokens': 1536 * 128 * config['epochs'], 'training_seconds': 1.,
            'training_tokens_per_second': 1., 'training_memory': {'peak_allocated_gib': 1.},
            'reload_verified': True})
    for variant in ['teacher', *config['benchmark_jobs']]:
        for device in ['cpu', 'cuda']:
            atomic_json(root / f'metrics/benchmark_{variant}_{device}.json', {
                'quality_subset': {'test': {'perplexity': 10.}, 'piqa': {'accuracy': .7}},
                'cases': [{'prompt_tokens': n, 'median_seconds': {'model': 2. if device == 'cpu' else 1.,
                          'e2e': 2.2 if device == 'cpu' else 1.2}} for n in [32, 128, 512]]})
    for device in ['cpu', 'cuda']:
        atomic_json(root / f'metrics/tiny_{device}.json', {'benchmark': {
            'matched_weights_sha256': 'synthetic',
            'cases': [{'prompt_tokens': n, 'median_seconds': .1} for n in [1, 8, 32, 128, 512]]}})
    build(root, config)
    assert (root / 'RESULTS.zip').exists()
    for name in ['REPORT.md', 'compression_quality.png', 'recovery.png', 'pruning_maps.png', 'cpu_gpu_scaling.png']:
        assert (root / 'reports' / name).stat().st_size > 0


def main():
    import test_package
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_package))
    if not result.wasSuccessful():
        raise RuntimeError('Packaging tests failed')
    for path in PACKAGE.glob('*.ipynb'):
        nbformat.validate(nbformat.read(path, as_version=4))
    temp_parent = '/tmp/opencode' if Path('/tmp/opencode').exists() else None
    with tempfile.TemporaryDirectory(prefix='dgx-local-', dir=temp_parent) as d:
        root = Path(d)
        (root / 'state').mkdir()
        shutil.copytree(PACKAGE / 'data', root / 'data')
        core.ROOT = root
        core.C['cpu_threads'] = 2
        torch.set_num_threads(2)
        import runtime_tests
        runtime_tests.run(root)
        runtime = json.loads((root / 'state/tests.json').read_text())
    with tempfile.TemporaryDirectory(prefix='dgx-synthetic-report-', dir=temp_parent) as d:
        report_fixture(Path(d))
    atomic_json(PACKAGE / 'LOCAL_VALIDATION.json', {
        'status': 'PASS_CPU_AND_PACKAGE_TESTS_ONLY',
        'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'package_tests': result.testsRun, 'runtime_tests': runtime,
        'notebook_schema_checks': 10,
        'report_render_test': 'PASS_SYNTHETIC_FIXTURE_ONLY_NOT_DGX_RESULTS',
        'not_tested': ['B200 execution/VRAM fit', '3B full-weight training', 'DGX quality and timing',
                       'TensorRT conversion/build/runtime API compatibility']})


if __name__ == '__main__':
    main()
