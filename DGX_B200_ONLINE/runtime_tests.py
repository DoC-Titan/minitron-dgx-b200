"""Meaningful CPU tests for KL, physical removal, reload, full-weight resume."""
import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
import core
from tests import tiny_model, CorrectnessTests


class FullWeightResume(unittest.TestCase):
    def test_data_roles_are_disjoint(self):
        root = core.ROOT / 'data'
        if not (root / 'train.pt').exists():
            self.skipTest('Bundled input tensors not copied to test root')
        groups = {'train': torch.load(root / 'train.pt', weights_only=True),
                  'calibration': torch.load(root / 'calibration.pt', weights_only=True)}
        groups.update(torch.load(root / 'evaluation.pt', weights_only=True))
        seen = {}
        for name, blocks in groups.items():
            self.assertEqual(blocks.ndim, 2)
            self.assertEqual(blocks.shape[1], 128)
            hashes = {core.tensor_hash(row) for row in blocks}
            for previous, values in seen.items():
                self.assertFalse(hashes & values, f'Exact block overlap: {name}/{previous}')
            seen[name] = hashes

    def test_optimizer_resume_and_frozen_teacher(self):
        teacher = tiny_model(1).eval().requires_grad_(False)
        m = tiny_model(2).train()
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=False)
        x = torch.arange(16).reshape(1, -1)
        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            z = model(x, use_cache=False).logits
            with torch.no_grad():
                target = teacher(x, use_cache=False).logits
            loss = core.full_kl(z[:, :-1], target[:, :-1], 2)
            loss.backward()
            optimizer.step()
        before = m.model.layers[0].mlp.up_proj.weight.detach().clone()
        step(m, o)
        self.assertFalse(torch.equal(before, m.model.layers[0].mlp.up_proj.weight))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'resume.pt'
            value = {'model': m.state_dict(), 'optimizer': o.state_dict()}
            core.atomic_torch(path, value)
            core.atomic_torch(path, value, backup=True)
            path.write_bytes(b'interrupted write')
            state = core.checked_load(path, backup=True)
            r = tiny_model(2).train()
            r.load_state_dict(state['model'])
            ro = torch.optim.AdamW(r.parameters(), lr=1e-3, foreach=False)
            ro.load_state_dict(state['optimizer'])
            step(m, o)
            step(r, ro)
            for a, b in zip(m.parameters(), r.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def run(root):
    suite = unittest.TestSuite([
        CorrectnessTests('test_kl_direction_temperature_and_gradient'),
        CorrectnessTests('test_physical_shapes_and_round_trip'),
        FullWeightResume('test_data_roles_are_disjoint'),
        FullWeightResume('test_optimizer_resume_and_frozen_teacher')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    core.atomic_json(root / 'state/tests.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
                                              'tests': result.testsRun})
    if not result.wasSuccessful():
        raise RuntimeError('Correctness tests failed')
