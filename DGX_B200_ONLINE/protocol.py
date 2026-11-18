"""Pure-Python protocol helpers, also used by local tests."""
import hashlib
import itertools
import json
import os
from pathlib import Path


def jobs(config):
    return [dict(id=f'{level}_{criterion}_s{seed}_{objective}', level=level,
                 criterion=criterion, seed=seed, objective=objective)
            for level, criterion, seed, objective in itertools.product(
                config['structures'], config['criteria'], config['seeds'], config['objectives'])]


def recipe_name(job):
    return f'{job["level"]}_{job["criterion"]}_s{job["seed"]}.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def validate_config(c):
    assert c['epochs'] > 0 and c['batch_size'] > 0 and c['accumulation'] > 0
    assert c['checkpoint_updates'] > 0 and c['temperature'] > 0
    assert c['repeats'] >= 3 and c['warmups'] >= 1
    assert len(c['seeds']) >= 3 and len(set(c['seeds'])) == len(c['seeds'])
    assert set(c['criteria']) <= {'random', 'magnitude', 'sensitivity'}
    assert set(c['objectives']) <= {'ce', 'kl'}
    for size in c['structures'].values():
        assert 0 < size['layers'] < 36 and 0 < size['ffn'] < 11008
    assert set(c['benchmark_jobs']) <= {j['id'] for j in jobs(c)}
