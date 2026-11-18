"""Notebook-friendly isolated worker launcher. GPU selection precedes torch import."""
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent


def run(stage, *args):
    config = json.loads((PACKAGE / 'config.json').read_text())
    env = os.environ.copy()
    if config['gpu_uuid']:
        env['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    command = [sys.executable, '-u', str(PACKAGE / 'engine.py'), stage, *map(str, args)]
    subprocess.run(command, cwd=PACKAGE, env=env, check=True)


def train_all():
    from protocol import jobs
    config = json.loads((PACKAGE / 'config.json').read_text())
    for job in jobs(config):
        run('train', '--job', job['id'])


def benchmarks():
    config = json.loads((PACKAGE / 'config.json').read_text())
    for job in ['teacher', *config['benchmark_jobs']]:
        for device in ['cpu', 'cuda']:
            run('benchmark', '--job', job, '--device', device)
