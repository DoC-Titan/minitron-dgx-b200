"""New DGX study: full-weight BF16-autocast recovery with an online teacher.

Each command runs in a separate process. Existing research checkpoints are read
only; all generated artifacts belong to config.run_dir.
"""
import argparse
import contextlib
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

from protocol import atomic_json, jobs, recipe_name, sha, validate_config

PACKAGE = Path(__file__).resolve().parent
C = json.loads((PACKAGE / 'config.json').read_text())
validate_config(C)
ROOT = Path(C['run_dir'])
sys.path.insert(0, str(PACKAGE / 'vendor'))
import core
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse audited physical slicing and scoring, without the old 8 GiB policy.
core.ROOT = ROOT
core.C.update(cpu_threads=C['cpu_threads'], seeds=C['seeds'], temperature=C['temperature'])
core.TEACHER = Path(C['teacher'])
core.guard = lambda *args, **kwargs: None
core.configure = lambda device: seed(42)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.set_num_threads(C['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device='cuda'):
    if device == 'cuda':
        torch.cuda.synchronize()


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()


def load(path=None, device='cuda', train=False):
    dtype = torch.float32 if train or device == 'cpu' else torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(
        path or C['teacher'], local_files_only=True, torch_dtype=dtype,
        device_map={'': device}, low_cpu_mem_usage=True, attn_implementation='sdpa')
    m.requires_grad_(train).eval()
    assert all(p.device.type == device for p in m.parameters())
    return m


def tok():
    return AutoTokenizer.from_pretrained(C['teacher'], local_files_only=True)


def data(name):
    return torch.load(ROOT / 'data' / f'{name}.pt', map_location='cpu', weights_only=True)


def task_records():
    return json.loads((ROOT / 'data/piqa.json').read_text())['records']


def memory(device='cuda'):
    d = {'rss_gib': psutil.Process().memory_info().rss / 2**30}
    if device == 'cuda':
        d.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                 board_sample=core.gpu_sample())
    return d


def setup():
    for folder in ['data', 'recipes', 'metrics', 'state', 'models', 'reports', 'logs']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    upstream = json.loads((PACKAGE / 'UPSTREAM.json').read_text())
    for name, expected in upstream['files'].items():
        assert sha(PACKAGE / name) == expected, f'Bundled source/data changed: {name}'
    for p in (PACKAGE / 'data').iterdir():
        destination = ROOT / 'data' / p.name
        if destination.exists():
            assert sha(destination) == sha(p), f'Existing data differ: {destination}'
        else:
            shutil.copy2(p, destination)
    teacher = Path(C['teacher'])
    cfg = json.loads((teacher / 'config.json').read_text())
    assert (cfg['num_hidden_layers'], cfg['hidden_size'], cfg['intermediate_size']) == (36, 2048, 11008)
    teacher_files = {p.name: sha(p) for p in teacher.iterdir()
                     if p.suffix in {'.safetensors', '.json'}}
    assert any(n.endswith('.safetensors') for n in teacher_files), 'Teacher weights missing'
    provenance = teacher / 'DOWNLOAD_RECORD.json'
    if provenance.exists():
        receipt = json.loads(provenance.read_text())
        assert receipt['revision'] == C['teacher_revision']
        for name, entry in receipt['files'].items():
            assert sha(teacher / name) == entry['sha256'], f'Teacher differs: {name}'
    else:
        raise RuntimeError('Teacher DOWNLOAD_RECORD.json missing. Run models/download_teacher.py from the earlier setup to verify the pinned snapshot.')
    lock = {'config': C, 'teacher': teacher_files,
            'source': {str(p.relative_to(PACKAGE)): sha(p) for p in scientific_files()},
            'data': {p.name: sha(p) for p in (ROOT / 'data').iterdir()},
            'packages': versions()}
    target = ROOT / 'state/protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == lock, 'Protocol changed; choose a new run_dir'
    else:
        atomic_json(target, lock)
    assert torch.cuda.is_available(), 'This main study requires CUDA'
    assert torch.cuda.is_bf16_supported(), 'BF16 CUDA support required'
    x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all()
    info = {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
            'cpu_threads': C['cpu_threads'], 'cpu_count': os.cpu_count(),
            'ram_gib': psutil.virtual_memory().total / 2**30,
            'available_ram_gib': psutil.virtual_memory().available / 2**30,
            'free_disk_gib': shutil.disk_usage(ROOT).free / 2**30,
            'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
            'capability': torch.cuda.get_device_capability(),
            'visible_vram_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
            'packages': versions(), 'seeds': C['seeds'], 'teacher_revision': C['teacher_revision'],
            'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
            'memory_policy': 'No artificial allocator cap; selected visible GPU only',
            'precision': 'FP32 trainable student/AdamW states; BF16 autocast; frozen BF16 teacher',
            'data_scope': 'Pinned earlier RTX corpus, newly measured DGX results. Not prior notebook validation tensors.',
            'framework': 'PyTorch SDPA; custom physical slicing; no ModelOpt/NeMo; RAPIDS N/A'}
    atomic_json(ROOT / 'reports/environment.json', info)
    print(json.dumps(info, indent=2))
    print('Jobs:', len(jobs(C)), '| Input tokens per job:', len(data('train')) * 128 * C['epochs'])


def versions():
    out = {}
    for p in ['torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'psutil', 'tensorrt_llm', 'nvidia-modelopt', 'nemo_toolkit']:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = 'not installed'
    return out


def scientific_files():
    return sorted([*PACKAGE.glob('*.py'), *PACKAGE.glob('vendor/*.py'), PACKAGE / 'vendor/config.json'])


def check_lock():
    lock = json.loads((ROOT / 'state/protocol.json').read_text())
    assert C == lock['config'], 'Configuration changed: use a new run directory'
    assert versions() == lock['packages'], 'Package versions changed since setup'
    for name, expected in lock['source'].items():
        assert sha(PACKAGE / name) == expected, f'Source changed: {name}'
    for name, expected in lock['data'].items():
        assert sha(ROOT / 'data' / name) == expected, f'Data changed: {name}'
    # Full teacher hashing is done in setup; avoid reading 6 GB before every worker.


def recipes():
    output = ROOT / 'metrics/pruning_scores.json'
    receipt_path = ROOT / 'state/recipes_complete.json'
    if receipt_path.exists():
        for name, expected in json.loads(receipt_path.read_text()).items():
            assert sha(ROOT / name) == expected, f'Recipe artifact changed: {name}'
        print('Pruning scoring and recipes already complete')
        return
    model = load()
    n = len(model.model.layers)
    cal = data('calibration')
    start = time.perf_counter()
    with torch.no_grad():
        layer_scores = [sum(float(p.float().square().sum()) for p in layer.parameters())
                        for layer in model.model.layers]
        channels = []
        for layer in model.model.layers:
            mlp = layer.mlp
            channels.append((mlp.gate_proj.weight.float().square().sum(1)
                             + mlp.up_proj.weight.float().square().sum(1)
                             + mlp.down_proj.weight.float().square().sum(0)).cpu())
    baseline = core.nll(model, cal, 'cuda')
    ablations = []
    original = list(model.model.layers)
    for index in range(n):
        model.model.layers = nn.ModuleList([l for i, l in enumerate(original) if i != index])
        core.fix_layers(model)
        measured = core.nll(model, cal, 'cuda')
        ablations.append(measured['nll'] - baseline['nll'])
        print('Layer', index, 'calibration delta NLL', ablations[-1], flush=True)
    model.model.layers = nn.ModuleList(original)
    core.fix_layers(model)
    for job in jobs(C):
        path = ROOT / 'recipes' / recipe_name(job)
        size = C['structures'][job['level']]
        count = n - size['layers']
        if job['criterion'] == 'random':
            removed = sorted(random.Random(job['seed']).sample(range(n), count))
        else:
            scores = layer_scores if job['criterion'] == 'magnitude' else ablations
            removed = sorted(sorted(range(n), key=lambda i: (scores[i], i))[:count])
        kept = [i for i in range(n) if i not in removed]
        selected = []
        for i in kept:
            if job['criterion'] == 'random':
                ids = torch.randperm(len(channels[i]), generator=torch.Generator().manual_seed(job['seed'] + i))[:size['ffn']]
            else:
                ids = torch.argsort(channels[i], descending=True, stable=True)[:size['ffn']]
            selected.append({'original_layer': i, 'retained': sorted(ids.tolist())})
        value = {'retained_layers': kept, 'removed_layers': removed, 'ffn': size['ffn'],
                 'channels': selected, 'criterion': job['criterion'], 'seed': job['seed'],
                 'teacher_revision': C['teacher_revision'],
                 'note': 'Arbitrary layer subsets, including boundaries. Sensitivity ranks single-layer calibration ablations; joint optimum is not guaranteed.'}
        if path.exists():
            assert json.loads(path.read_text()) == value, 'Recipe mismatch; do not overwrite'
        else:
            atomic_json(path, value)
    atomic_json(output, {'baseline': baseline, 'layer_squared_l2': layer_scores,
                        'single_layer_delta_nll': ablations,
                        'channel_quantiles': [torch.quantile(x, torch.linspace(0, 1, 11)).tolist() for x in channels],
                        'seconds': time.perf_counter() - start})
    artifacts = [output, *sorted((ROOT / 'recipes').glob('*.json'))]
    atomic_json(receipt_path, {str(p.relative_to(ROOT)): sha(p) for p in artifacts})


def recipe(job):
    return json.loads((ROOT / 'recipes' / recipe_name(job)).read_text())


def student(job, train=False):
    model = load(train=False)
    core.apply_recipe(model, recipe(job))
    if train:
        model.float().requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    return model


def score(model, device='cuda', subset=False):
    blocks = data('evaluation')['test']
    records = task_records()
    if subset:
        blocks, records = blocks[:16], records[:32]
    return {'test': core.nll(model, blocks, device),
            'piqa': core.piqa(model, tok(), records, device),
            'parameters': sum(p.numel() for p in model.parameters()), 'memory': memory(device)}


def teacher():
    p = ROOT / 'metrics/teacher.json'
    if p.exists():
        return
    model = load()
    result = score(model)
    result['validation'] = core.nll(model, data('evaluation')['validation'], 'cuda')
    atomic_json(p, result)


def loss_fn(model, teacher_model, x, objective):
    with amp('cuda'):
        z = model(input_ids=x, use_cache=False).logits[:, :-1]
        if objective == 'kl':
            with torch.no_grad():
                target = teacher_model(input_ids=x, use_cache=False).logits[:, :-1]
            return core.full_kl(z, target, C['temperature'])
        return F.cross_entropy(z.float().reshape(-1, z.shape[-1]), x[:, 1:].reshape(-1))


def smoke():
    assert json.loads((ROOT / 'state/tests.json').read_text())['status'] == 'PASS'
    job = next(j for j in jobs(C) if j['level'] == 'moderate' and j['objective'] == 'kl')
    m = student(job, True)
    t = load()
    assert not any(p.requires_grad for p in t.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=C['learning_rate'], foreach=False)
    p = m.model.layers[0].mlp.up_proj.weight
    before = p.detach().clone()
    x = data('train')[:C['batch_size']].cuda()
    m.train()
    loss = loss_fn(m, t, x, 'kl')
    assert torch.isfinite(loss)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
    assert torch.isfinite(grad) and grad > 0
    opt.step()
    assert not torch.equal(before, p), 'No parameter update'
    assert all(p.grad is None for p in t.parameters()), 'Teacher received gradients'
    atomic_json(ROOT / 'state/smoke.json', {'status': 'PASS', 'loss': float(loss), 'gradient_norm': float(grad),
                'memory': memory(), 'teacher_and_student_resident': True,
                'trainable_parameters': sum(p.numel() for p in m.parameters() if p.requires_grad)})
    print('Full-weight online-KL smoke PASS', memory())


def train(job):
    assert json.loads((ROOT / 'state/smoke.json').read_text())['status'] == 'PASS'
    final = ROOT / 'metrics' / f'{job["id"]}.json'
    if final.exists():
        print('Already complete:', job['id'])
        return
    seed(job['seed'])
    construction_start = time.perf_counter()
    model = student(job, True)
    sync()
    student_construction_seconds = time.perf_counter() - construction_start
    teacher_start = time.perf_counter()
    teacher_model = load() if job['objective'] == 'kl' else None
    sync()
    teacher_load_seconds = time.perf_counter() - teacher_start
    opt = torch.optim.AdamW(model.parameters(), lr=C['learning_rate'], weight_decay=C['weight_decay'], foreach=False)
    blocks = data('train')
    effective = C['batch_size'] * C['accumulation']
    assert len(blocks) % effective == 0, 'Use an effective batch dividing the 1536 fixed blocks'
    per_epoch = len(blocks) // effective
    total = per_epoch * C['epochs']
    orders = [torch.randperm(len(blocks), generator=torch.Generator().manual_seed(job['seed'] + 1000 + e))
              for e in range(C['epochs'])]
    checkpoint = ROOT / 'state' / job['id'] / 'resume.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    curve, losses, elapsed, start = [], [], 0., 0
    if checkpoint.exists():
        state = core.checked_load(checkpoint, backup=True)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start, curve, losses, elapsed = state['step'], state['curve'], state['losses'], state['seconds']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    else:
        with amp('cuda'):
            val = core.nll(model, data('evaluation')['validation'], 'cuda')
        curve = [{'tokens': 0, 'validation': val}]

    def save(step):
        core.atomic_torch(checkpoint, {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': opt.state_dict(), 'step': step, 'curve': curve, 'losses': losses,
            'seconds': elapsed, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, backup=True)

    for step in range(start, total):
        if (ROOT / 'STOP').exists():
            save(step)
            raise RuntimeError('Paused at update boundary. Remove STOP to resume.')
        epoch, offset = divmod(step, per_epoch)
        ids = orders[epoch][offset * effective:(offset + 1) * effective]
        model.train()
        opt.zero_grad(set_to_none=True)
        sync()
        tick = time.perf_counter()
        value = 0.
        for micro in ids.split(C['batch_size']):
            x = blocks[micro].cuda()
            loss = loss_fn(model, teacher_model, x, job['objective']) / C['accumulation']
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            value += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm), 'Nonfinite gradients'
        opt.step()
        sync()
        elapsed += time.perf_counter() - tick
        losses.append(value)
        if (step + 1) % C['checkpoint_updates'] == 0 or step + 1 == total:
            with amp('cuda'):
                val = core.nll(model, data('evaluation')['validation'], 'cuda')
            curve.append({'tokens': (step + 1) * effective * blocks.shape[1], 'validation': val})
            save(step + 1)
            print(job['id'], step + 1, '/', total, 'loss', value, 'validation PPL', val['perplexity'], flush=True)
    training_memory = memory()
    opt.zero_grad(set_to_none=True)
    del opt, teacher_model
    gc.collect()
    torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    output = ROOT / 'models' / job['id']
    temporary = output.with_name(output.name + '.incomplete')
    if output.exists():
        # Recover an interruption between durable export and metrics publication.
        receipt = json.loads((output / 'EXPORT_COMPLETE.json').read_text())
        for name, expected_hash in receipt.items():
            assert sha(output / name) == expected_hash, 'Existing export changed'
        assert json.loads((output / 'pruning_recipe.json').read_text()) == recipe(job)
        exported = output
    else:
        # A previous incomplete export belongs only to this run/job.
        temporary.mkdir(exist_ok=True)
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size='2GB')
        tok().save_pretrained(temporary)
        shutil.copy2(PACKAGE / 'licenses/QWEN_RESEARCH_LICENSE.txt', temporary / 'LICENSE.txt')
        shutil.copy2(PACKAGE / 'licenses/NOTICE.txt', temporary / 'NOTICE.txt')
        atomic_json(temporary / 'pruning_recipe.json', recipe(job))
        exported = temporary
    probe = blocks[:1].cuda()
    with torch.no_grad():
        expected = model(probe, use_cache=False).logits[:, -1].cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = load(exported)
    with torch.no_grad():
        actual = model(probe, use_cache=False).logits[:, -1].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = score(model)
    result.update(job=job, recovery=curve, losses=losses,
                  input_tokens=total * effective * blocks.shape[1],
                  unique_input_tokens=len(blocks) * blocks.shape[1], updates=total,
                  training_seconds=elapsed, training_tokens_per_second=total * effective * blocks.shape[1] / elapsed,
                  current_process_student_construction_seconds=student_construction_seconds,
                  current_process_teacher_load_seconds=teacher_load_seconds,
                  training_memory=training_memory, reload_verified=True,
                  precision='BF16 exported/evaluated; FP32 training parameters with BF16 autocast')
    if exported == temporary:
        receipt = {p.name: sha(p) for p in temporary.iterdir() if p.is_file() and p.name != 'EXPORT_COMPLETE.json'}
        atomic_json(temporary / 'EXPORT_COMPLETE.json', receipt)
        temporary.rename(output)
    atomic_json(final, result)
    # Final model is durable; large optimizer checkpoint is no longer needed.
    checkpoint.unlink()
    checkpoint.with_suffix('.sha256').unlink()
    for suffix in ['.previous.pt', '.previous.sha256']:
        checkpoint.with_suffix(suffix).unlink(missing_ok=True)


def demo(job_id):
    prompt = 'Machine learning is a branch of artificial intelligence that'
    teacher_model = load()
    student_model = load(ROOT / 'models' / job_id)
    tokenizer = tok()
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    result = {'prompt': prompt, 'student': job_id, 'scope': 'Illustrative deterministic continuation, not a quality test'}
    for label, model in [('teacher', teacher_model), ('student', student_model)]:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        result[label + '_output'] = tokenizer.decode(out[0], skip_special_tokens=True)
        result[label + '_parameters'] = sum(p.numel() for p in model.parameters())
    result['both_resident_memory'] = memory()
    atomic_json(ROOT / 'reports/demo.json', result)
    print(json.dumps(result, indent=2))


def benchmark(job_id, device):
    path = ROOT / 'metrics' / f'benchmark_{job_id}_{device}.json'
    if path.exists():
        return
    source = Path(C['teacher']) if job_id == 'teacher' else ROOT / 'models' / job_id
    if job_id != 'teacher':
        assert (source / 'EXPORT_COMPLETE.json').exists()
    tick = time.perf_counter()
    model = load(source, device)
    sync(device)
    load_seconds = time.perf_counter() - tick
    tokenizer = tok()
    cases = []
    for length in C['benchmark_lengths']:
        text = 'Machine learning uses data to learn useful patterns. ' * (length + 1)
        def request():
            begin = time.perf_counter()
            host = tokenizer(text, return_tensors='pt', truncation=True, max_length=length)
            preprocessing = time.perf_counter() - begin
            tick = time.perf_counter()
            ids = host.to(device)
            sync(device)
            transfer = time.perf_counter() - tick
            tick = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**ids, min_new_tokens=C['benchmark_new_tokens'],
                    max_new_tokens=C['benchmark_new_tokens'], do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            sync(device)
            execution = time.perf_counter() - tick
            tick = time.perf_counter()
            tokenizer.decode(output[0].cpu(), skip_special_tokens=True)
            decode = time.perf_counter() - tick
            return {'preprocess': preprocessing, 'transfer': transfer, 'model': execution,
                    'decode': decode, 'e2e': time.perf_counter() - begin}
        for _ in range(C['warmups']):
            request()
        rows = [request() for _ in range(C['repeats'])]
        medians = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        cases.append({'prompt_tokens': length, 'generated_tokens': C['benchmark_new_tokens'],
                      'runs': rows, 'median_seconds': medians,
                      'model_iqr_seconds': float(np.percentile([r['model'] for r in rows], 75) - np.percentile([r['model'] for r in rows], 25)),
                      'output_tokens_per_second': C['benchmark_new_tokens'] / medians['model']})
    atomic_json(path, {'variant': job_id, 'device': device, 'dtype': str(next(model.parameters()).dtype),
        'load_seconds': load_seconds, 'cases': cases, 'quality_subset': score(model, device, True),
        'warmups': C['warmups'], 'repeats': C['repeats'], 'memory': memory(device),
        'caveat': 'CPU FP32 vs CUDA BF16 combines precision and device effects. Request E2E excludes model load; throughput includes prefill.'})


def tiny(device):
    import tiny_reference
    # Tiny path retains the audited random/informed CE/KL reference protocol.
    # Increase nominal positions to cover 512 prompt + 16 generated tokens.
    original = tiny_reference.tiny_model
    def make(*args, **kwargs):
        m = original(*args, **kwargs)
        m.config.max_position_embeddings = 1024
        return m
    tiny_reference.tiny_model = make
    tiny_reference.run(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['setup', 'recipes', 'teacher', 'smoke', 'train', 'benchmark', 'tiny', 'report', 'tests', 'demo'])
    p.add_argument('--job')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed(42)
        if args.stage != 'setup':
            check_lock()
        tick = time.perf_counter()
        if args.stage == 'train':
            train(next(j for j in jobs(C) if j['id'] == args.job))
        elif args.stage == 'benchmark':
            benchmark(args.job, args.device)
        elif args.stage == 'tiny':
            tiny(args.device)
        elif args.stage == 'demo':
            demo(args.job)
        elif args.stage == 'report':
            from report import build
            build(ROOT, C)
        elif args.stage == 'tests':
            import runtime_tests
            runtime_tests.run(ROOT)
        else:
            globals()[args.stage]()
        with (ROOT / 'logs/phases.jsonl').open('a') as f:
            f.write(json.dumps({'stage': args.stage, 'job': args.job, 'device': args.device,
                                'seconds': time.perf_counter() - tick}) + '\n')


if __name__ == '__main__':
    main()
