"""Measured-only charts, tables and requirement completeness gate."""
import csv
import json
from pathlib import Path
import statistics
import zipfile
from protocol import jobs, atomic_json, sha


def build(root, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    read = lambda p: json.loads((root / p).read_text())
    out = root / 'reports'
    out.mkdir(exist_ok=True)
    expected = ['metrics/teacher.json', 'state/tests.json', 'state/smoke.json',
                'metrics/pruning_scores.json', 'metrics/tiny_cpu.json', 'metrics/tiny_cuda.json']
    expected += [f'metrics/{j["id"]}.json' for j in jobs(config)]
    expected += [f'metrics/benchmark_{j}_{d}.json' for j in ['teacher', *config['benchmark_jobs']] for d in ['cpu', 'cuda']]
    missing = [n for n in expected if not (root / n).exists()]
    backend_path = out / 'tensorrt_support.json'
    backend = json.loads(backend_path.read_text()) if backend_path.exists() else {'status': 'NOT_YET_AUDITED'}
    atomic_json(out / 'completeness.json', {'status': 'INCOMPLETE' if missing else 'PASS',
                'scope': 'PyTorch study; conditional TensorRT status recorded separately',
                'tensorrt': backend, 'missing': missing})
    if missing:
        raise RuntimeError('Report incomplete; see reports/completeness.json: ' + ', '.join(missing))
    assert read('state/tests.json')['status'] == read('state/smoke.json')['status'] == 'PASS'
    teacher = read('metrics/teacher.json')
    rows, groups = [], {}
    for job in jobs(config):
        value = read(f'metrics/{job["id"]}.json')
        assert value['reload_verified']
        assert value['input_tokens'] == 1536 * 128 * config['epochs']
        export = root / 'models' / job['id']
        for name, expected_hash in json.loads((export / 'EXPORT_COMPLETE.json').read_text()).items():
            assert sha(export / name) == expected_hash, f'Export changed: {export / name}'
        row = {**job, 'parameters': value['parameters'], 'ppl': value['test']['perplexity'],
               'nll': value['test']['nll'], 'piqa': value['piqa']['accuracy'],
               'initial_val_nll': value['recovery'][0]['validation']['nll'],
               'final_val_nll': value['recovery'][-1]['validation']['nll'],
               'training_seconds': value['training_seconds'], 'tokens': value['input_tokens'],
               'training_tokens_per_second': value['training_tokens_per_second'],
               'peak_allocated_gib': value['training_memory']['peak_allocated_gib']}
        rows.append(row)
        groups.setdefault((job['level'], job['criterion'], job['objective']), []).append(value)
    def write_csv(name, values):
        with (out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_csv('per_seed.csv', rows)
    aggregate = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (level, method, objective), values in groups.items():
        ppls = [v['test']['perplexity'] for v in values]
        tasks = [v['piqa']['accuracy'] for v in values]
        params = values[0]['parameters']
        aggregate.append({'level': level, 'criterion': method, 'objective': objective,
                          'parameters': params, 'ppl_mean': statistics.mean(ppls), 'ppl_sd': statistics.stdev(ppls),
                          'piqa_mean': statistics.mean(tasks), 'piqa_sd': statistics.stdev(tasks)})
        label = f'{level}/{method}/{objective}'
        axes[0].errorbar(params / 1e9, statistics.mean(ppls), yerr=statistics.stdev(ppls), fmt='o', label=label)
        axes[1].errorbar(params / 1e9, statistics.mean(tasks), yerr=statistics.stdev(tasks), fmt='o', label=label)
    axes[0].scatter(teacher['parameters'] / 1e9, teacher['test']['perplexity'], marker='*', s=120, label='teacher')
    axes[1].scatter(teacher['parameters'] / 1e9, teacher['piqa']['accuracy'], marker='*', s=120)
    for ax, y in zip(axes, ['Test perplexity ↓', 'PIQA accuracy ↑']):
        ax.set(xlabel='Parameters (billions)', ylabel=y)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'compression_quality.png', dpi=170)
    plt.close(fig)
    write_csv('aggregate.csv', aggregate)
    points = [{'id': r['level'] + '/' + r['criterion'] + '/' + r['objective'],
               'parameters': r['parameters'], 'ppl': r['ppl_mean'], 'piqa': r['piqa_mean']} for r in aggregate]
    points.append({'id': 'teacher', 'parameters': teacher['parameters'], 'ppl': teacher['test']['perplexity'], 'piqa': teacher['piqa']['accuracy']})
    frontiers = {}
    for metric, sign in [('ppl', 1), ('piqa', -1)]:
        frontiers[metric] = [p['id'] for p in points if not any(
            q['parameters'] <= p['parameters'] and sign*q[metric] <= sign*p[metric]
            and (q['parameters'] < p['parameters'] or sign*q[metric] < sign*p[metric]) for q in points)]
    atomic_json(out / 'pareto.json', {'points': points, 'frontiers': frontiers, 'scope': 'Observed finite points only'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (level, method, objective), values in groups.items():
        ax = axes[list(config['structures']).index(level)]
        marks = values[0]['recovery']
        ax.plot([m['tokens'] for m in marks],
                [statistics.mean(v['recovery'][i]['validation']['nll'] for v in values) for i in range(len(marks))],
                label=f'{method}/{objective}')
        ax.set(title=level, xlabel='Input tokens processed (including repeated epochs)', ylabel='Validation NLL')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'recovery.png', dpi=170)
    plt.close(fig)
    scores = read('metrics/pruning_scores.json')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(scores['single_layer_delta_nll'])
    axes[0].set(title='Single-layer calibration ablation', xlabel='Removed layer (zero-based)', ylabel='Delta NLL')
    axes[1].plot(scores['layer_squared_l2'])
    axes[1].set(title='Layer weight magnitude', xlabel='Layer', ylabel='Squared L2')
    im = axes[2].imshow(scores['channel_quantiles'], aspect='auto')
    axes[2].set(title='FFN magnitude quantiles (not ablation)', xlabel='Quantile decile', ylabel='Layer')
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    fig.savefig(out / 'pruning_maps.png', dpi=170)
    plt.close(fig)
    ranking = []
    for level in config['structures']:
        for seed in config['seeds']:
            candidates = [r for r in rows if r['level'] == level and r['seed'] == seed and r['objective'] == 'kl']
            winners = lambda field, reverse=False: [r['criterion'] for r in candidates if r[field] == (max if reverse else min)(x[field] for x in candidates)]
            ranking.append({'level': level, 'seed': seed, 'initial': winners('initial_val_nll'),
                            'final_validation': winners('final_val_nll'), 'test': winners('nll'), 'piqa': winners('piqa', True)})
    atomic_json(out / 'ranking.json', ranking)
    contrasts = []
    for level in config['structures']:
        for criterion in config['criteria']:
            for seed in config['seeds']:
                pair = {r['objective']: r for r in rows if (r['level'], r['criterion'], r['seed']) == (level, criterion, seed)}
                contrasts.append({'level': level, 'criterion': criterion, 'seed': seed,
                                  'kl_minus_ce_nll': pair['kl']['nll'] - pair['ce']['nll'],
                                  'kl_minus_ce_piqa': pair['kl']['piqa'] - pair['ce']['piqa']})
    write_csv('paired_objective_effects.csv', contrasts)
    timings, quality = [], []
    for variant in ['teacher', *config['benchmark_jobs']]:
        cpu, gpu = [read(f'metrics/benchmark_{variant}_{d}.json') for d in ['cpu', 'cuda']]
        quality.append({'model': variant, 'cpu_ppl': cpu['quality_subset']['test']['perplexity'],
                        'cuda_ppl': gpu['quality_subset']['test']['perplexity'],
                        'cpu_piqa': cpu['quality_subset']['piqa']['accuracy'],
                        'cuda_piqa': gpu['quality_subset']['piqa']['accuracy']})
        for a, b in zip(cpu['cases'], gpu['cases']):
            timings.append({'model': variant, 'prompt_tokens': a['prompt_tokens'],
                            'cpu_model_seconds': a['median_seconds']['model'], 'gpu_model_seconds': b['median_seconds']['model'],
                            'cpu_e2e_seconds': a['median_seconds']['e2e'], 'gpu_e2e_seconds': b['median_seconds']['e2e'],
                            'e2e_cpu_over_gpu': a['median_seconds']['e2e'] / b['median_seconds']['e2e']})
    write_csv('deployment.csv', timings)
    write_csv('paired_quality_subset.csv', quality)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for variant in ['teacher', *config['benchmark_jobs']]:
        subset = [r for r in timings if r['model'] == variant]
        for device, style in [('cpu', '--'), ('gpu', '-')]:
            axes[0].plot([r['prompt_tokens'] for r in subset], [r[f'{device}_model_seconds'] for r in subset],
                         style, marker='o', label=f'{variant}/{device}')
    axes[0].set(xlabel='Prompt tokens', ylabel='Median model seconds', title='Real: CPU FP32 / CUDA BF16')
    axes[0].legend(fontsize=6)
    for device in ['cpu', 'cuda']:
        cases = read(f'metrics/tiny_{device}.json')['benchmark']['cases']
        axes[1].plot([r['prompt_tokens'] for r in cases], [r['median_seconds'] for r in cases], marker='o', label=device)
    axes[1].set(xlabel='Prompt tokens', ylabel='Median model seconds', title='Tiny: matched FP32, same weights')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'cpu_gpu_scaling.png', dpi=170)
    plt.close(fig)
    lines = ['# DGX online full-weight study — measured results', '',
             f'{len(rows)} completed runs; three seeds; full-weight AdamW recovery; concurrent BF16 teacher and student during KL.',
             f'Each run processes {rows[0]["tokens"]:,} tokens, repeating 196,608 unique training input tokens.',
             'New run/data provenance: bundled pinned RTX corpus, not the prior notebook validation split. Earlier PPL 13.53/34.357 is not a comparable baseline.',
             '', '|Size|Pruning|Loss|PPL mean ± SD|PIQA mean ± SD|', '|---|---|---|---:|---:|']
    for r in aggregate:
        lines.append(f'|{r["level"]}|{r["criterion"]}|{r["objective"]}|{r["ppl_mean"]:.3f} ± {r["ppl_sd"]:.3f}|{r["piqa_mean"]:.3f} ± {r["piqa_sd"]:.3f}|')
    lines += ['', '## Deployment crossover (observed samples)', 'CPU FP32 vs GPU BF16 changes both device and precision. Tiny reference uses matched FP32 weights.']
    for variant in ['teacher', *config['benchmark_jobs']]:
        wins = [r['prompt_tokens'] for r in timings if r['model'] == variant and r['e2e_cpu_over_gpu'] > 1]
        lines.append(f'- {variant}: smallest sampled CUDA-beneficial prompt: {min(wins) if wins else "none"}. No claim about unsampled crossover.')
    tiny = [read(f'metrics/tiny_{d}.json')['benchmark'] for d in ['cpu', 'cuda']]
    assert tiny[0]['matched_weights_sha256'] == tiny[1]['matched_weights_sha256']
    lines += ['', '## Limits and evidence',
              '- Three seeds; fixed custom PIQA200; repeated corpus epochs; no unseen generalization claim.',
              '- Sensitivity selects individual layers, not a globally optimal joint cut. FFN scores are magnitude heuristics.',
              '- All students and teacher are compared on common pinned inputs. Tiny byte PPL is not comparable with Qwen subword PPL.',
              '- Raw repeated timings, preprocessing, transfer, loading, quality and allocator memory: metrics/benchmark_*.json.',
              '- Training/computation time excludes checkpoint and validation overhead; logs/phases.jsonl records whole-stage wall time.',
              f'- TensorRT-LLM: {backend["status"]}; see tensorrt_support.json if present. PyTorch speed is not TensorRT speed.',
              '', '![Compression](compression_quality.png)', '![Recovery](recovery.png)', '![Pruning](pruning_maps.png)',
              '![CPU/GPU](cpu_gpu_scaling.png)']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    # Compact transfer bundle, not hundreds of GB of model/optimizer weights.
    with zipfile.ZipFile(root / 'RESULTS.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['reports', 'metrics', 'recipes', 'logs']:
            for p in (root / folder).rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(root))
        z.write(root / 'state/protocol.json', 'protocol.json')
        for p in (root / 'tensorrt').glob('*/*.json'):
            z.write(p, p.relative_to(root))
    print('Report:', out / 'REPORT.md', '\nHandoff:', root / 'RESULTS.zip')
