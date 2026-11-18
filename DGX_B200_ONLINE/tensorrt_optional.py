"""Conditional TensorRT-LLM converter/builder/benchmark for compatible releases.

Run in a SEPARATE TensorRT-LLM environment, after confirming the installed release
supports B200 and Qwen2 custom depth/FFN dimensions. Uses that release's own Qwen
conversion script rather than vendoring an unpinned converter. Explicit failures
remain failures, never silently fall back to PyTorch.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from protocol import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Matching TensorRT-LLM source checkout')
    p.add_argument('--job', default='moderate_sensitivity_s42_kl')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--build-only', action='store_true')
    args = p.parse_args()
    version = importlib.metadata.version('tensorrt_llm')
    converter = args.source / 'examples/qwen/convert_checkpoint.py'
    if not converter.exists():
        raise RuntimeError('This release does not expose examples/qwen/convert_checkpoint.py; use its documented conversion API. No fallback.')
    builder = shutil.which('trtllm-build')
    if not builder:
        raise RuntimeError('trtllm-build missing in this environment')
    model_dir = args.run / 'models' / args.job
    receipt = json.loads((model_dir / 'EXPORT_COMPLETE.json').read_text())
    for name, expected in receipt.items():
        assert sha(model_dir / name) == expected
    output = args.run / 'tensorrt' / args.job
    output.mkdir(parents=True, exist_ok=True)
    converted, engine = output / 'converted', output / 'engine'
    config = json.loads((model_dir / 'config.json').read_text())
    source_commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    build_receipt = output / 'build.json'
    start = time.perf_counter()
    if not build_receipt.exists():
        with (output / 'conversion.log').open('w') as log:
            subprocess.run([sys.executable, str(converter), '--model_dir', str(model_dir),
                            '--output_dir', str(converted), '--dtype', 'bfloat16', '--tp_size', '1'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        conversion_seconds = time.perf_counter() - start
        help_text = subprocess.check_output([builder, '--help'], text=True)
        required = ['--checkpoint_dir', '--output_dir', '--gemm_plugin', '--gather_context_logits']
        if not all(flag in help_text for flag in required):
            raise RuntimeError('Builder API differs; use the installed release documentation. No unverified engine benchmark.')
        command = [builder, '--checkpoint_dir', str(converted), '--output_dir', str(engine),
                   '--gemm_plugin', 'bfloat16', '--max_batch_size', '1', '--max_input_len', '512', '--gather_context_logits']
        if '--max_seq_len' in help_text:
            command += ['--max_seq_len', '528']
        elif '--max_output_len' in help_text:
            command += ['--max_output_len', '16']
        else:
            raise RuntimeError('Cannot establish engine sequence bounds for this release')
        start = time.perf_counter()
        with (output / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(build_receipt, {'version': version, 'source_commit': source_commit,
                    'converter_sha256': sha(converter), 'model_receipt_sha256': sha(model_dir / 'EXPORT_COMPLETE.json'),
                    'model_config': config, 'command': command, 'conversion_seconds': conversion_seconds,
                    'engine_build_seconds': time.perf_counter() - start})
    else:
        existing = json.loads(build_receipt.read_text())
        assert existing['version'] == version and existing['source_commit'] == source_commit
        assert existing['model_receipt_sha256'] == sha(model_dir / 'EXPORT_COMPLETE.json')
    if args.build_only:
        return
    import torch
    import torch.nn.functional as F
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner
    from transformers import AutoTokenizer
    assert tensorrt_llm.mpi_world_size() == 1, 'This package benchmarks one GPU'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    start = time.perf_counter()
    runner = ModelRunner.from_dir(engine_dir=str(engine), rank=0)
    load_seconds = time.perf_counter() - start
    end_id = tokenizer.eos_token_id
    @torch.inference_mode()
    def generate(ids, count, logits=False):
        return runner.generate([ids], max_new_tokens=count, end_id=end_id, pad_id=end_id,
                               num_beams=1, top_k=1, return_dict=True,
                               output_sequence_lengths=True, output_context_logits=logits)
    cases = []
    for length in [32, 128, 512]:
        text = 'Machine learning uses data to learn useful patterns. ' * (length + 1)
        def request():
            start = time.perf_counter()
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=length).input_ids[0].int()
            preprocess = time.perf_counter() - start
            tick = time.perf_counter()
            result = generate(ids, 16)
            torch.cuda.synchronize()
            execution = time.perf_counter() - tick
            # Report actual output count: early EOS must not inflate throughput.
            if 'sequence_lengths' not in result:
                raise RuntimeError('Runtime did not return sequence_lengths; adapt to this release before timing claims')
            count = int(result['sequence_lengths'].reshape(-1)[0]) - len(ids)
            tick = time.perf_counter()
            tokenizer.decode(result['output_ids'].reshape(-1).cpu(), skip_special_tokens=True)
            decode = time.perf_counter() - tick
            return {'preprocess': preprocess, 'engine_including_transfer': execution,
                    'decode': decode, 'e2e': time.perf_counter() - start, 'generated_tokens': count}
        for _ in range(2):
            request()
        runs = [request() for _ in range(5)]
        cases.append({'prompt_tokens': length, 'runs': runs,
                      'median_e2e_seconds': statistics.median(r['e2e'] for r in runs),
                      'median_output_tokens_per_second': statistics.median(r['generated_tokens'] / r['engine_including_transfer'] for r in runs)})
    def context_logits(ids):
        result = generate(ids.int(), 1, logits=True)
        if 'context_logits' not in result:
            raise RuntimeError('No context logits: scientific engine quality check cannot be completed on this API')
        value = result['context_logits']
        value = value[0] if isinstance(value, (list, tuple)) or value.ndim == 3 else value
        return value[:len(ids), :config['vocab_size']].float()
    blocks = torch.load(args.run / 'data/evaluation.pt', weights_only=True)['test'][:16]
    losses = []
    for row in blocks:
        z = context_logits(row)
        losses.append(float(F.cross_entropy(z[:-1], row[1:].to(z.device))))
    records = json.loads((args.run / 'data/piqa.json').read_text())['records'][:32]
    predictions = []
    for rec in records:
        prefix = tokenizer('Question: ' + rec['goal'] + '\nAnswer:', add_special_tokens=False)['input_ids']
        scores = []
        for key in ['sol1', 'sol2']:
            answer = tokenizer(' ' + rec[key], add_special_tokens=False)['input_ids']
            ids = torch.tensor(prefix + answer)
            if len(ids) > 512:
                raise RuntimeError('PIQA example exceeds engine input bounds; rebuild with sufficient length')
            z = context_logits(ids)[len(prefix)-1:-1]
            scores.append(-float(F.cross_entropy(z, torch.tensor(answer, device=z.device))))
        predictions.append({'label': rec['label'], 'prediction': int(scores[1] > scores[0])})
    import math
    atomic_json(output / 'benchmark.json', {'version': version, 'gpu': torch.cuda.get_device_name(),
                'load_seconds': load_seconds, 'cases': cases,
                'subset_ppl': math.exp(statistics.mean(losses)),
                'subset_piqa': statistics.mean(r['label'] == r['prediction'] for r in predictions),
                'predictions': predictions, 'scope': '16 test blocks/32 PIQA; compare with paired PyTorch CUDA subset',
                'timing_scope': 'Context-logit-enabled engine; call includes runtime-managed transfer. Early EOS count recorded, not assumed 16. Not a fixed-output-length speedup claim.'})
    atomic_json(args.run / 'reports/tensorrt_support.json', {'status': 'BUILT_AND_BENCHMARKED',
                'version': version, 'source_commit': source_commit, 'job': args.job,
                'evidence': str(output / 'benchmark.json')})
    print('Measured TensorRT artifacts:', output)


if __name__ == '__main__':
    main()
