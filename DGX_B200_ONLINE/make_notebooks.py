"""Build checked-in notebooks without requiring notebook dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def notebook(name, title, sections):
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {title}\n',
              'Run from this package directory in your existing Blackwell-capable kernel.\n']}]
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': ['from pathlib import Path\n', 'import json, sys\n',
                             'assert Path("launch.py").exists(), "Open the notebook inside the extracted package folder"\n',
                             'from launch import run, train_all, benchmarks\n',
                             'C = json.loads(Path("config.json").read_text())\n',
                             'ROOT = Path(C["run_dir"])\n']})
    for markdown, code in sections:
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': markdown.splitlines(True)})
        if code:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': code.splitlines(True)})
    value = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3 (choose DGX kernel)',
              'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}},
             'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:03d}'
    (ROOT / name).write_text(json.dumps(value, indent=2))


notebook('00_START_HERE.ipynb', 'Start: environment, configuration and scientific lock', [
    ('## Read first\nThis is a NEW 36-run full-weight study. Existing teacher/recipes/checkpoints remain in their original folders. Read README.md and REQUIREMENTS_MAP.md. Default inputs are bundled pinned tensors; earlier notebook PPL is not comparable. Run all numbered notebooks in order; long commands block until finished.', 'print(Path("README.md").read_text())'),
    ('## Configuration\nEdit config.json NOW if needed. The UUID is from your last successful cell. No old 20 GB cap is imposed. Defaults process 1.966M tokens per run over 10 repeated epochs. Allow approximately 300 GB disk plus sufficient host RAM. Once setup runs, changes require a new run_dir.', 'print(json.dumps(C, indent=2))\nprint("Kernel:", sys.executable)'),
    ('## Install missing Python helpers (optional)\nKeep the working campus torch installation. Uncomment only if these helper packages are missing; restart the kernel after installation. This does not install a GPU driver.', '# import subprocess\n# subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-extra.txt"], check=True)'),
    ('## Initialize and verify teacher/data/environment\nRequires your existing teacher DOWNLOAD_RECORD.json. This hashes the existing teacher, records software and hardware, and runs an actual BF16 CUDA operation. It does not train.', 'run("setup")'),
    ('## Mathematical and structural correctness\nChecks forward KL direction/temperature/gradients, physical shape reduction and reload, frozen teacher, and full-weight optimizer checkpoint recovery.', 'run("tests")'),
])
notebook('01_PRUNING.ipynb', 'Score structures and save exact recipes', [
    ('## Three separate pruning arms\n**Random:** seeded arbitrary layers/channels. **Magnitude:** your last cell’s squared-L2 layer and coupled FFN rule. **Sensitivity:** independently ablate each layer on calibration text, select lowest damage, then use magnitude FFN. Layers need not be consecutive. Selection never uses validation/test. Joint cuts need not be globally least-damaging.', 'run("recipes")'),
    ('## Inspect actual selections\nThis creates recipes; physical students are built during the smoke and recovery stages.', 'for p in sorted((ROOT / "recipes").glob("*.json")):\n    r = json.loads(p.read_text())\n    print(p.name, "removed:", r["removed_layers"], "FFN:", r["ffn"])'),
])
notebook('02_TEACHER_BASELINE.ipynb', 'Measure the original teacher on the locked evaluation', [
    ('## Fresh baseline\n256 held-out test blocks and fixed PIQA200; validation is separate. This is BF16 on DGX. Do not substitute the old 13.53 notebook result or RTX 17.95 figure.', 'run("teacher")\nprint((ROOT / "metrics/teacher.json").read_text())'),
])
notebook('03_ONLINE_KD_SMOKE.ipynb', 'Prove full-weight online distillation fits and updates', [
    ('## Teacher and student resident together\nThe teacher stays frozen BF16. Student parameters, gradients and AdamW states are FP32; compute uses BF16 autocast and gradient checkpointing. Run one actual update and check finite nonzero gradients and a changed weight. This temporary smoke state is discarded.', 'run("smoke")\nprint((ROOT / "state/smoke.json").read_text())'),
    ('## If this fails\nSend the traceback and environment.json. Do not start the matrix until smoke passes. Two inference models fitting is not proof that full-weight optimizer training fits.', ''),
])
notebook('04_TRAIN_AND_EVALUATE.ipynb', 'Run and resume the full 36-condition study', [
    ('## Main experiment\nRuns one condition per subprocess. Completed jobs skip; partial jobs resume a hash-verified checkpoint. KL uses the online teacher; CE uses text labels only, but begins from the same pruned pretrained weights. Every student gets the same corpus/order/budget within a seed. Test results never choose checkpoints.', 'from protocol import jobs\nprint("Conditions:", len(jobs(C)))\nfor j in jobs(C):\n    print(j["id"])'),
    ('## Launch\nThis is a long training job. Keep the Jupyter server alive according to your campus policy. No runtime promise. Each job saves initial/final validation, a recovery curve, an export, reload proof and final test/PIQA scores.', 'train_all()'),
    ('## Pause/resume controls\nUse a SECOND notebook or terminal to create STOP while training is running. Do not start another training process. The commented cells below are intentional; uncomment only the action you want.', '# (ROOT / "STOP").touch()  # request pause at next optimizer boundary\n# (ROOT / "STOP").unlink(missing_ok=True)  # after inspecting progress, allow resume'),
    ('## Progress', 'completed = sorted((ROOT / "metrics").glob("*_s*_*.json"))\nprint("Completed:", len(completed), "/", len(jobs(C)))\nfor p in completed:\n    r = json.loads(p.read_text())\n    print(p.stem, "PPL", r["test"]["perplexity"], "PIQA", r["piqa"]["accuracy"])'),
])
notebook('05_CPU_REFERENCE.ipynb', 'Execute the reduced CPU and GPU scientific reference', [
    ('## Reduced reference\nTiny byte-level Qwen trains, physically prunes and recovers on CPU and GPU, with three seeds and CE/KL controls. CPU runs FIRST because the matched-FP32 benchmark uses that saved checkpoint on both devices. Byte PPL is not comparable with Qwen subword PPL.', 'run("tiny", "--device", "cpu")\nrun("tiny", "--device", "cuda")'),
])
notebook('06_BENCHMARKS.ipynb', 'Measure original and compressed CPU/GPU execution', [
    ('## Mandatory comparisons\nTeacher and two predeclared sensitivity/KL/seed42 students. Each device measures 32/128/512-token prompts, fixed 16-token output, two warmups and five repeats. Median/IQR, preprocessing, transfer, model execution, decode and request E2E are recorded; loading is separate. CPU uses FP32, CUDA BF16. This is a deployment comparison, not pure device speedup. CPU evaluation may be slow.', 'benchmarks()'),
    ('## Same-input quality subset\nEach model/device evaluates the SAME 16 test blocks and 32 PIQA cases. Subset results are not interchangeable with the full test.', 'for p in sorted((ROOT / "metrics").glob("benchmark_*.json")):\n    v = json.loads(p.read_text())\n    print(p.stem, v["dtype"], v["quality_subset"]["test"]["perplexity"])'),
])
notebook('07_REPORT_AND_EXPORT.ipynb', 'Validate, plot, summarize and package measured evidence', [
    ('## Completion gate\nRequires all main runs, CPU/GPU references, benchmarks and checkpoint reload proofs. This also hashes all exported weights and can take time. It creates measured-only tables and plots; no old results are copied in.', 'run("report")'),
    ('## Read report', 'from IPython.display import Markdown, display, Image\ndisplay(Markdown((ROOT / "reports/REPORT.md").read_text()))\nfor name in ["compression_quality.png", "recovery.png", "pruning_maps.png"]:\n    display(Image(filename=str(ROOT / "reports" / name)))'),
    ('## Download compact evidence\nRESULTS.zip excludes model weights. Keep server models/ for your live demo. The source/config/data upload package should accompany the results for reproduction.', 'print("Download:", ROOT / "RESULTS.zip")\nprint("Models retained at:", ROOT / "models")'),
])
notebook('08_LIVE_DEMO.ipynb', 'Live teacher versus distilled student', [
    ('## Both models on GPU\nLoads the teacher and the selected KL-recovered student concurrently, then shows the same deterministic continuation. It is a BASE-model completion demo, not a chat assistant. Use the measured PPL/PIQA table as quantitative evidence.', 'run("demo", "--job", "moderate_sensitivity_s42_kl")'),
    ('## Compare measured metrics', 'for name in ["teacher", "moderate_sensitivity_s42_kl"]:\n    v = json.loads((ROOT / "metrics" / f"{name}.json").read_text())\n    print(name, "parameters", v["parameters"], "PPL", v["test"]["perplexity"], "PIQA", v["piqa"]["accuracy"])'),
])
notebook('09_TENSORRT_SUPPORT.ipynb', 'Conditional TensorRT-LLM support audit', [
    ('## Backend-dependent requirement\nB200/Linux alone does not establish TensorRT-LLM support for a release and a custom-pruned Qwen checkpoint. Record the installed release and its documented supported hardware/model shapes. PyTorch remains the measured backend unless a separate engine is built and tested.',
     'import importlib.metadata, shutil, platform\nfrom protocol import atomic_json\ntry:\n    version = importlib.metadata.version("tensorrt_llm")\nexcept importlib.metadata.PackageNotFoundError:\n    version = None\naudit = {"platform": platform.platform(), "tensorrt_llm": version, "builder": shutil.which("trtllm-build"), "status": "UNAVAILABLE_NOT_BENCHMARKED" if version is None else "INSTALLED_SUPPORT_REQUIRES_VERIFICATION", "reference": "https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html", "note": "Absence is not proof of hardware incompatibility; do not call PyTorch timing TensorRT timing."}\natomic_json(ROOT / "reports/tensorrt_support.json", audit)\nprint(json.dumps(audit, indent=2))'),
    ('## If supported: conversion, build and benchmark\nThe optional helper uses your installed release’s own Qwen converter and trtllm-build. It measures conversion/build/load time, warmed repeated generation, actual generated-token counts, and the paired PPL/PIQA subset via context logits. APIs differ by release: it fails explicitly rather than claiming a fallback result. Set the two paths below and uncomment the subprocess call ONLY in a supported TensorRT environment. Keep it separate from the locked training environment.',
     'import os, subprocess\nTRT_PYTHON = Path("/path/to/tensorrt/environment/bin/python")\nTRT_SOURCE = Path("/path/to/matching/TensorRT-LLM/source")\nenv = os.environ.copy()\nenv["CUDA_VISIBLE_DEVICES"] = C["gpu_uuid"]\n# subprocess.run([str(TRT_PYTHON), "-u", "tensorrt_optional.py", "--source", str(TRT_SOURCE), "--run", str(ROOT), "--job", "moderate_sensitivity_s42_kl"], env=env, check=True)'),
    ('## Refresh handoff\nAfter recording support status or completing the optional benchmark, rebuild the report/ZIP to include that evidence.', 'run("report")'),
])

print('Generated', len(list(ROOT.glob('*.ipynb'))), 'notebooks')
