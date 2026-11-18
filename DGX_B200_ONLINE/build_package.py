"""Create a clean upload ZIP and checksum after validation."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
validation = json.loads((ROOT / 'LOCAL_VALIDATION.json').read_text())
assert validation['status'] == 'PASS_CPU_AND_PACKAGE_TESTS_ONLY'
files = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and p.suffix != '.pyc'
               and p.name != 'CHECKSUMS.sha256')
(ROOT / 'CHECKSUMS.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in files))
archive = ROOT.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in [*files, ROOT / 'CHECKSUMS.sha256']:
        z.write(p, p.relative_to(ROOT.parent))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
archive.with_suffix('.zip.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
print(archive, archive.stat().st_size, 'bytes; ZIP integrity PASS')
