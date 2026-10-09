"""Replace one wheel in an immutable offline archive without touching the source."""
import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('wheel',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
assert not a.output.exists(), 'Refuse to overwrite an archive'
manifest=json.loads(a.source.with_name(a.source.name+'.json').read_text())
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==manifest['sha256']
prefix='python/lib/python3.12/site-packages/'
with zipfile.ZipFile(a.wheel) as wheel:
    names=wheel.namelist()
    roots={name.split('/')[0] for name in names}
    assert roots=={'cryptography','cryptography-50.0.1.dist-info'},roots
    with tarfile.open(a.source,'r:gz') as source, tarfile.open(a.output,'x:gz',compresslevel=1) as out:
        for member in source:
            relative=member.name.removeprefix(prefix)
            if member.name.startswith(prefix) and relative.split('/')[0] in roots:continue
            # Portable ownership: no LUMI build-host UID escapes into the archive.
            member.uid=member.gid=0;member.uname=member.gname='root'
            out.addfile(member,source.extractfile(member) if member.isfile() else None)
        for name in names:
            if name.endswith('/'):continue
            assert '..' not in Path(name).parts and not name.startswith('/')
            data=wheel.read(name);member=tarfile.TarInfo(prefix+name)
            member.size=len(data);member.mode=0o644;member.uid=member.gid=0
            member.uname=member.gname='root';out.addfile(member,io.BytesIO(data))
manifest['source_sha256']=manifest['sha256']
manifest['sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
manifest['replacement_wheel']={'name':a.wheel.name,'sha256':hashlib.sha256(a.wheel.read_bytes()).hexdigest()}
a.output.with_name(a.output.name+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(a.output),'sha256':manifest['sha256']},indent=2))
