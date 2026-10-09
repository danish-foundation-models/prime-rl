"""Read-only parking observer for exactly one Slurm pilot label."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from ucloud_sandboxes_sdk import SandboxClient

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job-id',type=int,required=True)
p.add_argument('--run-label',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path('/scratch/project_465002751/rasmus')
for file in [root/'post/.env',root/'prime-rl/.env']:load_dotenv(file,override=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
client=SandboxClient.from_env(timeout_seconds=30)
try:
    deadline=time.monotonic()+7*3600
    while time.monotonic()<deadline:
        monitor=a.output.parent/'status.json'
        status=json.loads(monitor.read_text()) if monitor.exists() else {}
        assert not status or status.get('job_id')==a.job_id
        job=status.get('state','UNKNOWN')
        if job=='ENDED':break
        try:
            rows=[]
            for row in client.list_sandboxes():
                spec=row.get('spec') or {};labels=spec.get('labels') or row.get('labels') or {}
                if labels.get('run')!=a.run_label:continue
                rows.append({'id':row.get('id',spec.get('id')),'state':row.get('state',row.get('status')),
                             'parkable':spec.get('parkable'), 'managed_process':spec.get('managed_process'),
                             'generation':row.get('generation')})
            result={'time':time.time(),'job':job,'run_label':a.run_label,'count':len(rows),
                    'states':dict(Counter(str(r['state']) for r in rows)),
                    'nonparkable':sum(r['parkable'] is not True for r in rows),
                    'unmanaged':sum(r['managed_process'] is not True for r in rows),'sandboxes':rows}
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.output)
            print(json.dumps({k:v for k,v in result.items() if k!='sandboxes'}),flush=True)
        except Exception as error:
            print(json.dumps({'time':time.time(),'error_type':type(error).__name__}),flush=True)
        time.sleep(30)
finally:
    pass  # The synchronous SDK client has no persistent session or close method.
