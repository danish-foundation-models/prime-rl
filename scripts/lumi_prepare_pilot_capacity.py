"""Reserve named pilot CPU/build capacity; run inside the operational SIF."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from ucloud_sandboxes_sdk import SandboxClient, SandboxApiError

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--builder-prepare-id',required=True)
parser.add_argument('--capacity-prepare-id',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--release',action='store_true')
parser.add_argument('--job-id',type=int,required=True)
args=parser.parse_args()
if any(f'-{args.job_id}-' not in value for value in [args.builder_prepare_id,args.capacity_prepare_id]):
    parser.error('Reservation IDs must be bound to the Slurm job ID')
for path in [args.root/'post/.env',args.root/'prime-rl/.env']:load_dotenv(path,override=True)
client=SandboxClient.from_env(timeout_seconds=120)
if args.release:
    for method,identifier in [(client.delete_prepared_builder,args.builder_prepare_id),(client.delete_prepared_capacity,args.capacity_prepare_id)]:
        try:
            method(identifier)
        except SandboxApiError as error:
            if error.status_code != 404:
                raise
    args.output.write_text(json.dumps({'released_reservations':[args.builder_prepare_id,args.capacity_prepare_id]},indent=2))
    print('Released only the named job-bound reservations; no sandbox/image deletion',flush=True)
else:
    builders=client.prepare_builder(count=16,ttl_seconds=25200,prepare_id=args.builder_prepare_id)
    capacity=client.prepare_capacity(count=100,parkable=True,cpus=4,memory_mb=4096,disk_mb=10240,ttl_seconds=3600,prepare_id=args.capacity_prepare_id)
    args.output.write_text(json.dumps({'builders':builders,'capacity':capacity},indent=2))
    print('Named pilot capacity reservations accepted',flush=True)
