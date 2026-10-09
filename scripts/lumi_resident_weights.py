"""GPU-resident checkpoint source for isolated NCCL update benchmarks."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


class ResidentWeights:
    def __init__(self, model: Path, output: Path):
        self.rank = dist.get_rank()
        self.device = torch.device('cuda', self.rank)
        self.output = output
        self.iteration = 0
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        files = defaultdict(list)
        for key, filename in index.items():
            if not key.startswith('mtp.'):
                files[filename].append(key)
        dtype_map = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}
        layers = defaultdict(list)
        for filename, keys in files.items():
            with safe_open(model / filename, framework='pt', device='cpu') as f:
                for key in keys:
                    if key.startswith('backbone.layers.'):
                        layer = int(key.split('.')[2])
                        if layer >= 88:
                            continue
                    elif key in {'backbone.embeddings.weight', 'backbone.norm_f.weight', 'lm_head.weight'}:
                        layer = -1
                    else:
                        raise ValueError(key)
                    info = f.get_slice(key)
                    shape = tuple(info.get_shape())
                    dtype = dtype_map[info.get_dtype()]
                    layers[layer].append((key, filename, shape, math.prod(shape), dtype))
        assert len(layers) == 89
        sizes = {layer: sum(n * torch.empty((), dtype=d).element_size() for _, _, _, n, d in entries) for layer, entries in layers.items()}
        loads = [0] * dist.get_world_size()
        owners = {}
        for layer in sorted(layers, key=lambda k: (-sizes[k], k)):
            owner = min(range(len(loads)), key=lambda r: (loads[r], r))
            owners[layer] = owner
            loads[owner] += sizes[layer]
        free, _ = torch.cuda.mem_get_info(self.device)
        if loads[self.rank] + 2 * max(sizes.values()) + 4 * 1024**3 > free:
            raise RuntimeError(f'Insufficient source GPU memory: rank={self.rank}, resident={loads[self.rank]}, free={free}')
        self.layers = []
        self.resident = {}
        for layer in sorted(layers):
            by_dtype = defaultdict(list)
            for entry in layers[layer]:
                by_dtype[entry[-1]].append(entry)
            buckets = []
            for dtype, entries in by_dtype.items():
                chunk, nbytes = [], 0
                for entry in entries:
                    size = entry[3] * torch.empty((), dtype=dtype).element_size()
                    if chunk and nbytes + size > 268435456:
                        buckets.append((dtype, chunk))
                        chunk, nbytes = [], 0
                    chunk.append(entry)
                    nbytes += size
                if chunk:
                    buckets.append((dtype, chunk))
            self.layers.append((layer, owners[layer], buckets))
            if owners[layer] != self.rank:
                continue
            flats = []
            for dtype, entries in buckets:
                if time.monotonic() - started > 600:
                    raise TimeoutError("GPU checkpoint staging exceeded ten minutes")
                # Fill final resident storage directly: no full-model host packing.
                flat = torch.empty(sum(e[3] for e in entries), dtype=dtype, device=self.device)
                offset = 0
                for key, filename, shape, numel, _ in entries:
                    with safe_open(model / filename, framework='pt', device='cpu') as f:
                        flat[offset:offset+numel].view(shape).copy_(f.get_tensor(key))
                    offset += numel
                flats.append(flat)
            self.resident[layer] = flats
            print(f'RESIDENT rank={self.rank} layer={layer}', flush=True)
        torch.cuda.synchronize(self.device)
        (output / f'load-rank-{self.rank}.json').write_text(json.dumps({'rank': self.rank, 'resident_bytes': loads[self.rank], 'seconds': time.monotonic()-started}))
        dist.barrier()

    def command(self, value=0):
        command = torch.tensor([value], dtype=torch.int32, device=self.device)
        dist.broadcast(command, src=0)
        return int(command.item())

    def send(self, communicator=None):
        from prime_rl.transports.weights.nccl import broadcast_integer, broadcast_state_dict
        if self.rank == 0:
            self.command(1)
            broadcast_integer(len(self.layers), communicator)
        started = time.monotonic()
        assemble_s = broadcast_s = 0.0
        for layer, owner, buckets in self.layers:
            torch.cuda.synchronize(self.device)
            begin = time.monotonic()
            state = {}
            for bi, (dtype, entries) in enumerate(buckets):
                numel = sum(e[3] for e in entries)
                flat = self.resident[layer][bi] if self.rank == owner else torch.empty(numel, dtype=dtype, device=self.device)
                dist.broadcast(flat, src=owner)
                if self.rank == 0:
                    offset = 0
                    for key, _, shape, count, _ in entries:
                        state[key] = flat[offset:offset+count].view(shape)
                        offset += count
                del flat
            torch.cuda.synchronize(self.device)
            assembled = time.monotonic()
            assemble_s += assembled - begin
            if self.rank == 0:
                broadcast_state_dict(state, communicator)
                torch.cuda.synchronize(self.device)
            broadcast_s += time.monotonic() - assembled
            del state
        dist.barrier()
        if self.rank == 0:
            result = {'iteration': self.iteration, 'assembly_s': assemble_s, 'broadcast_pack_and_receiver_wait_s': broadcast_s, 'total_s': time.monotonic()-started}
            (self.output / f'iteration-{self.iteration}.json').write_text(json.dumps(result, indent=2))
            print('RESIDENT_TIMING '+json.dumps(result), flush=True)
        self.iteration += 1

    def serve(self):
        while self.command() == 1:
            self.send()
