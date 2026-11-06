"""Relay 127.0.0.1:<local_port> to 127.0.0.1:<remote_port> on <host> over `ssh host nc`.

Compute nodes refuse ssh port forwarding but allow command execution, so each
connection runs nc remotely over one multiplexed ssh master.
"""

import asyncio
import os
import sys

host, local_port, remote_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
    "-o", f"ControlPath={os.path.expanduser('~/.ssh')}/cm-dash-%C", "-o", "ControlPersist=600",
    host, "nc", "-N", "127.0.0.1", str(remote_port),
]


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    proc = await asyncio.create_subprocess_exec(
        *SSH, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    await asyncio.gather(pipe(reader, proc.stdin), pipe(proc.stdout, writer), return_exceptions=True)
    if proc.returncode is None:
        proc.kill()
    await proc.wait()


async def main() -> None:
    server = await asyncio.start_server(handle, "127.0.0.1", local_port)
    async with server:
        await server.serve_forever()


asyncio.run(main())
