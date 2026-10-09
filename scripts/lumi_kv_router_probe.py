"""Check native llm-d routing against two controlled cache-occupancy endpoints."""

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from jinja2 import Environment, FileSystemLoader


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    loads = [0.9, 0.1]
    servers = []
    processes = []
    logs = []
    results = []
    root = Path(__file__).resolve().parents[1]
    for worker in range(2):

        def handler_for(worker):
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    if self.path == "/metrics":
                        body = (
                            "# TYPE vllm:kv_cache_usage_perc gauge\n"
                            f'vllm:kv_cache_usage_perc{{model_name="probe"}} {loads[worker]}\n'
                            "# TYPE vllm:num_requests_running gauge\n"
                            'vllm:num_requests_running{model_name="probe"} 0\n'
                            "# TYPE vllm:num_requests_waiting gauge\n"
                            'vllm:num_requests_waiting{model_name="probe"} 0\n'
                        ).encode()
                        kind = "text/plain; version=0.0.4"
                    else:
                        body = b"{}"
                        kind = "application/json"
                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_POST(self):
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    body = json.dumps(
                        {
                            "id": "probe",
                            "object": "chat.completion",
                            "model": "probe",
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                            ],
                            "token_ids": [42],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("x-probe-worker", str(worker))
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            return Handler

        server = ThreadingHTTPServer(("127.0.0.1", 19511 + worker), handler_for(worker))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    env = Environment(loader=FileSystemLoader(root / "src/prime_rl/templates"))
    router = {"scorers": {"kv-cache-utilization-scorer": 3.0, "queue-scorer": 2.0, "active-request-scorer": 1.0}}
    if args.prefix:
        router["scorers"]["prefix-cache-scorer"] = 1.0
    epp = env.get_template("llmd/epp_estimate.yaml.j2").render(router=router).replace("$LLMD_DIR", str(args.output))
    (args.output / "epp.yaml").write_text(epp)
    (args.output / "endpoints.yaml").write_text(
        "endpoints:\n"
        + "".join(
            f'  - name: worker-{i}\n    address: 127.0.0.1\n    port: "{19511 + i}"\n    namespace: default\n    labels:\n      llm-d.ai/pool: prime-rl\n'
            for i in range(2)
        )
    )
    envoy = (
        env.get_template("llmd/envoy.yaml.j2")
        .render(router_port=19300)
        .replace("port_value: 9901", "port_value: 19901")
        .replace("port_value: 9002", "port_value: 19502")
    )
    (args.output / "envoy.yaml").write_text(envoy)
    commands = [
        [
            "epp",
            "--pool-name",
            "prime-rl",
            "--pool-namespace",
            "slurm",
            "--config-file",
            str(args.output / "epp.yaml"),
            "--grpc-port",
            "19502",
            "--grpc-health-port",
            "19513",
            "--metrics-port",
            "19590",
            "--secure-serving=false",
            "--metrics-endpoint-auth=false",
            "--tracing=false",
        ],
        ["envoy", "-c", str(args.output / "envoy.yaml"), "--concurrency", "2"],
    ]
    try:
        for cmd in commands:
            log = (args.output / (cmd[0] + ".log")).open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [str(root / "third_party/llmd/bin" / cmd[0]), *cmd[1:]], stdout=log, stderr=subprocess.STDOUT
                )
            )
        with httpx.Client(timeout=20) as client:
            for _ in range(60):
                assert all(p.poll() is None for p in processes), "Router component exited; inspect logs"
                try:
                    if client.get("http://127.0.0.1:19901/ready").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            for preferred in [1, 0]:
                loads[:] = [0.9, 0.1] if preferred == 1 else [0.1, 0.9]
                time.sleep(10)
                for path, body in [
                    (
                        "/v1/chat/completions",
                        {"model": "probe", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1},
                    ),
                    (
                        "/inference/v1/generate",
                        {"model": "probe", "token_ids": [10, 20, 30], "sampling_params": {"max_tokens": 1}},
                    ),
                ]:
                    counts = [0, 0]
                    for _ in range(20):
                        r = client.post("http://127.0.0.1:19300" + path, json=body)
                        r.raise_for_status()
                        counts[int(r.headers["x-probe-worker"])] += 1
                    row = {"path": path, "preferred": preferred, "counts": counts}
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    assert counts[preferred] >= 18, row
            if args.prefix:
                loads[:] = [0.1, 0.1]
                time.sleep(10)
                for path, body in [
                    (
                        "/v1/chat/completions",
                        {
                            "model": "probe",
                            "messages": [{"role": "user", "content": "shared prefix " * 4096}],
                            "max_tokens": 1,
                        },
                    ),
                    (
                        "/inference/v1/generate",
                        {"model": "probe", "token_ids": list(range(1000, 5096)), "sampling_params": {"max_tokens": 1}},
                    ),
                ]:
                    counts = [0, 0]
                    for _ in range(20):
                        r = client.post("http://127.0.0.1:19300" + path, json=body)
                        r.raise_for_status()
                        counts[int(r.headers["x-probe-worker"])] += 1
                    row = {"path": path, "prefix_affinity": True, "counts": counts}
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    assert max(counts) == 20, row
                    owner = counts.index(20)
                    loads[owner] = 0.95
                    time.sleep(10)
                    r = client.post("http://127.0.0.1:19300" + path, json=body)
                    r.raise_for_status()
                    assert int(r.headers["x-probe-worker"]) != owner, "Prefix affinity must yield to high KV occupancy"
                    loads[:] = [0.1, 0.1]
                    time.sleep(10)
        (args.output / "result.json").write_text(json.dumps({"passed": True, "results": results}, indent=2))
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for server in servers:
            server.shutdown()
        for log in logs:
            log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
