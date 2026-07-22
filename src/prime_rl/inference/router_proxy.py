"""Small OpenAI-compatible HTTP proxy for external-LB vLLM workers."""

import argparse
import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

logger = logging.getLogger("prime_rl.inference.router_proxy")


class WorkerRouter:
    def __init__(self, worker_urls: list[str], policy: str) -> None:
        if not worker_urls:
            raise ValueError("At least one worker URL is required")
        self.worker_urls = [url.rstrip("/") for url in worker_urls]
        self.policy = policy
        self._next_worker = 0
        self._lock = asyncio.Lock()

    async def choose(self, request: Request) -> str:
        if self.policy == "consistent_hash":
            session_id = request.headers.get("x-session-id")
            if session_id:
                digest = hashlib.blake2b(session_id.encode(), digest_size=8).digest()
                return self.worker_urls[int.from_bytes(digest, "big") % len(self.worker_urls)]

        async with self._lock:
            worker = self.worker_urls[self._next_worker % len(self.worker_urls)]
            self._next_worker += 1
            return worker


def _forward_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}
    }


async def _wait_for_workers(client: httpx.AsyncClient, worker_urls: list[str], timeout_seconds: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    pending = set(worker_urls)
    while pending and asyncio.get_running_loop().time() < deadline:
        ready = set()
        for worker in pending:
            try:
                response = await client.get(f"{worker.rstrip('/')}/v1/models", timeout=5.0)
                if response.status_code < 500:
                    ready.add(worker)
            except Exception:
                pass
        pending -= ready
        if pending:
            await asyncio.sleep(5.0)
    if pending:
        raise TimeoutError(f"Timed out waiting for workers: {sorted(pending)}")


def create_app(worker_urls: list[str], policy: str, worker_startup_timeout_secs: float) -> FastAPI:
    app = FastAPI()
    router = WorkerRouter(worker_urls, policy)
    client = httpx.AsyncClient(timeout=None)

    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Starting fallback router for workers: %s", ", ".join(worker_urls))
        await _wait_for_workers(client, worker_urls, worker_startup_timeout_secs)
        logger.info("All fallback router workers are ready")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request) -> Response:
        worker = await router.choose(request)
        target = f"{worker}/{path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"

        body = await request.body()
        try:
            upstream_request = client.build_request(
                request.method,
                target,
                content=body,
                headers=_forward_headers(request),
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("Worker request failed (%s): %s", target, exc)
            return JSONResponse({"error": f"worker request failed: {exc}"}, status_code=502)

        async def stream_response() -> AsyncIterator[bytes]:
            async for chunk in upstream_response.aiter_raw():
                yield chunk

        return StreamingResponse(
            stream_response(),
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response),
            background=BackgroundTask(upstream_response.aclose),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback router for external-LB vLLM workers")
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument("--policy", default="consistent_hash", choices=["consistent_hash", "round_robin"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-startup-timeout-secs", type=float, default=4200.0)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    app = create_app(args.worker_urls, args.policy, args.worker_startup_timeout_secs)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
