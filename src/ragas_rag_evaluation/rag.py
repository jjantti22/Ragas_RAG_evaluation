import os
import json
import time
import asyncio
import hashlib
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()


class LiteratureExplorerRAG:
    def __init__(self, log_dir: str = "evals/logs", max_concurrency: int = 2):
        self.base_url = os.getenv("SUT_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("SUT_API_KEY", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.log_dir / "cache.json"
        self.raw_file = self.log_dir / "raw_responses.jsonl"
        self.cache = json.loads(self.cache_file.read_text(encoding="utf-8")) if self.cache_file.exists() else {}

    async def post_raw(self, client: httpx.AsyncClient, route: str, payload: dict, use_cache: bool = True) -> dict:
        key = hashlib.sha256(f"{route}:{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()
        if use_cache and key in self.cache:
            return self.cache[key]

        async with self.semaphore:
            for wait in (3, 6, 12):
                t0 = time.perf_counter()
                resp = await client.post(f"{self.base_url}{route}", headers=self.headers, json=payload, timeout=60.0)
                latency = round(time.perf_counter() - t0, 3)

                if resp.status_code in (429, 502):
                    await asyncio.sleep(wait)
                    continue

                body = resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text
                record = {"route": route, "request": payload, "status": resp.status_code, "latency": latency, "body": body}

                if use_cache and resp.status_code == 200:
                    self.cache[key] = record
                    self.cache_file.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

                with self.raw_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                await asyncio.sleep(0.4)
                return record

        raise RuntimeError(f"Endpoint failed after retries: {route}")

    async def query(self, question: str, use_cache: bool = True) -> dict:
        async with httpx.AsyncClient() as client:
            ctx_res, chat_res = await asyncio.gather(
                self.post_raw(client, "/v1/context", {"query": question}, use_cache=use_cache),
                self.post_raw(client, "/v1/chat/completions", {"messages": [{"role": "user", "content": question}]}, use_cache=use_cache),
            )

        chat_body = chat_res["body"] if isinstance(chat_res["body"], dict) else {}
        answer = chat_body.get("choices", [{}])[0].get("message", {}).get("content", "")

        # /v1/context may return a list of chunks or a dict wrapping 'chunks'/'results'/'contexts'
        ctx_body = ctx_res["body"]
        if isinstance(ctx_body, list):
            raw_chunks = ctx_body
        elif isinstance(ctx_body, dict):
            raw_chunks = (
                ctx_body.get("chunks")
                or ctx_body.get("results")
                or ctx_body.get("contexts")
                or ctx_body.get("documents")
                or []
            )
        else:
            raw_chunks = []

        retrieved_docs = []
        for c in raw_chunks:
            if isinstance(c, dict):
                retrieved_docs.append({
                    "content": c.get("text") or c.get("content") or str(c),
                    "source": c.get("source") or c.get("metadata", {}).get("source", ""),
                    "score": c.get("score", 0.0),
                })
            else:
                retrieved_docs.append({"content": str(c), "source": "", "score": 0.0})

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "chat_latency": chat_res["latency"],
            "context_latency": ctx_res["latency"],
            "status": chat_res["status"],
        }