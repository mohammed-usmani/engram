"""Score extraction on the fixed test set (src/eval/extraction.py) for one provider/model.

    uv run python scripts/eval_extraction.py dashscope deepseek-v4-flash-0731
    uv run python scripts/eval_extraction.py ollama qwen2.5-coder:7b --concurrency 1

Makes real LLM calls (about 20 cases); writes nothing. The Evaluation page runs the same set and keeps history.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: E402,F401  loads .env
from src.eval.extraction import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel calls (use 1 for local Ollama)")
    a = ap.parse_args()
    score, results, summary = asyncio.run(run(a.provider, a.model, a.concurrency))
    for r in results:
        status = f"ERROR {r['error']}" if r["error"] else ("ok" if not r["failed"] else "fail: " + ", ".join(r["failed"]))
        print(f"{r['case']:42} {status}")
    print(f"\n{summary['provider']}:{summary['model']}  {summary['passed']}/{summary['total']} checks ({score}%)  "
          f"median {summary['median_s']}s · max {summary['max_s']}s · errors {summary['errors']}")


if __name__ == "__main__":
    main()
