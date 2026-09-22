"""Benchmark laya's Python agent on CPU, same state/preset as the Rust bench."""

import json
import pathlib
import time

from laya import Agent

ROOT = pathlib.Path(__file__).resolve().parent.parent
agent = Agent(str(ROOT / "cli" / "model" / "laya-en"), device="cpu")

state = {
    "subject": "Invoice #4471 is attached",
    "body": "Hi team, please find attached the invoice for our March order. The total is "
    "$4,120.00 and it looks like the PO number is missing. Can you confirm the due date? "
    "Thanks, Priya",
    "from": "priya@example.com",
}
qs = json.loads((ROOT / "cli" / "presets" / "email.json").read_text())


def bench(label: str, questions, n: int = 10) -> None:
    for _ in range(2):
        agent.predict(state, questions)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        agent.predict(state, questions)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    print(f"{label}: p50 {times[n // 2]:7.0f} ms  p95 {times[int(n * 0.95) % n]:7.0f} ms")


bench("predict(5 questions, one batched forward)", qs)
single = {k: qs[k] for k in ("category",)}
bench("predict(1 question)                    ", single)