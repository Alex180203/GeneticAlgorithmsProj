#!/usr/bin/env python3
"""
Ask Gemini and/or ChatGPT "What is the time?" every minute for 4 hours.

Each answer is printed and appended to a CSV together with your real local
time, so you can compare what the model said with what the time actually was.

Setup:
    pip install requests
    export GEMINI_API_KEY="..."      # Windows PowerShell: $env:GEMINI_API_KEY="..."
    export OPENAI_API_KEY="..."

Run:
    python ask_time.py                       # uses every provider that has a key set
    python ask_time.py --provider gemini
    python ask_time.py --provider openai --interval 60 --duration 240
    python ask_time.py --gemini-model gemini-flash-latest --openai-model gpt-5.4-mini

Stop early with Ctrl+C; everything sent so far is already saved.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import requests

QUESTION = "What is the time?"
TIMEOUT_S = 30

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"  # alias that tracks Google's newest Flash-Lite
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def ask_gemini(question: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"contents": [{"role": "user", "parts": [{"text": question}]}]},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def ask_openai(question: str, model: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": question}]},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def build_providers(choice: str, gemini_model: str, openai_model: str):
    """Return a list of (name, model, callable) for the providers to query."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    providers = []

    if choice in ("gemini", "both", "auto"):
        if gemini_key:
            providers.append(("gemini", gemini_model,
                              lambda q, m=gemini_model: ask_gemini(q, m, gemini_key)))
        elif choice != "auto":
            sys.exit("GEMINI_API_KEY is not set.")

    if choice in ("openai", "both", "auto"):
        if openai_key:
            providers.append(("openai", openai_model,
                              lambda q, m=openai_model: ask_openai(q, m, openai_key)))
        elif choice != "auto":
            sys.exit("OPENAI_API_KEY is not set.")

    if not providers:
        sys.exit("No API key found. Set GEMINI_API_KEY and/or OPENAI_API_KEY.")
    return providers


def describe_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = exc.response.text.replace("\n", " ")[:300]
        return f"HTTP {exc.response.status_code}: {body}"
    return f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Ask LLM APIs the time on a schedule.")
    parser.add_argument("--provider", choices=["auto", "gemini", "openai", "both"], default="auto",
                        help="auto = every provider with a key set (default)")
    parser.add_argument("--interval", type=float, default=60, help="seconds between rounds (default 60)")
    parser.add_argument("--duration", type=float, default=240, help="total minutes to run (default 240 = 4h)")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--out", default=None, help="CSV file (default: time_answers_<start>.csv)")
    args = parser.parse_args()

    providers = build_providers(args.provider, args.gemini_model, args.openai_model)
    out_path = args.out or f"time_answers_{datetime.now():%Y%m%d_%H%M%S}.csv"
    total_rounds = int(args.duration * 60 // args.interval)

    print(f"Asking {', '.join(f'{n} ({m})' for n, m, _ in providers)}")
    print(f"{total_rounds} rounds, every {args.interval:g}s, for {args.duration:g} min -> {out_path}")
    print("Ctrl+C to stop.\n")

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["round", "local_time_sent", "provider", "model",
                             "status", "latency_ms", "answer_or_error"])

        start = time.monotonic()
        try:
            for i in range(total_rounds):
                # Schedule against the start time so slow responses don't make the loop drift.
                wait = start + i * args.interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

                for name, model, ask in providers:
                    sent_at = datetime.now().astimezone()
                    t0 = time.perf_counter()
                    try:
                        answer, status = ask(args.question), "ok"
                    except Exception as exc:  # keep going no matter what one call does
                        answer, status = describe_error(exc), "error"
                    latency = int((time.perf_counter() - t0) * 1000)

                    writer.writerow([i + 1, sent_at.isoformat(timespec="seconds"),
                                     name, model, status, latency, answer])
                    f.flush()

                    one_line = answer.replace("\n", " ")
                    print(f"[{i + 1:>3}/{total_rounds}] {sent_at:%H:%M:%S} {name:<6} "
                          f"{status:<5} {latency:>5} ms | {one_line[:150]}")
        except KeyboardInterrupt:
            print("\nStopped by user.")

    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()