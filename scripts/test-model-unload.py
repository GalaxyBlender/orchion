#!/usr/bin/env python3
"""Exercise real inference and model eviction on an isolated CUDA server."""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


CASES = {
    "asr": ("asr", "alibaba/qwen3-asr-0.6b", "/v1/audio/transcriptions"),
    "tts": ("tts", "alibaba/qwen3-tts-12hz-0.6b-customvoice", "/v1/audio/speech"),
    "ocr": ("ocr", "paddlepaddle/pp-ocrv6-tiny", "/v1/ocr"),
    "ocr-vl": ("ocr_vl", "paddlepaddle/paddleocr-vl-1.6", "/v1/ocr"),
    "llm": ("llm", "alibaba/qwen3.5-0.8b", "/v1/chat/completions"),
    "embedding": ("llm", "qwen/qwen3-embedding-0.6b", "/v1/embeddings"),
}


def process_gpu_mib(pid):
    if os.name == "nt":
        # WDDM reports N/A for per-process memory in nvidia-smi.
        command = f"""
$ErrorActionPreference = 'Stop'
Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
  Where-Object Name -like 'pid_{pid}_*' |
  ForEach-Object {{ '{{0}},{{1}}' -f $_.Name,([long]($_.DedicatedUsage / 1MB)) }}
"""
        rows = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
        )
        return {name: int(mib) for name, mib in csv.reader(rows.splitlines())}
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        gpu: int(mib)
        for gpu, owner, mib in csv.reader(rows.splitlines(), skipinitialspace=True)
        if owner == str(pid)
    }


def run(args):
    service, default_model, route = CASES[args.case]
    model = args.model or default_model

    def request(path, *options):
        result = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(args.wait),
                *options,
                args.url.rstrip("/") + path,
            ],
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"{path}: {result.stderr.decode(errors='replace')}"
                f"{result.stdout[:500].decode(errors='replace')}"
            )
        return result.stdout

    def json_request(path, body):
        return request(path, "--json", json.dumps(body))

    def statuses():
        return {
            (row["service"], row["id"]): row["status"]
            for row in json.loads(request("/api/models/status"))["data"]
        }

    def infer(index):
        fields = {"model": model}
        text = [
            "Hello from Orchion.",
            "This is a slightly longer CUDA test.",
            "Do the next requests still work?",
        ][index % 3]
        if args.case in ("asr", "ocr", "ocr-vl"):
            paths = args.audio if args.case == "asr" else args.image
            fields["response_format"] = (
                "markdown" if args.case != "asr" and index % 2 else "json"
            )
            if args.case == "ocr-vl":
                fields["max_tokens"] = "32"
            options = [
                part
                for key, value in fields.items()
                for part in ("--form-string", f"{key}={value}")
            ]
            return request(
                route, *options, "--form", f"file=@{paths[index % len(paths)]}"
            )
        if args.case == "tts":
            fields.update(input=text, voice="ryan", language="english", max_length=64)
        elif args.case == "llm":
            fields.update(messages=[{"role": "user", "content": text}], max_tokens=16)
        else:
            fields.update(input=text)
        return json_request(route, fields)

    key = (service, model)
    initial = statuses()
    if initial.get(key) != "unloaded" or any(
        value != "unloaded" for value in initial.values()
    ):
        raise RuntimeError("start with an isolated server, all its models unloaded")
    baseline = process_gpu_mib(args.pid)
    first_unload = None
    for round_number in range(1, args.rounds + 1):
        for index in range(args.requests):
            if not infer(index):
                raise RuntimeError(f"{model} returned an empty HTTP body")
            if statuses()[key] != "loaded":
                raise RuntimeError(
                    f"{model} was not resident immediately after request {index + 1}"
                )
        loaded = process_gpu_mib(args.pid)
        if not any(mib > baseline.get(gpu, 0) + 64 for gpu, mib in loaded.items()):
            raise RuntimeError(
                f"{model}: no GPU allocations observed after inference: {loaded}"
            )
        if args.explicit:
            json_request("/api/models/unload", {"model": model, "service": service})
        deadline = time.monotonic() + args.wait
        while any(value != "unloaded" for value in statuses().values()):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{model}: server did not finish unloading")
            time.sleep(0.25)
        while True:
            retired = process_gpu_mib(args.pid)
            within_baseline = all(
                mib <= baseline.get(gpu, 0) + args.retained_mib
                for gpu, mib in retired.items()
            )
            stable = first_unload is None or all(
                mib <= first_unload.get(gpu, 0) + 128 for gpu, mib in retired.items()
            )
            if within_baseline and stable:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{model} round {round_number}: GPU MiB remain after unload: "
                    f"baseline={baseline}, loaded={loaded}, unloaded={retired}"
                )
            time.sleep(0.25)
        print(
            f"{model} round {round_number}: GPU MiB baseline={baseline}, "
            f"loaded={loaded}, unloaded={retired}",
            flush=True,
        )
        if first_unload is None:
            first_unload = retired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--pid", required=True, type=int, help="isolated server PID on this machine"
    )
    parser.add_argument("--model", help="override the default model ID for the case")
    parser.add_argument(
        "--audio", type=Path, action="append", default=[], help="speech WAV"
    )
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--requests", type=int, default=4, help="inferences before each unload"
    )
    parser.add_argument(
        "--wait", type=float, default=60, help="request/eviction timeout in seconds"
    )
    parser.add_argument(
        "--retained-mib", type=int, default=256, help="allowed runtime overhead per GPU"
    )
    parser.add_argument(
        "--explicit", action="store_true", help="use the unload API instead of idle TTL"
    )
    args = parser.parse_args()
    if (
        args.rounds < 2
        or args.requests < 2
        or args.pid <= 0
        or args.wait <= 0
        or args.retained_mib < 0
    ):
        parser.error(
            "need >=2 rounds and requests, positive PID/timeout, and nonnegative allowance"
        )
    if args.case == "asr" and not args.audio:
        parser.error("ASR requires --audio")
    if args.case in ("ocr", "ocr-vl") and not args.image:
        parser.error("OCR requires --image")
    run(args)


if __name__ == "__main__":
    main()
