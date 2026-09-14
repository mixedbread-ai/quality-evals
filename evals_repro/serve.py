import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from evals_repro.local_rerank import LOCAL_MODELS


def serve(args) -> None:
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "Install GPU dependencies with uv sync --group gpu, then run uv run --group gpu evals-repro serve-reranker"
        )
    spec = LOCAL_MODELS[args.model]
    processes = []
    with TemporaryDirectory(prefix="reranker-") as directory:
        template = Path(directory) / "chat.jinja"
        template.write_text(spec.template)
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            spec.model,
            "--revision",
            spec.revision,
            "--runner",
            "pooling",
            "--dtype",
            "bfloat16",
            "--hf-overrides",
            json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": spec.classifier,
                    "method": spec.method,
                    "num_labels": 1,
                }
            ),
            "--chat-template",
            str(template),
            "--host",
            args.host,
            "--max-model-len",
            str(spec.context),
            "--max-num-batched-tokens",
            str(spec.context),
            "--max-num-seqs",
            "128",
            "--gpu-memory-utilization",
            str(args.memory_fraction),
            "--enforce-eager",
            "--no-enable-prefix-caching",
        ]
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            for offset, device in enumerate(args.devices):
                processes.append(
                    subprocess.Popen(
                        [*command, "--port", str((args.port or spec.port) + offset)],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                        start_new_session=True,
                    )
                )
            while all(p.poll() is None for p in processes):
                time.sleep(1)
            raise RuntimeError("A reranker server exited; see its output above")
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
