"""Lightweight, serialized client for the isolated prediction subprocess."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "INSPECTION_RESULT:"


class PredictionWorker:
    def __init__(self, idle_seconds=120):
        self.process = None
        self.idle_seconds = idle_seconds
        self.lock = threading.Lock()

    def _stop(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                process.stdin.close()  # EOF lets the Windows venv child exit too.
            except OSError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.kill()
                process.wait(timeout=15)
            process.stdout.close()

    def close(self):
        with self.lock:
            self._stop()

    def _start(self, env):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "course/predict.py"), "--worker", "--idle-seconds", str(self.idle_seconds)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def predict(self, request, log, env=None):
        with self.lock:
            # The worker may expire between poll() and write(); retry once after EOF.
            for attempt in range(2):
                if self.process is None or self.process.poll() is not None:
                    self._stop()
                    self._start(env)
                try:
                    self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                    self.process.stdin.flush()
                    for line in self.process.stdout:
                        if line.startswith(PREFIX):
                            result = json.loads(line[len(PREFIX):])
                            if not result["ok"]:
                                raise RuntimeError(result["error"])
                            log.write(json.dumps(result, ensure_ascii=False) + "\n")
                            log.flush()
                            return result
                        log.write(line)
                        log.flush()
                except (BrokenPipeError, OSError):
                    pass
                except Exception:
                    self._stop()
                    raise
                self._stop()
            raise RuntimeError("Prediction worker exited before returning a result")
