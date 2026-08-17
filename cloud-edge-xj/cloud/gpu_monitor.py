import statistics
import subprocess
import threading
from collections import deque
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Tuple


class GpuSampler:
    def __init__(self, gpu_index: int, interval_seconds: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_seconds = max(0.1, interval_seconds)
        self._samples: List[Tuple[float, float, float]] = []
        self._errors: List[str] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    str(self.gpu_index),
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            line = completed.stdout.strip().splitlines()[0]
            utilization, memory_used, memory_total = (
                float(value.strip()) for value in line.split(",")
            )
            self._samples.append((utilization, memory_used, memory_total))
        except Exception as exc:
            message = str(exc)
            if message not in self._errors:
                self._errors.append(message)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
        utilization = [sample[0] for sample in self._samples]
        memory_used = [sample[1] for sample in self._samples]
        memory_total = self._samples[-1][2] if self._samples else None
        return {
            "gpu_index": self.gpu_index,
            "sample_count": len(self._samples),
            "utilization_samples_percent": utilization,
            "memory_used_samples_mib": memory_used,
            "average_utilization_percent": statistics.fmean(utilization)
            if utilization
            else None,
            "peak_utilization_percent": max(utilization) if utilization else None,
            "average_memory_used_mib": statistics.fmean(memory_used)
            if memory_used
            else None,
            "peak_memory_used_mib": max(memory_used) if memory_used else None,
            "memory_total_mib": memory_total,
            "errors": self._errors,
        }


class ContinuousGpuMonitor:
    """云端网关级单例采样器，为每个评测 run 提供同一时间轴上的 GPU 指标。"""

    def __init__(self, gpu_index: int, interval_seconds: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_seconds = max(0.1, interval_seconds)
        self._samples = deque(maxlen=200_000)
        self._errors: List[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _query(gpu_index: int) -> Tuple[float, float, float]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(gpu_index),
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        line = completed.stdout.strip().splitlines()[0]
        return tuple(float(value.strip()) for value in line.split(","))

    def start(self) -> None:
        self._thread.start()

    def capture_sample(self) -> Optional[int]:
        """同步采一帧并返回时间戳，用于准确覆盖 run 的起止边界。"""
        try:
            utilization, memory_used, memory_total = self._query(self.gpu_index)
            sampled_ns = perf_counter_ns()
            with self._lock:
                self._samples.append(
                    (sampled_ns, utilization, memory_used, memory_total)
                )
            return sampled_ns
        except Exception as exc:
            message = str(exc)
            with self._lock:
                if message not in self._errors:
                    self._errors.append(message)
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.capture_sample()
            self._stop_event.wait(self.interval_seconds)

    def summary(self, started_ns: int, finished_ns: int) -> Dict[str, Any]:
        with self._lock:
            samples = sorted(
                (
                    sample
                    for sample in self._samples
                    if started_ns <= sample[0] <= finished_ns
                ),
                key=lambda sample: sample[0],
            )
            errors = list(self._errors)
        utilization = [sample[1] for sample in samples]
        memory_used = [sample[2] for sample in samples]
        gpu_seconds = 0.0
        for index, sample in enumerate(samples):
            next_ns = samples[index + 1][0] if index + 1 < len(samples) else finished_ns
            duration_seconds = max(0.0, (next_ns - sample[0]) / 1_000_000_000)
            gpu_seconds += sample[1] / 100.0 * duration_seconds
        window_seconds = max(0.0, (finished_ns - started_ns) / 1_000_000_000)
        average_utilization = (
            gpu_seconds / window_seconds * 100.0
            if utilization and window_seconds > 0
            else None
        )
        return {
            "gpu_index": self.gpu_index,
            "scope": "完整评测运行窗口内的指定 GPU 主机级指标",
            "sample_count": len(samples),
            "average_utilization_percent": average_utilization,
            "p95_utilization_percent": _percentile(utilization, 95),
            "peak_utilization_percent": max(utilization) if utilization else None,
            "gpu_seconds": gpu_seconds if utilization else None,
            "average_memory_used_mib": statistics.fmean(memory_used)
            if memory_used
            else None,
            "peak_memory_used_mib": max(memory_used) if memory_used else None,
            "memory_total_mib": samples[-1][3] if samples else None,
            "errors": errors,
        }

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=4)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
