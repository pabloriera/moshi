"""Graph-safe activation capture and nonblocking runtime telemetry."""

from collections import deque
import multiprocessing as mp
import os
from queue import Empty, Full
import time

import numpy as np
import torch

from .projection import projection_worker


class RuntimeTelemetry:
    def __init__(self, transformer, dimension, frame_rate):
        parameter = next(transformer.parameters())
        self.layer_count = len(transformer.layers)
        self.dimension = dimension
        self.frame_rate = frame_rate
        self.layer = max(1, self.layer_count // 2)
        self.window = 50
        self.components = 3
        self.explicit_window = False
        self.enabled = True
        # Capture before warmup. Hooks remain tensor-only inside CUDA graphs.
        # One fixed vector per layer permits live selection without recapture.
        self.activations = torch.zeros(self.layer_count, dimension, device=parameter.device)
        self.handles = [layer.register_forward_hook(self._hook(index))
                        for index, layer in enumerate(transformer.layers)]
        self.frames = deque(maxlen=50)
        self.frame_ids = deque(maxlen=50)
        self.step = 0
        self.generation = 0
        self.in_flight = False
        self.last_submit = 0.0
        self.update_interval = 0.2
        self.benchmark = None
        self.error = None
        self.process = None
        context = mp.get_context("spawn")
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=2)
        self.context = context

    def _hook(self, index):
        def capture(_module, _args, output):
            self.activations[index].copy_(output[0, -1].detach())
        return capture

    def start(self):
        if self.process is not None:
            return
        # Set only the child process's native BLAS limits. No worker GPU work.
        names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        saved = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = "1"
            self.process = self.context.Process(
                target=projection_worker,
                args=(self.jobs, self.results, self.dimension, self.frame_rate), daemon=True,
            )
            self.process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def reset(self):
        self.generation += 1
        self.frames.clear()
        self.frame_ids.clear()
        self.step = 0
        self.last_submit = 0.0

    def update(self, params):
        previous = (self.layer, self.window, self.enabled)
        previous_components = self.components
        layer = params.get("layer")
        if isinstance(layer, int) and not isinstance(layer, bool):
            self.layer = max(1, min(self.layer_count, layer))
        if params.get("window") in (20, 50) and not isinstance(params.get("window"), bool):
            self.window = int(params["window"])
            self.explicit_window = True
        if isinstance(params.get("enabled"), bool):
            self.enabled = params["enabled"]
        if params.get("components") in (2, 3) and not isinstance(params.get("components"), bool):
            self.components = int(params["components"])
        if previous != (self.layer, self.window, self.enabled):
            self.reset()
        elif previous_components != self.components:
            # Reproject the same collected frames with a fresh orientation.
            self.generation += 1
            self.last_submit = 0.0
        return self.status()

    def status(self):
        return {"layer": self.layer, "layer_count": self.layer_count, "dimension": self.dimension,
                "window": self.window, "components": self.components,
                "frame_rate": self.frame_rate, "enabled": self.enabled,
                "method": "diffusion", "benchmark": self.benchmark,
                "update_hz": round(1 / self.update_interval, 2), "error": self.error}

    def poll(self):
        messages = []
        while True:
            try:
                result = self.results.get_nowait()
            except Empty:
                break
            if result["kind"] == "ready":
                self.benchmark = result["benchmark"]
                if not self.explicit_window:
                    self.window = result["recommended_window"]
                self.update_interval = result["update_interval"]
                messages.append({"telemetry_status": self.status()})
            else:
                self.in_flight = False
                if result["generation"] != self.generation:
                    continue
                if result["kind"] == "error":
                    self.error = result["error"]
                    messages.append({"telemetry_status": self.status()})
                elif self.enabled:
                    result.pop("kind")
                    result.pop("generation")
                    result["latency_ms"] = round((time.perf_counter() - result.pop("submitted")) * 1000, 3)
                    result.update(layer=self.layer, window=self.window, method="diffusion")
                    messages.append({"trajectory": result})
        if self.process is not None and not self.process.is_alive() and self.error is None:
            self.error = "Projection worker stopped"
            self.in_flight = False
            messages.append({"telemetry_status": self.status()})
        return messages

    def measure(self, entropy):
        self.step += 1
        # A single small GPU->CPU transfer contains entropies and the selected
        # residual vector. No projection or wait for worker results here.
        norms = torch.linalg.vector_norm(self.activations, dim=-1)
        metrics = torch.cat((entropy, norms))
        values = torch.cat((metrics, self.activations[self.layer - 1])) if self.enabled else metrics
        values = values.detach().float().cpu().numpy()
        bits = values[:len(entropy)]
        audio = bits[1:].tolist()
        measurement = {"frame": self.step, "text_bits": float(bits[0]),
                       "audio_bits": audio, "audio_mean_bits": float(np.mean(audio)) if audio else 0.0,
                       "layer_norms": values[len(entropy):len(metrics)].tolist()}
        if self.enabled:
            self.frames.append(values[len(metrics):].copy())
            self.frame_ids.append(self.step)
            now = time.perf_counter()
            if (len(self.frames) >= self.window and not self.in_flight and self.benchmark is not None
                    and self.error is None and now - self.last_submit >= self.update_interval):
                ids = list(self.frame_ids)[-self.window:]
                frames = np.stack(list(self.frames)[-self.window:])
                try:
                    self.jobs.put_nowait((self.generation, ids, frames, now, self.components))
                    self.in_flight = True
                    self.last_submit = now
                except Full:
                    pass
        return measurement

    def close(self):
        # A pending activation window may exceed the pipe buffer. Do not wait
        # for its feeder thread after terminating the consumer on shutdown.
        self.jobs.cancel_join_thread()
        self.results.cancel_join_thread()
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=2)
        self.jobs.close()
        self.results.close()
        for handle in self.handles:
            handle.remove()
