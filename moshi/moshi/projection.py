"""Small-window nonlinear projection, executed in a separate CPU process."""

import time

import numpy as np


def diffusion_map(frames: np.ndarray, components: int = 3) -> np.ndarray:
    """Two or three nonconstant diffusion coordinates of a locally scaled kernel.

    Distances use the full residual vector. Alpha=1 density correction and
    symmetric normalisation produce the diffusion operator's eigenvectors.
    No temporal edges or random noise are added to manufacture a trajectory.
    """
    if components not in (2, 3):
        raise ValueError("Projection requires two or three components")
    x = np.asarray(frames, dtype=np.float64)
    count = len(x)
    if count < 4:
        return np.zeros((count, components))
    x = x - x.mean(axis=0)
    norms = np.einsum("ij,ij->i", x, x)
    distances = np.maximum(norms[:, None] + norms[None, :] - 2 * x @ x.T, 0)
    if distances.max() < 1e-12:
        return np.zeros((count, components))
    neighbour = min(7, count - 1)
    scale = np.partition(distances, neighbour, axis=1)[:, neighbour]
    floor = max(float(distances.max()) * 1e-9, 1e-12)
    bandwidth = np.sqrt(np.maximum(scale[:, None] * scale[None, :], floor * floor))
    kernel = np.exp(-distances / bandwidth)
    density = kernel.sum(axis=1)
    kernel /= density[:, None] * density[None, :]
    degree = kernel.sum(axis=1)
    inverse_root = 1 / np.sqrt(degree)
    symmetric = kernel * inverse_root[:, None] * inverse_root[None, :]
    values, vectors = np.linalg.eigh(symmetric)
    indices = np.argsort(values)[::-1][1:components + 1]
    return vectors[:, indices] * inverse_root[:, None] * values[indices]


def align_projection(points, ids, previous_points, previous_ids):
    """Rigid Procrustes alignment using the same frames in adjacent windows."""
    points = points - points.mean(axis=0)
    radius = np.max(np.linalg.norm(points, axis=1), initial=0)
    if radius > 1e-12:
        points = points / radius
    old_index = {frame: index for index, frame in enumerate(previous_ids)}
    common = [(index, old_index[frame]) for index, frame in enumerate(ids) if frame in old_index]
    if len(common) >= 4:
        new_indices, old_indices = zip(*common)
        source = points[list(new_indices)]
        target = previous_points[list(old_indices)]
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
        # Fix orientation while keeping every window centered and bounded.
        # Translating to the preceding overlap would accumulate camera drift.
        points = points @ (u @ vt)
    return points


def benchmark_projection(dimension, repeats=30):
    """Representative correlated windows; includes alignment, excludes IPC."""
    rng = np.random.default_rng(7123)
    directions = rng.standard_normal((5, dimension)).astype(np.float32)
    results = {}
    for count in (20, 50):
        t = np.linspace(0, 4, count)
        curve = np.column_stack((np.sin(t), np.cos(t), np.sin(2 * t), t, np.cos(3 * t)))
        frames = (curve @ directions + rng.standard_normal((count, dimension)) * 0.08).astype(np.float32)
        previous = diffusion_map(frames)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            points = diffusion_map(frames)
            align_projection(points, list(range(count)), previous, list(range(count)))
            timings.append((time.perf_counter() - start) * 1000)
        results[str(count)] = {
            "median_ms": round(float(np.median(timings)), 3),
            "p95_ms": round(float(np.percentile(timings, 95)), 3),
        }
    return results


def projection_worker(jobs, results, dimension, frame_rate):
    """One bounded job at a time; failures are reported instead of stopping audio."""
    previous_points = np.empty((0, 3))
    previous_ids = []
    previous_generation = None
    timings = benchmark_projection(dimension)
    # Reserve half a model frame for CPU projection, with a 5 Hz display cap.
    window = 50 if timings["50"]["p95_ms"] < 500 / frame_rate else 20
    interval = max(0.2, timings[str(window)]["p95_ms"] * 4 / 1000)
    results.put({"kind": "ready", "benchmark": timings, "recommended_window": window,
                 "update_interval": interval})
    while True:
        job = jobs.get()
        if job is None:
            return
        generation, ids, frames, submitted, components = job
        try:
            started = time.perf_counter()
            points = diffusion_map(frames, components)
            if previous_generation != generation:
                previous_points, previous_ids = np.empty((0, components)), []
            points = align_projection(points, ids, previous_points, previous_ids)
            previous_points, previous_ids = points, ids
            previous_generation = generation
            finished = time.perf_counter()
            results.put({
                "kind": "projection", "generation": generation, "frame_ids": ids,
                "points": points.round(5).tolist(),
                "components": components,
                "compute_ms": round((finished - started) * 1000, 3),
                "worker_ms": round((finished - submitted) * 1000, 3),
                "submitted": submitted,
            })
        except Exception as error:
            results.put({"kind": "error", "generation": generation, "error": str(error)})
