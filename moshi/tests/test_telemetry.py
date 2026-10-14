import time

import numpy as np
import pytest
import torch
from torch import nn

from moshi.models.lm import LMGen, LMModel
from moshi.projection import align_projection, diffusion_map
from moshi.telemetry import RuntimeTelemetry
from moshi.utils.compile import CUDAGraphed, no_compile


@pytest.mark.parametrize("components", [2, 3])
def test_nonlinear_projection_is_finite_deterministic_and_handles_constant_frames(components):
    t = np.linspace(0, 4 * np.pi, 50)
    frames = np.column_stack((np.sin(t), np.cos(t), t / 10, np.sin(2 * t)))
    points = diffusion_map(frames, components)
    assert points.shape == (50, components)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points, diffusion_map(frames, components))
    assert np.linalg.matrix_rank(points) == components
    np.testing.assert_array_equal(diffusion_map(np.ones((20, 16)), components), np.zeros((20, components)))


@pytest.mark.parametrize("components", [2, 3])
def test_alignment_corrects_arbitrary_eigenvector_rotation_and_reflection(components):
    points = np.random.default_rng(5).normal(size=(20, components))
    reference = align_projection(points, list(range(20)), np.empty((0, components)), [])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])[:components, :components]
    rotated = points @ rotation
    aligned = align_projection(rotated, list(range(20)), reference, list(range(20)))
    np.testing.assert_allclose(aligned, reference, atol=1e-10)


class ToyTransformer(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16, device=device) for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_capture_replays_live_activations_without_changing_model_output():
    model = ToyTransformer("cuda")
    x = torch.ones(1, 1, 16, device="cuda")
    baseline = model(x)
    monitor = RuntimeTelemetry(model, 16, 12.5)
    try:
        forward = CUDAGraphed(model)
        forward(x)
        torch.testing.assert_close(forward(x), baseline)
        graph = forward._graph
        output = forward(x * 2)
        torch.testing.assert_close(monitor.activations[-1], output[0, -1])
        monitor.update({"layer": 2, "window": 20})
        measurement = monitor.measure(torch.tensor([1.0, 2.0, 3.0], device="cuda"))
        assert measurement["audio_bits"] == [2.0, 3.0]
        assert measurement["audio_mean_bits"] == 2.5
        np.testing.assert_allclose(measurement["layer_norms"], monitor.activations.norm(dim=-1).cpu().numpy())
        np.testing.assert_allclose(monitor.frames[-1], output[0, -1].cpu().numpy())
        assert forward._graph is graph
    finally:
        monitor.close()


@torch.no_grad()
def test_norms_cover_every_layer_and_continue_when_projection_is_paused():
    monitor = RuntimeTelemetry(ToyTransformer(), 16, 12.5)
    try:
        monitor.activations[0].fill_(3)
        monitor.activations[1].fill_(4)
        measurement = monitor.measure(torch.tensor([1.0, 2.0]))
        assert measurement["layer_norms"] == [12.0, 16.0]
        # The projection buffer must contain the original residual vector,
        # excluding the additional norm metrics in the transfer packet.
        np.testing.assert_array_equal(monitor.frames[-1], np.full(16, 3))
        monitor.update({"enabled": False})
        monitor.activations[0].fill_(5)
        monitor.activations[1].zero_()
        measurement = monitor.measure(torch.tensor([0.0, 0.0]))
        assert measurement["layer_norms"] == [20.0, 0.0]
        assert measurement["text_bits"] == 0.0
        assert measurement["audio_bits"] == [0.0]
        assert not monitor.frames
    finally:
        monitor.close()


@torch.no_grad()
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_every_lm_sampling_procedure_has_its_own_entropy_slot(device):
    model = LMModel(
        delays=[0, 1, 1], n_q=2, dep_q=2, card=32, text_card=48,
        dim=16, num_layers=2, num_heads=1, hidden_scale=1,
        depformer_dim=16, depformer_num_heads=1, context=4,
        depformer_weights_per_step=True,
        depformer_gating="silu",
        device=device,
    ).eval()
    generator = LMGen(model, measure_entropy=True, top_k=5, top_k_text=3)
    with no_compile(), generator.streaming(1):
        for _ in range(4):
            generator.step(torch.zeros(1, 2, 1, dtype=torch.long, device=device))
            entropy = generator.sampling_entropy
            assert entropy.shape == (3,)
            assert (entropy >= 0).all()
            assert entropy[0] <= np.log2(3) + 1e-5
            assert (entropy[1:] <= np.log2(5) + 1e-5).all()
            assert (entropy > 0).all()


@torch.no_grad()
@pytest.mark.parametrize("components", [2, 3])
def test_worker_returns_projection_and_reset_drops_old_generation(components):
    monitor = RuntimeTelemetry(ToyTransformer(), 16, 12.5)
    try:
        monitor.start()
        deadline = time.monotonic() + 20
        while monitor.benchmark is None and time.monotonic() < deadline:
            monitor.poll()
            time.sleep(0.02)
        assert monitor.benchmark is not None
        monitor.update({"window": 20, "components": components})
        for step in range(20):
            monitor.activations[monitor.layer - 1].copy_(torch.sin(torch.arange(16) + step / 4))
            monitor.measure(torch.tensor([1.0, 2.0]))
        assert monitor.in_flight
        messages = []
        while not messages and time.monotonic() < deadline:
            messages = monitor.poll()
            time.sleep(0.01)
        trajectory = next(message["trajectory"] for message in messages if "trajectory" in message)
        assert len(trajectory["points"]) == 20
        assert trajectory["components"] == components
        assert all(len(point) == components for point in trajectory["points"])
        assert trajectory["frame_ids"] == list(range(1, 21))
        assert trajectory["latency_ms"] >= trajectory["compute_ms"]
        monitor.last_submit = 0
        monitor.measure(torch.tensor([1.0, 2.0]))
        assert monitor.in_flight
        monitor.reset()
        while monitor.in_flight and time.monotonic() < deadline:
            assert not any("trajectory" in message for message in monitor.poll())
            time.sleep(0.01)
        assert not monitor.in_flight
        assert not monitor.frames
    finally:
        monitor.close()


@torch.no_grad()
def test_switching_dimensions_preserves_collected_frames_and_discards_old_projection():
    monitor = RuntimeTelemetry(ToyTransformer(), 16, 12.5)
    try:
        for _ in range(20):
            monitor.measure(torch.tensor([1.0, 2.0]))
        frames = list(monitor.frames)
        ids = list(monitor.frame_ids)
        generation = monitor.generation
        status = monitor.update({"components": 2})
        assert status["components"] == 2
        assert monitor.generation == generation + 1
        assert list(monitor.frame_ids) == ids
        assert all(np.array_equal(before, after) for before, after in zip(frames, monitor.frames))
        monitor.update({"components": 4})
        assert monitor.components == 2
    finally:
        monitor.close()
