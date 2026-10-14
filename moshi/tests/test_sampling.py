import torch
import math
import pytest

from moshi.utils.sampling import sample_token
from moshi.utils.compile import CUDAGraphed


def test_sample_token_bias_promotes_selected_token():
    logits = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    token = sample_token(
        logits,
        use_sampling=True,
        temp=1.0,
        top_k=1,
        bias_token_id=2,
        bias=2.0,
    )

    assert token.item() == 2


def test_sample_token_bias_demotes_selected_token():
    logits = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    token = sample_token(
        logits,
        use_sampling=True,
        temp=1.0,
        top_k=1,
        bias_token_id=3,
        bias=-2.0,
    )

    assert token.item() != 3


@pytest.mark.parametrize("kwargs,expected", [
    ({}, 2.0), ({"top_k": 2}, 1.0), ({"top_p": 0.49}, 1.0),
    ({"top_k": 1}, 0.0), ({"temp": 0.0}, 0.0), ({"use_sampling": False}, 0.0),
])
def test_entropy_of_actual_sampling_support(kwargs, expected):
    entropy = torch.full((1,), -1.0)
    params = {"use_sampling": True, "temp": 1.0, **kwargs}
    sample_token(torch.zeros(1, 4), entropy_out=entropy, **params)
    assert entropy.item() == pytest.approx(expected)


def test_entropy_after_temperature_and_pad_bias_preserves_rng():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    params = {"use_sampling": True, "temp": 0.5, "bias_token_id": 0, "bias": 5.0, "top_k": 3}
    entropy = torch.zeros(1)
    torch.manual_seed(99)
    baseline = sample_token(logits, **params)
    rng_after = torch.random.get_rng_state()
    torch.manual_seed(99)
    measured = sample_token(logits, entropy_out=entropy, **params)
    assert torch.equal(baseline, measured)
    assert torch.equal(rng_after, torch.random.get_rng_state())
    p = torch.softmax(torch.tensor([5.0, 4.0, 6.0]), dim=-1)
    expected = -(p * p.log2()).sum()
    assert entropy.item() == pytest.approx(expected.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_entropy_updates_during_cuda_graph_replay():
    entropy = torch.zeros(1, device="cuda")
    def sample(logits):
        return sample_token(logits, use_sampling=True, top_k=3, entropy_out=entropy)
    graphed = CUDAGraphed(sample)
    x = torch.zeros(1, 4, device="cuda")
    graphed(x)
    graphed(x)
    assert entropy.item() == pytest.approx(math.log2(3))
    graph = graphed._graph
    graphed(torch.tensor([[30.0, 0.0, 0.0, 0.0]], device="cuda"))
    assert entropy.item() < 1e-8
    assert graphed._graph is graph
