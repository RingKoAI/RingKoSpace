"""Streaming equivalence, gradient continuity and explicit cache ownership."""
from dataclasses import replace
import pytest
import torch
from ringkospace.model import RingKoSSM
from ringkospace._scan import scan_first_order


def make_model(variant=0, conv_k=4):
    torch.manual_seed(19)
    return RingKoSSM(12, 2, conv_k=conv_k, evolve=bool(variant), scan_floor=0.0)


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("chunk", [1, 7, 32])
@pytest.mark.parametrize("conv_k", [1, 4])
def test_stream_matches_full_prefix(variant, chunk, conv_k):
    model = make_model(variant, conv_k).eval()
    x = torch.randint(4, 260, (2, 37))
    with torch.no_grad():
        full, final = model(x)
        state, outputs = None, []
        for part in x.split(chunk, dim=1):
            output, state = model.forward_stream(part, state)
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, 1), full, atol=2e-6, rtol=2e-5)
        for expected, actual in zip(final, state.layers):
            torch.testing.assert_close(actual.recurrent, expected, atol=2e-6, rtol=2e-5)
            assert actual.convolution.shape == (2, model.dim, conv_k-1)


def test_stream_gradients_match_full_sequence():
    model = make_model(1)
    x = torch.randint(4, 260, (2, 13))
    full, states = model(x)
    loss = full.square().sum() + sum(s.square().sum() for s in states)
    expected = torch.autograd.grad(loss, tuple(model.parameters()))
    state, outputs = None, []
    for part in x.split(5, 1):
        output, state = model.forward_stream(part, state)
        outputs.append(output)
    loss = torch.cat(outputs, 1).square().sum() + sum(s.recurrent.square().sum() for s in state.layers)
    actual = torch.autograd.grad(loss, tuple(model.parameters()))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-4)
    assert all(s.recurrent.grad_fn is None and s.convolution.grad_fn is None for s in state.detach().layers)


def test_states_are_independent_and_reject_wrong_owner_or_shape():
    model = make_model()
    x = torch.randint(4, 260, (2, 9))
    with torch.no_grad():
        _, a = model.forward_stream(x[:, :4])
        saved = [(s.recurrent.clone(), s.convolution.clone()) for s in a.layers]
        y, _ = model.forward_stream(x[:, 4:], a)
        model.forward_stream(x.flip(1))
        z, _ = model.forward_stream(x[:, 4:], a)
        torch.testing.assert_close(y, z)
        for s, (r, c) in zip(a.layers, saved):
            assert torch.equal(s.recurrent, r) and torch.equal(s.convolution, c)
        with pytest.raises(ValueError, match="another model"):
            make_model().forward_stream(x, a)
        with pytest.raises(ValueError, match="shape"):
            model.forward_stream(x[:1], a)
        with pytest.raises(ValueError, match="layer count"):
            model.forward_stream(x, replace(a, layers=()))
        with pytest.raises(ValueError, match="nonempty"):
            model.forward_stream(x[:, :0])


@pytest.mark.parametrize("length", [1, 31, 32, 33, 97])
def test_scan_zero_tiny_gates_and_all_gradients(length):
    torch.manual_seed(3)
    keep = (torch.rand(2, length, 7) * 1e-4).requires_grad_()
    with torch.no_grad():
        keep[:, ::3] = 0
    inc = torch.randn_like(keep, requires_grad=True)
    initial = torch.randn(2, 7, requires_grad=True)
    h = initial
    values = []
    for k, w in zip(keep.unbind(1), inc.unbind(1)):
        h = k * h + w
        values.append(h)
    reference = torch.stack(values, 1)
    output, carry = scan_first_order(keep, inc, initial)
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(carry, h)
    inputs = (keep, inc, initial)
    actual = torch.autograd.grad(output.square().sum() + carry.square().sum(), inputs)
    expected = torch.autograd.grad(reference.square().sum() + h.square().sum(), inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_invalid_scan_inputs():
    with pytest.raises(ValueError):
        scan_first_order(torch.empty(1, 0, 4), torch.empty(1, 0, 4))
    with pytest.raises(ValueError):
        scan_first_order(torch.ones(1, 3, 4), torch.ones(1, 3, 5))
    with pytest.raises(ValueError):
        scan_first_order(torch.ones(1, 3, 4), torch.ones(1, 3, 4), torch.zeros(1, 5))



def test_single_token_zero_initial_has_zero_keep_gradient():
    keep = torch.ones(1, 1, 1, requires_grad=True)
    inc = torch.ones_like(keep, requires_grad=True)
    output, _ = scan_first_order(keep, inc)
    grad = torch.autograd.grad(output.sum(), keep)[0]
    assert torch.equal(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_generation_stream_uses_complete_state(precision):
    from ringkospace.generate import generate
    model = make_model().eval()
    prompt = [40, 41, 42, 43]
    streamed, _ = generate(model, prompt, 4, 1.0, 1, 0, 32, mode="stream", device="cpu", precision=precision)
    windowed, _ = generate(model, prompt, 4, 1.0, 1, 0, 32, mode="window", device="cpu", precision=precision)
    assert len(streamed) == 4
    assert streamed == windowed
