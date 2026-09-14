"""Production PPO/Actor autograd checks for trajectory weights and 15 ranks.

Ray, FSDP communication and model forwarding are substituted.  The production
PPO clipping primitive, masked mean, and Actor update method run unchanged.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
_ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location("recovery_actor_fixture", Path(__file__).with_name("test_entropy_actor.py"))
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)
Batch, Proto, Config = _FIXTURE.Batch, _FIXTURE.Proto, _FIXTURE.Config


def _production_function(path, names, environment):
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), environment)
    return environment


_MEAN = _production_function("verl/utils/torch_functional.py", {"masked_mean"}, {"torch": torch})["masked_mean"]
_CORE = _production_function("verl/trainer/ppo/core_algos.py", {"agg_loss", "compute_policy_loss"},
                             {"torch": torch, "verl_F": SimpleNamespace(masked_mean=_MEAN)})


def _log_probs(parameter, batch):
    # Different sampled tokens give different gradients for different actions.
    distribution = parameter.log_softmax(-1)
    return distribution[batch["responses"].long()]


def _data():
    # First trajectory has two actions (lengths 1 and 3); second has one
    # length-2 action. Their combined weighted token counts both equal 3.
    responses = torch.tensor([[0, 0, 0, 0], [1, 2, 1, 0], [2, 0, 0, 0], [0, 1, 2, 0]])
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)
    batch = Batch(responses=responses,
                  input_ids=responses.clone(), position_ids=torch.zeros_like(responses),
                  attention_mask=mask,
                  advantages=torch.tensor([.2, .2, .2, -.3], dtype=torch.float64)[:, None].expand(4, 4).clone(),
                  policy_action_weight=torch.tensor([.75, .75, 1.5, 1.], dtype=torch.float64))
    initial = torch.tensor([.2, -.3, .4], dtype=torch.float64)
    old = _log_probs(initial, batch).detach()
    # Include clipped and unclipped positive/negative token contributions.
    batch["old_log_probs"] = old + torch.tensor([0., -.4, .3, -.1], dtype=torch.float64)[:, None]
    return batch


def _take(batch, indices):
    return Batch({key: value[indices].clone() for key, value in batch.items()})


def _reference_gradient(batch, *, weighted=True):
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .4], dtype=torch.float64))
    mask = batch["attention_mask"]
    if weighted:
        mask = mask * batch["policy_action_weight"].detach()[:, None]
    loss = _CORE["compute_policy_loss"](
        batch["old_log_probs"], _log_probs(parameter, batch), batch["advantages"], mask,
        cliprange=.2, cliprange_low=.2, cliprange_high=.2, clip_ratio_c=3., loss_agg_mode="token-mean")[0]
    loss.backward()
    return parameter.grad.detach().clone()


def _update(batch, *, micro_size=1, enabled=True, dynamic_chunks=None):
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .4], dtype=torch.float64))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    captured = []
    ppo_calls = []

    def ppo(**kwargs):
        result = _CORE["compute_policy_loss"](**kwargs)
        ppo_calls.append((kwargs["response_mask"].detach().clone(), result[0].detach().clone()))
        return result

    def forward(micro_batch, temperature, calculate_entropy):
        assert not calculate_entropy
        return None, _log_probs(parameter, micro_batch)

    def repack(batch, max_token_len):
        return [_take(batch, indices) for indices in dynamic_chunks], dynamic_chunks

    def step():
        captured.append(parameter.grad.detach().clone())
        return parameter.grad.norm().detach()

    actor = SimpleNamespace(
        actor_module=model, actor_optimizer=optimizer, _forward_micro_batch=forward,
        _optimizer_step=step, ulysses_sequence_parallel_size=1,
        config=Config(use_kl_loss=False, use_adaptive_ppo_mini_batch_size=False,
                      ppo_mini_batch_size=len(batch), ppo_micro_batch_size_per_gpu=micro_size,
                      ppo_max_token_len_per_gpu=64, ppo_epochs=1,
                      use_dynamic_bsz=dynamic_chunks is not None,
                      clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None,
                      entropy_coeff=0., loss_agg_mode="token-mean",
                      entropy_control={"enabled": False}, advantage_recovery={"enable": enabled}))
    method = _FIXTURE.actor_method("update_policy", compute_policy_loss=ppo,
                                   rearrange_micro_batches=repack, agg_loss=_CORE["agg_loss"])
    metrics = method(actor, Proto(batch, {"temperature": 1.}, {"wg_id": ["solver"] * len(batch)}))
    assert len(captured) == 1
    return captured[0], metrics, ppo_calls


@pytest.mark.parametrize("micro_size", [1, 2, 4])
def test_weighted_production_ppo_matches_single_global_token_mean(micro_size):
    batch = _data()
    actual, _, _ = _update(batch, micro_size=micro_size)
    torch.testing.assert_close(actual, _reference_gradient(batch), rtol=1e-7, atol=1e-9)


def test_dynamic_partition_and_order_preserve_weighted_gradient():
    batch = _data()
    actual, _, _ = _update(batch, dynamic_chunks=[[2], [3, 0, 1]])
    torch.testing.assert_close(actual, _reference_gradient(batch), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("micro_size", [1, 2, 4, 8])
def test_padding_only_microbatches_have_zero_finite_policy_gradient(micro_size):
    original = _data()
    padded = _take(original, [0, 1, 2, 3, 0, 0, 1, 2])
    padded["policy_action_weight"][4:] = 0
    actual, metrics, calls = _update(padded, micro_size=micro_size)
    torch.testing.assert_close(actual, _reference_gradient(original), rtol=1e-7, atol=1e-9)
    assert all(torch.isfinite(loss) for _, loss in calls)
    assert all(torch.isfinite(torch.tensor(values)).all() for values in metrics.values())
    if micro_size <= 4:
        assert any(not mask.any() and loss.item() == 0 for mask, loss in calls)


def test_entire_padding_rank_can_backpropagate_zero():
    batch = _data()
    batch["policy_action_weight"][:] = 0
    actual, _, _ = _update(batch, micro_size=1)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


def test_15_rank_gradient_average_matches_all_unique_actions(monkeypatch):
    original = _data()
    # Fifteen ranks with four rows each; most rows are padding and eleven ranks
    # have no real policy tokens. FSDP must still include these ranks in average.
    all_rows = _take(original, [0, 1, 2, 3] + [0] * 56)
    all_rows["policy_action_weight"][4:] = 0
    total_mass = (all_rows["attention_mask"] * all_rows["policy_action_weight"][:, None]).sum().item()
    all_reduce_calls = []

    def all_reduce(value):
        all_reduce_calls.append(value.detach().clone())
        value.fill_(total_mass)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 15)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    gradients = []
    for rank in range(15):
        # Interleave to distribute the four true actions to four separate ranks.
        local = _take(all_rows, list(range(rank, 60, 15)))
        gradient, _, _ = _update(local, micro_size=1)
        gradients.append(gradient)
    averaged = torch.stack(gradients).mean(0)
    torch.testing.assert_close(averaged, _reference_gradient(original), rtol=2e-7, atol=1e-9)
    assert len(all_reduce_calls) == 15
    assert sum(value.item() > 0 for value in all_reduce_calls) == 4


def test_disabled_path_uses_original_microbatch_mean_and_no_weight_field():
    batch = _data()
    del batch["policy_action_weight"]
    actual, _, _ = _update(batch, micro_size=2, enabled=False)
    expected = torch.stack([_reference_gradient(part, weighted=False) for part in batch.split(2)]).mean(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_disabled_path_ignores_accidental_weight_field():
    batch = _data()
    original, _, _ = _update(batch, micro_size=2, enabled=False)
    batch["policy_action_weight"][:] = float("nan")
    unchanged, _, _ = _update(batch, micro_size=2, enabled=False)
    torch.testing.assert_close(original, unchanged, rtol=0, atol=0)


def test_policy_weights_do_not_receive_gradients():
    batch = _data()
    weights = batch["policy_action_weight"].requires_grad_()
    _update(batch, micro_size=1)
    assert weights.grad is None


@pytest.mark.parametrize("bad", ["missing", "nan", "negative", "matrix"])
def test_enabled_actor_rejects_missing_or_invalid_weights(bad):
    batch = _data()
    if bad == "missing":
        del batch["policy_action_weight"]
    elif bad == "nan":
        batch["policy_action_weight"][0] = float("nan")
    elif bad == "negative":
        batch["policy_action_weight"][0] = -1
    else:
        batch["policy_action_weight"] = batch["policy_action_weight"][:, None]
    with pytest.raises(ValueError, match="policy_action_weight"):
        _update(batch)


def test_old_log_prob_does_not_require_not_yet_computed_recovery_weights():
    batch = _data()
    del batch["policy_action_weight"]
    actor = SimpleNamespace(actor_module=SimpleNamespace(eval=lambda: None),
                            ulysses_sequence_parallel_size=1,
                            config=Config(advantage_recovery={"enable": True}))
    actor._forward_micro_batch = lambda micro_batch, **kwargs: (None, micro_batch["responses"].float())
    method = _FIXTURE.actor_method("compute_log_prob")
    result, _ = method(actor, Proto(batch, {"micro_batch_size": 2, "temperature": 1., "use_dynamic_bsz": False}))
    torch.testing.assert_close(result, batch["responses"].float())
