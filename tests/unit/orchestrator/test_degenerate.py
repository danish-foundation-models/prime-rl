from types import SimpleNamespace

from prime_rl.orchestrator.degenerate import find_degenerate_turn


def node(logprobs, sampled=True):
    return SimpleNamespace(sampled=sampled, logprobs=logprobs)


def trace(*nodes):
    return SimpleNamespace(nodes=list(nodes))


def test_flags_a_long_near_random_turn():
    t = trace(node([], sampled=False), node([-0.5] * 300), node([-8.0] * 4000))
    turn = find_degenerate_turn(t, min_tokens=100, max_mean_nll=3.0)
    assert turn is not None and turn.index == 1 and turn.tokens == 4000 and round(turn.mean_nll, 1) == 8.0


def test_ignores_normal_and_short_turns():
    t = trace(node([-0.6] * 2000), node([-9.0] * 50), node([-2.9] * 500))
    assert find_degenerate_turn(t, min_tokens=100, max_mean_nll=3.0) is None


def test_ignores_unsampled_nodes_and_missing_logprobs():
    t = trace(node([-9.0] * 500, sampled=False), node(None), node([None] * 200))
    assert find_degenerate_turn(t, min_tokens=100, max_mean_nll=3.0) is None
