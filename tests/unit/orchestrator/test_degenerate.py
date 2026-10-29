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


def test_repetition_stopped_turns_are_judged_before_the_loop():
    def looped(prefix_lp):
        tokens = list(range(1000, 1300)) + [7 + i % 20 for i in range(700)]
        return SimpleNamespace(
            sampled=True, token_ids=tokens, mask=[True] * len(tokens), logprobs=[prefix_lp] * 300 + [-0.01] * 700
        )

    t = trace(node([-0.5] * 300), looped(-0.5), looped(-8.0))
    t.calls = [
        SimpleNamespace(node=i, finish_reason=f, error=None)
        for i, f in enumerate(["tool_calls", "repetition", "repetition"])
    ]
    turn = find_degenerate_turn(t, min_tokens=100, max_mean_nll=3.0)
    assert turn is not None and turn.index == 2 and turn.tokens == 300 and round(turn.mean_nll, 1) == 8.0
    t.calls[2].finish_reason = "length"  # judged whole, the loop dilutes it below the threshold
    assert find_degenerate_turn(t, min_tokens=100, max_mean_nll=3.0) is None
