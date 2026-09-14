from copy import deepcopy
import pytest

from firered_rl.linked_reward import EpisodeRewardState, LinkedRewardConfig, reward_points


@pytest.fixture
def reports():
    return [dict(party=[dict(slot=1, species_id=6, valid=True, is_egg=False, current_hp=100, max_hp=100),
                       dict(slot=2, species_id=9, valid=True, is_egg=False, current_hp=200, max_hp=200)],
                 battle=dict(turn=0, outcome="ongoing", battlers=[dict(side="player", valid=True, party_slot=1,
                             current_hp=100, max_hp=100)])) for _ in range(2)]


def damage(report, hp):
    report["battle"]["battlers"][0]["current_hp"] = hp


def test_burst_damage_beats_split_damage_and_ignores_stale_party_copy(reports):
    cfg = LinkedRewardConfig(damage=100, knockout=500, win=5000, loss=-5000)
    half, end = deepcopy(reports), deepcopy(reports)
    damage(half[1], 50)
    damage(end[1], 0)
    assert half[1]["party"][0]["current_hp"] == 100  # active overlay must win
    assert reward_points(reports, half, cfg, False)[0] == [25, 0]
    burst = reward_points(reports, end, cfg, False)[0][0]
    split = reward_points(reports, half, cfg, False)[0][0] + reward_points(half, end, cfg, False)[0][0]
    assert burst == 600 and split == 550
    assert reward_points(end, deepcopy(end), cfg, False)[0] == [0, 0]  # KO paid once


def test_win_stacks_with_last_knockout_and_loss_is_negative(reports):
    cfg = LinkedRewardConfig(damage=100, knockout=500, win=5000, loss=-5000)
    end = deepcopy(reports)
    damage(end[1], 0)
    end[0]["battle"]["outcome"], end[1]["battle"]["outcome"] = "won", "lost"
    points, parts = reward_points(reports, end, cfg, True)
    assert points == [5600, -5000]
    assert parts[0] == dict(damage=100, knockout=500, outcome=5000)
    assert reward_points(reports, end, cfg, False)[0] == [600, 0]  # no terminal bonus on truncation


def test_switches_and_healing_do_not_fabricate_damage(reports):
    cfg = LinkedRewardConfig(damage=100, knockout=500)
    after = deepcopy(reports)
    after[1]["battle"]["battlers"][0].update(party_slot=2, current_hp=200, max_hp=200)
    assert reward_points(reports, after, cfg, False)[0] == [0, 0]
    before = deepcopy(reports)
    damage(before[1], 50)
    assert reward_points(before, reports, cfg, False)[0] == [0, 0]
    after[1]["party"][1]["species_id"] = 999
    with pytest.raises(ValueError, match="party identity"):
        reward_points(reports, after, cfg, False)


@pytest.mark.parametrize("kwargs", [dict(scale=0), dict(damage=True), dict(knockout=-1),
                                   dict(win=float('nan')), dict(damage_exponent=.5),
                                   dict(turn_penalty=-1), dict(damage_cap_per_pokemon=-1),
                                   dict(turn_penalty=True), dict(damage_cap_per_pokemon=float('inf'))])
def test_invalid_reward_config_rejected(kwargs):
    with pytest.raises(ValueError):
        LinkedRewardConfig(**kwargs)


def test_thirty_turn_win_beats_sixty_turn_win_by_1500(reports):
    cfg = LinkedRewardConfig(damage=100, knockout=500, win=5000, loss=-5000,
                             turn_penalty=50, damage_cap_per_pokemon=100)
    def play(length):
        previous = deepcopy(reports)
        state = EpisodeRewardState(previous)
        total = [0, 0]
        for n in range(length):
            current = deepcopy(previous)
            terminal = n == length-1
            for i, r in enumerate(current):
                r["battle"]["turn"] = n if terminal else n+1
                if terminal:
                    r["battle"]["outcome"] = "won" if i == 0 else "lost"
            if terminal:
                damage(current[1], 0)
                current[1]["party"][1]["current_hp"] = 0
            values, _ = reward_points(previous, current, cfg, terminal, state)
            total = [a+b for a,b in zip(total, values)]
            previous = current
        assert state.turns == length
        return total
    short, long = play(30), play(60)
    assert short == [4700, -6500]  # win 5000 + two KOs 1000 + damage 200 - turns 1500
    assert long == [3200, -8000]
    assert short[0]-long[0] == 1500


def test_rest_cannot_reset_damage_budget_and_knockout_still_pays(reports):
    cfg = LinkedRewardConfig(damage=100, knockout=500, turn_penalty=50, damage_cap_per_pokemon=100)
    state = EpisodeRewardState(reports)
    previous, damage_total, total = deepcopy(reports), 0, 0
    for n in range(60):
        current = deepcopy(previous)
        damage(current[1], 1 if n % 2 == 0 else 100)
        for r in current:
            r["battle"]["turn"] = n+1
        points, parts = reward_points(previous, current, cfg, False, state)
        damage_total += parts[0]["damage"]
        total += points[0]
        previous = current
    assert damage_total == pytest.approx(100)
    assert total == pytest.approx(-2900)
    assert state.damage_paid[0][1] == 100
    current = deepcopy(previous)
    damage(current[1], 0)
    _, parts = reward_points(previous, current, cfg, False, state)
    assert parts[0]["damage"] == 0 and parts[0]["knockout"] == 500


def test_recipient_decisions_do_not_double_charge_and_terminal_turn_is_included(reports):
    cfg = LinkedRewardConfig(turn_penalty=50)
    state = EpisodeRewardState(reports)
    # Start native turn 0, pause for a recipient, then finish the same turn.
    _, parts = reward_points(reports, reports, cfg, False, state)
    assert parts[0]["turn_penalty"] == -50
    _, parts = reward_points(reports, reports, cfg, False, state)
    assert parts[0]["turn_penalty"] == 0
    next_turn = deepcopy(reports)
    for r in next_turn:
        r["battle"]["turn"] = 1
    _, parts = reward_points(reports, next_turn, cfg, False, state)
    assert parts[0]["turn_penalty"] == 0
    _, parts = reward_points(next_turn, next_turn, cfg, True, state)
    assert parts[0]["turn_penalty"] == -50 and state.turns == 2
    with pytest.raises(ValueError, match="already ended"):
        reward_points(next_turn, next_turn, cfg, True, state)


def test_damage_budgets_are_separate_for_slots_sides_and_episodes(reports):
    cfg = LinkedRewardConfig(damage=100, damage_cap_per_pokemon=100)
    state = EpisodeRewardState(reports)
    current = deepcopy(reports)
    for r in current:
        damage(r, 0)
        r["party"][1]["current_hp"] = 0
    assert reward_points(reports, current, cfg, False, state)[0] == [200, 200]
    assert state.damage_paid == [{1:100, 2:100}, {1:100, 2:100}]
    other_episode = EpisodeRewardState(reports)
    assert reward_points(reports, current, cfg, False, other_episode)[0] == [200, 200]
    with pytest.raises(ValueError, match="per-episode"):
        reward_points(reports, current, cfg, False)


def test_automatic_turns_and_truncation_have_cost_but_no_terminal_bonus(reports):
    cfg = LinkedRewardConfig(turn_penalty=50, win=5000)
    state = EpisodeRewardState(reports)
    current = deepcopy(reports)
    for r in current:
        r["battle"]["turn"] = 3
    points, parts = reward_points(reports, current, cfg, False, state)
    assert points == [-150, -150] and state.turns == 3
    assert all(p["outcome"] == 0 for p in parts)
