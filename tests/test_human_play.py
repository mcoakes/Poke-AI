from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

from firered_rl.human_play import play_battle
from firered_rl.linked_env import LinkedBattleSession, linked_observation
from firered_rl.link_inspection import pair_inspections
from test_linked_env import runtime


def enable_human(bridges, monkeypatch):
    kind = type(bridges[0])
    original = kind.status
    monkeypatch.setattr(kind, "status", lambda self: dict(original(self), link_human_supported=True))


@pytest.mark.parametrize("human", [0, 1])
def test_human_side_uses_observer_and_ai_side_only_uses_policy(runtime, monkeypatch, human):
    launch, _, _, bridges, state = runtime
    # Discover the test bridge type without launching a native process.
    from firered_rl import linked_env
    probe = linked_env.MGBABridgeBackend("127.0.0.1", 55355)
    enable_human(bridges, monkeypatch)
    bridges.clear()
    session = LinkedBattleSession(launch, human_player=human, timeout=0)
    original_status = type(probe).link_turn_status
    polls = [0]
    def status(bridge):
        polls[0] += 1
        if polls[0] <= 6:
            return dict(request_id=bridge.request_id, active=True, phase="observing" if bridge.index == human else "resolving")
        return original_status(bridge)
    monkeypatch.setattr(type(probe), "link_turn_status", status)
    seen = []
    class Opponent:
        def choose(self, report):
            seen.append(report)
            assert linked_observation(report)["action_mask"][0]
            return 0
    try:
        result = play_battle(session, Opponent(), human, announce=lambda _: None)
        assert len(seen) == 1 and seen[0]["battle"]["simulation_source_port"] == 55355
        assert bridges[human].payload["kind"] == "human"
        assert bridges[1-human].payload["kind"] == "move"
        assert bridges[1-human].payload["human_opponent"] is True
        assert result["human_outcome"] == ("lost" if human == 0 else "won")
        assert result["terminated"] and not result["truncated"]
        assert polls[0] > 6  # a human thinking interval does not use the AI-only wall timeout
        submitted = json.loads((session.runtime.run / "gym-steps.jsonl").read_text().splitlines()[1])
        assert submitted["actions"][human] is None
    finally:
        session.close()


def test_old_adapter_cannot_claim_human_input_support(runtime):
    launch, _, _, bridges, _ = runtime
    session = LinkedBattleSession(launch, human_player=0)
    try:
        with pytest.raises(RuntimeError, match="human play requires adapter 0.14.0"):
            session.reset_pair()
        assert all(not bridge.prepared for bridge in bridges)
    finally:
        session.close()


def test_ai_faint_returns_replacement_mask_while_human_keeps_gui_control(runtime, monkeypatch):
    launch, _, _, bridges, state = runtime
    from firered_rl import linked_env
    probe = linked_env.MGBABridgeBackend("127.0.0.1", 55355)
    enable_human(bridges, monkeypatch)
    bridges.clear()
    session = LinkedBattleSession(launch, human_player=1)
    try:
        session.reset_pair()
        def status(bridge):
            if bridge.index == 0:
                state[0]["party"][0]["current_hp"] = 0
                state[0]["battle"]["battlers"][0]["current_hp"] = 0
                state[0]["battle"]["menus"] = [dict(state="party", submenu="choose_pokemon")]
                return dict(request_id=bridge.request_id, phase="replacement_ready", active=False)
            return dict(request_id=bridge.request_id, phase="observing", active=True)
        monkeypatch.setattr(type(probe), "link_turn_status", status)
        result = session.step_pair([0, None])
        assert result[0][4]["reason"] == "forced_replacement"
        assert result[0][0]["action_mask"][5]
        assert not any(result[0][0]["action_mask"][:4])
        assert not result[0][2] and not result[0][3]
    finally:
        session.close()


def test_human_inspection_retries_party_index_in_flight(runtime, monkeypatch):
    launch, _, _, bridges, state = runtime
    from firered_rl import linked_env
    linked_env.MGBABridgeBackend("127.0.0.1", 55355)
    enable_human(bridges, monkeypatch)
    bridges.clear()
    session = LinkedBattleSession(launch, human_player=0)
    try:
        session.reset_pair()
        original = session.request
        calls = [0]
        def request(method):
            result = original(method)
            if method == "inspect":
                calls[0] += 1
                if calls[0] == 1:
                    result[1]["battle"]["battlers"][0]["party_slot"] = 2
            return result
        monkeypatch.setattr(session, "request", request)
        assert session.read_reports()[0]["battle"]["turn"] == 0
        assert calls[0] == 2
    finally:
        session.close()


def test_real_checkpoint_load_prediction_is_masked_and_does_not_train(tmp_path):
    torch = pytest.importorskip("torch")
    MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO
    from firered_rl.human_play import PPOOpponent
    from firered_rl.ppo_features import BattleFeatures
    from firered_rl.ppo_vec_env import LinkedVecEnv
    from test_ppo import Pair
    root = Path(__file__).parents[1]
    raw = json.loads((root / "tests/fixtures/link_master_follower.json").read_text())
    reports = pair_inspections(raw)
    torch.set_num_threads(1)
    vec = LinkedVecEnv([Pair([linked_observation(r) for r in reports])])
    try:
        model = MaskablePPO("MultiInputPolicy", vec, n_steps=2, batch_size=4, n_epochs=1,
                            policy_kwargs=dict(features_extractor_class=BattleFeatures))
        model.learn(4)
        checkpoint = tmp_path / "agent.zip"
        model.save(checkpoint)
        before_file = checkpoint.read_bytes()
        loaded = MaskablePPO.load(checkpoint, device="cpu")
        weights = {k: v.clone() for k, v in loaded.policy.state_dict().items()}
        for stochastic in (False, True):
            opponent = PPOOpponent(loaded, stochastic=stochastic)
            for report in reports:
                before = deepcopy(report)
                action = opponent.choose(report)
                assert linked_observation(report)["action_mask"][action]
                assert report == before
        assert loaded.num_timesteps == 4
        assert all(torch.equal(value, loaded.policy.state_dict()[k]) for k, value in weights.items())
        assert checkpoint.read_bytes() == before_file
        assert not loaded.policy.training
        # Forced continuation exposes only WAIT; it must never be interpreted as Run.
        reports[0]["battle"]["menus"] = [dict(state="busy_or_unknown")]
        assert opponent.choose(reports[0]) == 10
    finally:
        vec.close()


@pytest.mark.parametrize("human,fail_start", [(1, False), (2, False), (1, True)])
def test_cli_assigns_windows_loads_readonly_checkpoint_and_cleans_up(tmp_path, monkeypatch, human, fail_start):
    from types import SimpleNamespace
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("sb3_contrib")
    import gymnasium as gym
    from firered_rl.ppo_features import policy_space
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    spec = importlib.util.spec_from_file_location("human_cli", root / "mgba/play_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "mgba/play_agent.py"))
    monkeypatch.setattr(module, "validate_launch", lambda *args: None)
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"model bytes")
    states = [tmp_path / f"team{i}.ss1" for i in (1, 2)]
    for path in states:
        path.write_bytes(b"state")
    raw = json.loads((root / "tests/fixtures/link_master_follower.json").read_text())
    reports = pair_inspections(raw)
    model = SimpleNamespace(observation_space=policy_space(), action_space=gym.spaces.Discrete(11),
        policy=SimpleNamespace(set_training_mode=lambda mode: None), num_timesteps=64,
        set_random_seed=lambda seed: None,
        predict=lambda obs, action_masks, deterministic: (0, None))
    def load(stream, **kwargs):
        assert stream.read() == b"model bytes"
        return model
    monkeypatch.setattr(sb3.MaskablePPO, "load", load)
    processes = []
    class Runtime:
        def __init__(self, root, run, ports, states):
            self.run, self.ports, self.states = run, ports, states
            self.session_id, self.closed = "test", False
            processes.append(self)
        def start(self, rom, emulator, speed, headless):
            assert speed == 1 and not headless
            self.run.mkdir()
        def wait_ready(self, timeout):
            if fail_start:
                raise RuntimeError("test startup failure")
        def process_status(self): return {"state": "running"}
        def capture_failure(self, error): pass
        def close(self): self.closed = True
    def connect(ports, session, log, **kwargs):
        assert kwargs["start_battle"]
        kwargs["announce"]("battle_ready")
    class Session:
        def __init__(self, launch, human_player, max_steps):
            self.launch, self.human_player = launch, human_player
            self.steps = 0
        def reset_pair(self):
            self.runtime = self.launch()
            self.reports = deepcopy(reports)
        def step_pair(self, actions):
            assert actions[human-1] is None and actions[2-human] == 0
            self.steps = 1
            return [(None, 1 if i == human-1 else -1, True, False,
                     dict(outcome="won" if i == human-1 else "lost", turn_end=1, reason="battle_complete"))
                    for i in range(2)]
        def close(self): pass
    monkeypatch.setattr(module, "LinkProcess", Runtime)
    monkeypatch.setattr(module, "LinkedBattleSession", Session)
    monkeypatch.setattr(module, "connect_pair", connect)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    selection = ["--states", *map(str, states)]
    if human == 2:
        folder = tmp_path / "states/counter/profiles/casual/3"
        folder.mkdir(parents=True)
        for i in range(2):
            (folder / f"team{i}.ss1").write_bytes(b"state")
        selection = ["--team-profile", "casual", "--team-size", "3"]
    monkeypatch.setattr(sys, "argv", ["play_agent", "--rom", "FireRed.gba", "--checkpoint", str(checkpoint),
                        *selection, "--human-player", str(human)])
    assert module.main() == int(fail_start)
    assert processes[0].closed and checkpoint.read_bytes() == b"model bytes"
    run = processes[0].run.parent
    if fail_start:
        assert "test startup failure" in (run / "failure.json").read_text()
    else:
        assert json.loads((run / "result.json").read_text())["human_outcome"] == "won"
        config = json.loads((run / "config.json").read_text())
        assert config["training_timesteps"] == 64 and config["human_player"] == human
        if human == 2:
            assert config["state_source"]["team_profile"] == "casual"
            assert all(Path(p).parent.name == "3" for p in config["states"])
