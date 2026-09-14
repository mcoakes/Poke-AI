import importlib.util
from pathlib import Path

import pytest

from firered_rl.backends.mgba_bridge import MGBABridgeBackend, MGBAProtocolError


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "mgba" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_placeholder_observations_cannot_train():
    with pytest.raises(MGBAProtocolError, match="battle memory"):
        MGBABridgeBackend._state({"state_ready": False})


def test_bootstrap_configures_matching_state(tmp_path):
    launcher = load_script("launch_emulators")
    bootstrap = tmp_path / "bootstrap.lua"
    state = tmp_path / 'a "quoted" team.ss1'
    launcher.write_bootstrap(bootstrap, tmp_path / "adapter.lua", tmp_path / "profile.lua",
                             "firered_us_v1_0", 55356, state)
    content = bootstrap.read_text()
    assert "FIRERED_RL_PORT = 55356" in content
    assert 'a \\"quoted\\" team.ss1' in content
    assert content.index("FIRERED_RL_STATE_PATH") < content.index("dofile")


def test_reset_tester_detects_mismatch_and_closes(monkeypatch):
    tester = load_script("test_reset")
    class Bridge:
        closed = False
        frame = 0

        def status(self):
            return {"supported": True, "reset_configured": True}

        def reset_response(self):
            self.frame += 1
            return {"frame": self.frame, "ram_fnv1a": "12345678"}

        def step_response(self, *args):
            pass

        def close(self):
            self.closed = True

    bridge = Bridge()
    monkeypatch.setattr(tester, "MGBABridgeBackend", lambda *a, **kw: bridge)
    with pytest.raises(RuntimeError, match="reset differs"):
        tester.check_one(55355, 3)
    assert bridge.closed
