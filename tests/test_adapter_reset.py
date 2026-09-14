"""Lua callback tests with a fake emulator/socket, not a live game test.

Run with: pip install -e '.[dev,lua-test]'
"""
import json
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa.lua54")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter(request):
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute('''
        C = { GBA_KEY = {A=0,B=1,UP=6,DOWN=7,LEFT=5,RIGHT=4,
                         START=3,SELECT=2,L=9,R=8},
              CHECKSUM = {CRC32=0}, SAVESTATE = {ALL=31} }
        callbacks = {handlers={}}
        function callbacks:add(name, fn) self.handlers[name] = fn end
        console = {log=function() end, error=function() end}
        socket = {ERRORS={AGAIN=1}}
        server = {handlers={}}
        function socket.bind() return server end
        function server:listen() return true end
        function server:add(name, fn) self.handlers[name] = fn end
        function server:close() end
        client = {handlers={}, queue={}, responses={}}
        function client:add(name, fn) self.handlers[name] = fn end
        function client:receive() return table.remove(self.queue,1), socket.ERRORS.AGAIN end
        function client:send(payload) table.insert(self.responses, payload); return #payload end
        function client:close() self.closed = true end
        function server:accept() return client end
        emu = {frame=800, keys=0, supported=true, load_ok=true, loads=0, mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a) | (self:read8(a+1) << 8) end
        function emu:read32(a) return self:read16(a) | (self:read16(a+2) << 16) end
        function emu:getGameCode() return self.supported and "AGB-BPRE" or "AGB-XXXX" end
        function emu:checksum() return string.char(0xDD,0x88,0x76,0x1C) end
        function emu:setKeys(keys) self.keys = keys end
        function emu:currentFrame() return self.frame end
        function emu:loadStateFile(path, flags)
            self.loads = self.loads + 1
            self.flags = flags
            if not self.load_ok then return false end
            self.frame = 123
            return true
        end
        function emu:readRange(address, length) return string.rep("x", length) end
        function request(line)
            table.insert(client.queue, line .. "\\n")
            client.handlers.received()
        end
        function tick()
            emu.frame = emu.frame + 1
            callbacks.handlers.frame()
        end
        FIRERED_RL_STATE_PATH = "/test/team.ss1"
    ''')
    lua.globals().FIRERED_RL_PROFILE_PATH = str(ROOT / "mgba/profiles/firered_us_v1_0.lua")
    lua.globals().FIRERED_RL_NATIVE_LINK_SESSION = getattr(request, "param", False)
    lua.execute((ROOT / "mgba/adapter.lua").read_text())
    lua.execute("server.handlers.received()")
    return lua


def send(lua, **payload):
    lua.globals().request(json.dumps(payload))


def last(lua):
    responses = lua.globals().client.responses
    return json.loads(responses[len(responses)])


def test_reset_is_deferred_and_repeatable(adapter):
    for _ in range(3):
        count = adapter.globals().emu.loads
        send(adapter, command="reset", seed=123)
        assert adapter.globals().emu.loads == count
        adapter.globals().tick()
        result = last(adapter)
        assert result["ok"] and result["frame"] == 123
        assert result["seed_applied"] is False
        assert result["state_ready"] is False
        expected = 2166136261
        for _ in range(0x48000):
            expected = ((expected ^ ord("x")) * 16777619) & 0xFFFFFFFF
        assert result["ram_fnv1a"] == f"{expected:08x}"
        assert adapter.globals().emu["keys"] == 0
        assert adapter.globals().emu.flags == 31
        send(adapter, command="step", action=1, frames=2)
        adapter.globals().tick()
        adapter.globals().tick()
        assert last(adapter)["frame_end"] - last(adapter)["frame_start"] == 2


def test_overlapping_step_and_reset_rejected(adapter):
    send(adapter, command="step", action=1, frames=2)
    send(adapter, command="reset")
    assert not last(adapter)["ok"]
    adapter.globals().tick()
    adapter.globals().tick()
    send(adapter, command="reset")
    send(adapter, command="step", action=1, frames=1)
    assert not last(adapter)["ok"]


def test_load_failure_and_unsupported_rom(adapter):
    adapter.globals().emu.load_ok = False
    send(adapter, command="reset")
    adapter.globals().tick()
    assert not last(adapter)["ok"]
    assert adapter.globals().emu["keys"] == 0
    adapter.globals().emu.supported = False
    send(adapter, command="reset")
    assert last(adapter)["error"] == "unsupported ROM"


def test_close_cancels_pending_reset(adapter):
    send(adapter, command="reset")
    send(adapter, command="close")
    adapter.globals().tick()
    assert adapter.globals().emu.loads == 0


@pytest.mark.parametrize("adapter", [True], indirect=True)
def test_lobby_input_ownership_and_disconnect_release(adapter):
    g = adapter.globals()
    for address, value, size in [
        (0x03005008, 0x02025000, 4), (0x02025004, 5, 1), (0x02025005, 5, 1),
        (0x02036E38, 1, 1), (0x02036E3A, 1, 1),
        (0x02036E48, 17, 2), (0x02036E4A, 11, 2), (0x02036E50, 2, 1),
        (0x03000EA8, 2, 1), (0x030030F4, 0x080565B5, 4),
    ]:
        for i in range(size):
            g.emu.mem[address+i] = (value >> (8*i)) & 255
    send(adapter, command="status")
    assert last(adapter)["auto_connect_supported"]
    send(adapter, command="link_connect")
    assert last(adapter)["ok"] and last(adapter)["active"]
    send(adapter, command="step", action=1, frames=1)
    assert not last(adapter)["ok"]
    g.tick()
    assert g.emu["keys"] == 1
    send(adapter, command="close")
    assert g.emu["keys"] == 0
    g.tick()
    assert g.emu["keys"] == 0
