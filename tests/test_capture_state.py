"""Execute the capture Lua using synthetic RAM and a fake emulator."""
import json
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa.lua54")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def capture(tmp_path):
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute('''
        C={GBA_KEY={START=3,A=0,UP=6},CHECKSUM={CRC32=0},SAVESTATE={ALL=31}}
        callbacks={}
        function callbacks:add(name,fn) self[name]=fn end
        emu={memory={},frame=0,held=0,saves=0,loads=0,fail_save=false,bad_reload=false}
        function emu:read8(a) return self.memory[a] or 0 end
        function emu:read16(a) return self:read8(a) + 256*self:read8(a+1) end
        function emu:read32(a) return self:read16(a) + 65536*self:read16(a+2) end
        function put(a,v,n)
            for i=0,n-1 do emu.memory[a+i]=(v >> (8*i)) & 255 end
        end
        function emu:setKeys(k) self.held=k end
        function emu:getGameCode() return "AGB-BPRE" end
        function emu:checksum() return string.char(0xDD,0x88,0x76,0x1C) end
        function emu:currentFrame() return self.frame end
        function emu:readRange(a,n) return string.rep("x",n) end
        function emu:saveStateFile(path,flags)
            self.saves=self.saves+1
            self.saved_frame=self.frame
            if self.fail_save then return false end
            local f=assert(io.open(path,"wb")); f:write("synthetic state"); f:close()
            return true
        end
        function emu:loadStateFile(path,flags)
            self.loads=self.loads+1
            self.frame=self.saved_frame+(self.bad_reload and 1 or 0)
            return true
        end
        function tick(n)
            for i=1,n do emu.frame=emu.frame+1; callbacks.frame() end
        end
    ''')
    g = lua.globals()
    g.CAPTURE_PROFILE_PATH = str(ROOT / "mgba/profiles/firered_us_v1_0.lua")
    g.CAPTURE_STATE_PATH = str(tmp_path / "counter.ss1")
    g.CAPTURE_REPORT_PATH = str(tmp_path / "report.json")
    g.CAPTURE_PARTY_COUNT = 6
    g.CAPTURE_MAX_FRAMES = 600
    lua.execute((ROOT / "mgba/capture_state.lua").read_text())
    return lua, tmp_path


def counter(lua, facing=2):
    values = [(0x030030F0, 0x08056535, 4), (0x030030F4, 0x080565B5, 4),
              (0x03005008, 0x02025000, 4), (0x02025004, 5, 1), (0x02025005, 5, 1),
              (0x02036E38, 1, 1), (0x02036E3A, 1, 1), (0x02036E48, 17, 2),
              (0x02036E4A, 11, 2), (0x02036E50, facing, 1), (0x03000EA8, 2, 1),
              (0x02024029, 6, 1)]
    for address, value, size in values:
        lua.globals().put(address, value, size)


def test_title_continue_turn_and_capture(capture):
    lua, path = capture
    g = lua.globals()
    g.put(0x030030F4, 0x08078B9D, 4)
    g.tick(30)
    assert g.emu.held == 8  # START only on title
    for address, value, size in [(0x030030F4, 0x0800C2D5, 4),
                                 (0x03005090, 0x0800CA55, 4), (0x03005094, 1, 1),
                                 (0x03005098, 1, 2), (0x030053A0, 1, 2)]:
        g.put(address, value, size)
    g.tick(30)
    assert g.emu.held == 1  # A on selected Continue
    counter(lua, facing=1)
    g.tick(20)
    assert g.emu.held == 64  # UP at solid counter
    g.put(0x02036E50, 2, 1)
    g.tick(12)
    result = json.loads((path / "report.json").read_text())
    assert result["ok"] and result["reloads_verified"] == 2
    assert g.emu.saves == 1 and g.emu.loads == 2 and g.emu.held == 0
    assert result["diagnostics"]["x"] == 10


@pytest.mark.parametrize("address,value", [(0x03000F9C, 1), (0x02037ABF, 128),
                                         (0x03000EA8, 0), (0x0203707B, 1),
                                         (0x02024029, 5), (0x02036E48, 18)])
def test_never_capture_blocked_or_wrong_position(capture, address, value):
    lua, path = capture
    counter(lua)
    lua.globals().put(address, value, 1)
    lua.globals().tick(601)
    result = json.loads((path / "report.json").read_text())
    assert not result["ok"] and "timed out" in result["error"]
    assert lua.globals().emu.saves == 0 and lua.globals().emu.held == 0


@pytest.mark.parametrize("failure", ["fail_save", "bad_reload"])
def test_save_or_round_trip_failure_is_not_success(capture, failure):
    lua, path = capture
    counter(lua)
    lua.globals().emu[failure] = True
    lua.globals().tick(12)
    assert json.loads((path / "report.json").read_text())["ok"] is False


def test_missing_battery_save_does_not_start_new_game(capture):
    lua, path = capture
    g = lua.globals()
    g.put(0x030030F4, 0x0800C2D5, 4)
    g.put(0x03005090, 0x0800CA55, 4)
    g.put(0x03005094, 1, 1)
    g.tick(1)
    assert "battery save" in json.loads((path / "report.json").read_text())["error"]
    assert g.emu.held == 0
