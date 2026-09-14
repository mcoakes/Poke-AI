from pathlib import Path
import pytest


def test_link_status_decoder_without_battle_structures():
    lua = pytest.importorskip("lupa.lua54").LuaRuntime()
    lua.execute('''
        emu={mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a)+256*self:read8(a+1) end
        function emu:read32(a) return self:read16(a)+65536*self:read16(a+2) end
        function emu:currentFrame() return 10 end
    ''')
    decode = lua.execute((Path(__file__).parents[1] / "mgba/profiles/firered_us_v1_0_link.lua").read_text())
    assert not decode().connection_established
    mem = lua.globals().emu.mem
    mem[0x03003F20] = 0x49  # connected, two players, local ID 1
    mem[0x03003F64] = 1
    r = decode()
    assert r.local_id == 1 and r.player_count == 2 and r.connection_established
    assert r.received_remote_players and len(r.players) == 2
    assert not r.link_battle
    mem[0x03003529], mem[0x02022B4C] = 2, 2
    assert decode().link_battle
    mem[0x03003F21] = 0x10
    assert decode().error_bits == 0x1000
