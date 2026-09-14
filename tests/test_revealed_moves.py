from pathlib import Path
import pytest


def test_printed_moves_only_and_reset():
    lua = pytest.importorskip("lupa.lua54").LuaRuntime()
    lua.execute('''
        emu={mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a)+256*self:read8(a+1) end
        function emu:read32(a) return self:read16(a)+65536*self:read16(a+2) end
    ''')
    observer = lua.execute((Path(__file__).parents[1] / "mgba/profiles/firered_us_v1_0_revealed.lua").read_text())
    mem = lua.globals().emu.mem
    mem[0x03003529], mem[0x02023BCC], mem[0x02023BD7] = 2, 2, 1
    mem[0x02023BE4+0x58] = 13
    observer.sample(observer)
    assert len(observer.records) == 1 and len(observer.records[1].moves) == 0
    mem[0x02023D4A] = 99  # choosing a move is not a reveal
    observer.sample(observer)
    assert len(observer.records[1].moves) == 0
    base = 0x02022BC4+0x200
    mem[base], mem[base+2], mem[base+4] = 16, 4, 33
    mem[0x02023BC8] = 2
    observer.sample(observer)
    observer.sample(observer)
    assert len(observer.records[1].moves) == 1 and observer.records[1].moves[1].move_id == 33
    observer.reset(observer)
    mem[0x02023BC8] = 0
    observer.sample(observer)
    assert len(observer.records[1].moves) == 0
