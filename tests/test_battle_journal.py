import json
from pathlib import Path

import pytest

from firered_rl.battle_journal import BattleJournal
from firered_rl.link_inspection import pair_inspections

ROOT = Path(__file__).resolve().parents[1]


def test_real_lua_sampler_reads_text_hp_and_deduplicates_frames():
    lua = pytest.importorskip("lupa.lua54").LuaRuntime()
    lua.execute('''
        emu={mem={},frame=0}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a) | (self:read8(a+1) << 8) end
        function emu:read32(a) return self:read16(a) | (self:read16(a+2) << 16) end
        function emu:readRange(a,n)
            local out={}; for i=0,n-1 do out[#out+1]=string.char(self:read8(a+i)) end
            return table.concat(out)
        end
        function emu:currentFrame() return self.frame end
        function put(a,v,n) for i=0,n-1 do emu.mem[a+i]=(v >> (8*i)) & 255 end end
    ''')
    g = lua.globals()
    make = lua.execute((ROOT / 'mgba/profiles/firered_us_v1_0_events.lua').read_text())
    recorder = make()
    def tick():
        g.emu.frame += 1
        recorder.sample(recorder)
    def message(text):
        # Include text-format control bytes; they must not appear in the log.
        data = [0xFC, 1, 2]
        for char in text:
            data.append(0 if char == ' ' else 0xAB if char == '!' else
                        0xAD if char == '.' else 0xB4 if char == "'" else
                        0xBB + ord(char)-65 if char.isupper() else 0xD5 + ord(char)-97)
        data.append(255)
        for offset, value in enumerate(data):
            g.put(0x0202298C+offset, value, 1)
        g.put(0x03004FE0, 0x08030611, 4)
        g.put(0x02020034, 0x0202298C+4, 4)
        tick()
    for address, value, size in [(0x03003529, 2, 1), (0x02022B4C, 14, 4),
                                  (0x02023BCC, 2, 1)]:
        g.put(address, value, size)
    for i, species in enumerate((6, 9)):
        base = 0x02023BE4+i*0x58
        for offset, value in ((0, species), (0x28, 150), (0x2C, 150)):
            g.put(base+offset, value, 2)
    tick()
    message("Foe BLASTOISE used HYDRO PUMP!")
    assert recorder.events[1].text == "Foe BLASTOISE used HYDRO PUMP!"
    tick()
    assert recorder.seq == 1
    g.put(0x02023BE4+0x28, 70, 2)
    tick()
    assert recorder.events[2].before == 150 and recorder.events[2].after == 70
    message("A critical hit!")
    message("It's super effective!")
    assert recorder.events[3].text == "A critical hit!"
    assert recorder.events[4].text == "It's super effective!"
    g.put(0x03004FE0, 0, 4)
    tick()
    message("It's super effective!")  # same text printed again, not another frame
    assert recorder.seq == 5
    g.put(0x02023BCE, 1, 2)  # a different slot, even with the same species
    g.put(0x02024284+100+86, 150, 2)
    g.put(0x02024284+100+88, 150, 2)
    tick()  # index changes before the incoming HP is copied
    g.put(0x02023BE4+0x28, 150, 2)
    tick()
    assert recorder.seq == 5  # a switch is not healing/damage
    g.put(0x02023BE4+0x28, 120, 2)
    tick()
    assert recorder.seq == 6 and recorder.events[6].before == 150 and recorder.events[6].after == 120
    g.put(0x02022B4C, 10, 4)  # follower cannot emit a second copy
    g.put(0x02023BE4+0x28, 0, 2)
    tick()
    assert recorder.seq == 6


def test_compact_journal_is_live_deduplicated_and_resets_sequence(tmp_path):
    reports = pair_inspections(json.loads((ROOT / "tests/fixtures/link_master_follower.json").read_text()))
    path = tmp_path / "battle.log"
    log = BattleJournal(path)
    log.start("episode-1", reports)
    assert "P1 Charizard" in path.read_text() and "P2 Blastoise" in path.read_text()
    events = [{"seq": 1, "kind": "text", "text": "Foe BLASTOISE used HYDRO PUMP!"},
              {"seq": 2, "kind": "hp", "battler_id": 0, "species_id": 6,
               "before": 153, "after": 50, "max_hp": 153},
              {"seq": 3, "kind": "text", "text": "A critical hit!"},
              {"seq": 4, "kind": "text", "text": "It's super effective!"}]
    states = [{"combat": {"events": events}}, {}]
    log.collect(states)
    log.collect(states)
    text = path.read_text()
    assert text.count("HYDRO PUMP") == 1
    assert "P1 Charizard took 103 damage" in text
    assert "A critical hit!" in text and "super effective" in text
    assert "Legal actions" not in text and "components=" not in text
    log.start("episode-2", reports)
    log.collect(states)
    assert path.read_text().count("HYDRO PUMP") == 2
