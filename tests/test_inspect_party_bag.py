"""Check the decoder's permutations/checksums and encrypted bag quantities."""
import itertools
import struct
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa.lua54")


def test_all_party_orders_and_bag_encryption():
    lua = lupa.LuaRuntime()
    lua.execute('''
        emu={mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a)+256*self:read8(a+1) end
        function emu:read32(a) return self:read16(a)+65536*self:read16(a+2) end
        function emu:readRange(a,n)
            local bytes={}
            for i=0,n-1 do bytes[#bytes+1]=string.char(self:read8(a+i)) end
            return table.concat(bytes)
        end
        function emu:currentFrame() return 100 end
    ''')
    mem = lua.globals().emu.mem

    def write(address, data):
        for i, byte in enumerate(data):
            mem[address + i] = byte

    write(0x03005008, struct.pack("<I", 0x02025000))
    write(0x0300500C, struct.pack("<I", 0x0202A000))
    write(0x0202AF20, struct.pack("<I", 0x12345678))
    mem[0x02024029] = 1
    write(0x02025310, struct.pack("<HH", 13, 5 ^ 0x5678))
    write(0x08254784 + 9 * 28 + 6, bytes([11, 11]))
    write(0x08250C04 + 57 * 12 + 2, bytes([11, 100, 15]))
    growth = struct.pack("<HHIBBH", 9, 13, 10000, 1, 100, 0)
    attacks = struct.pack("<4H4B", 57, 0, 0, 0, 12, 0, 0, 0)
    blocks = [growth, attacks, bytes(12), bytes(12)]
    decoder = lua.execute((Path(__file__).parents[1] /
                           "mgba/profiles/firered_us_v1_0_inspect.lua").read_text())
    for pid, order in enumerate(itertools.permutations(range(4))):
        plain = b"".join(blocks[i] for i in order)
        checksum = sum(struct.unpack("<24H", plain)) & 0xFFFF
        raw = bytearray(100)
        struct.pack_into("<II", raw, 0, pid, 12345)
        struct.pack_into("<H", raw, 28, checksum)
        for i, word in enumerate(struct.unpack("<12I", plain)):
            struct.pack_into("<I", raw, 32 + i * 4, word ^ pid ^ 12345)
        struct.pack_into("<IBBHH", raw, 80, 16, 50, 0, 80, 100)
        write(0x02024284, raw)
        report = decoder()
        mon = report.party[1]
        assert mon.valid and mon.checksum_valid and mon.species_id == 9
        assert mon.current_hp == 80 and mon.held_item_id == 13
        assert mon.moves[1].max_pp == 18 and mon.moves[1].current_pp == 12
        assert report.bag["items"][1].quantity == 5
        assert report.remaining == 1 and report.fainted == 0
        assert not report.party[2].occupied
    write(0x0202402C, raw)
    mem[0x0202402A] = 1
    mem[0x03003529] = 2
    mem[0x02023BCC] = 2
    mem[0x02023BD7] = 1
    for index in range(2):
        base = 0x02023BE4 + index * 0x58
        write(base, struct.pack("<H", 9))
        write(base + 0x28, struct.pack("<H", 80 if index == 0 else 0))
        write(base + 0x2C, struct.pack("<H", 100))
    profile = str(Path(__file__).parents[1] / "mgba/profiles/firered_us_v1_0.lua")
    report = decoder(profile)
    assert report.battle.opponent_counts.valid
    assert report.battle.opponent_counts.remaining == 0
    assert report.battle.opponent_counts.fainted == 1
    assert report.battle.player_counts.remaining == 1
    # NPC party contents remain authoritative when the cached count is zero.
    mem[0x0202402A] = 0
    write(0x0202402C + 100, raw)
    report = decoder(profile)
    assert report.battle.opponent_counts.valid
    assert report.battle.opponent_counts.remaining == 1
    assert report.battle.opponent_counts.fainted == 1
    mem[0x0202402C + 100 + 32] ^= 1
    invalid = decoder(profile).battle.opponent_counts
    assert not invalid.valid
    assert "slot 2: checksum failure (expected=" in invalid.warnings[1]
    assert "Bad Egg" not in invalid.warnings[1]
    mem[0x0202402C + 100 + 32] ^= 1
    mem[0x0202402C + 100 + 19] = 1
    invalid = decoder(profile).battle.opponent_counts
    assert not invalid.valid and invalid.warnings[1] == "slot 2: Bad Egg flag set"
    mem[0x0202402C + 100 + 19] = 0
    assert decoder(profile).battle.opponent_counts.valid
    mem[0x02023E8A] = 1
    assert decoder(profile).battle.terminal
    mem[0x03003529] = 0
    report = decoder(profile)
    assert report.battle.opponent_counts is None and not report.battle.terminal
    mem[0x02024284 + 32] = (mem[0x02024284 + 32] + 1) % 256
    report = decoder()
    assert not report.party[1].valid and not report.counts_valid
