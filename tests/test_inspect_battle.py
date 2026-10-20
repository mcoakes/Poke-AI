from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa.lua54")


@pytest.mark.parametrize("local_id", [0, 1])
def test_link_input_handler_survives_cleared_execution_bit(local_id):
    lua = lupa.LuaRuntime()
    lua.execute('''
        emu={mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a) | (self:read8(a+1) << 8) end
        function emu:read32(a) return self:read16(a) | (self:read16(a+2) << 16) end
        function put(a,v,n)
            for i=0,n-1 do emu.mem[a+i]=(v >> (8*i)) & 255 end
        end
    ''')
    decode = lua.execute((Path(__file__).parents[1] /
                          "mgba/profiles/firered_us_v1_0_battle.lua").read_text())
    put = lua.globals().put
    for a, v, n in [(0x03003529, 2, 1), (0x02023BCC, 2, 1),
                    (0x02022B4C, 10, 4), (0x030030F4, 0x08011101, 4),
                    (0x02023BD6+local_id, 0, 1), (0x02023BD6+1-local_id, 1, 1),
                    (0x02023BC8, 0, 4), (0x02023FF8+local_id, 2, 1)]:
        put(a, v, n)
    controller = 0x03004FE0+local_id*4
    for fn, expected in [(0x0802E439, "command"), (0x0802EA11, "move"),
                         (0x0802E675, "target"), (0x08030611, "busy_or_unknown")]:
        put(controller, fn, 4)
        result = decode()
        assert result.menus[1].battler_id == local_id
        assert result.menus[1].state == expected
    put(controller, 0x0802E439, 4)
    assert decode().menus[1].cursor_raw == 2
    put(0x02037ABF, 128, 1)
    assert decode().menus[1].state == "busy_or_unknown"
    put(0x02037ABF, 0, 1)
    put(0x030030F4, 0x08000001, 4)
    assert decode().menus[1].state == "busy_or_unknown"
    put(0x030030F4, 0x08011101, 4)
    put(0x02022B4C, 8, 4)  # independent NPC behavior remains unchanged
    assert decode().menus[1].state == "busy_or_unknown"


def test_active_records_and_menu_transitions():
    lua = lupa.LuaRuntime()
    lua.execute('''
        emu={mem={}}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a)+256*self:read8(a+1) end
        function emu:read32(a) return self:read16(a)+65536*self:read16(a+2) end
        function put(a,v,n)
            for i=0,n-1 do emu.mem[a+i]=(v >> (8*i)) & 255 end
        end
    ''')
    decode = lua.execute((Path(__file__).parents[1] /
                          "mgba/profiles/firered_us_v1_0_battle.lua").read_text())
    put = lua.globals().put
    assert not decode().active
    for raw, expected in [(1, "won"), (2, "lost"), (3, "draw"), (4, "ran"),
                          (7, "caught"), (128, "link_battle_ran"), (99, "unknown")]:
        put(0x02023E8A, raw, 1)
        result = decode()
        assert result.outcome == expected
        assert result.outcome_scope == "last_battle_or_stale" and not result.terminal
    put(0x02023E8A, 0, 1)
    for a, v, n in [(0x03003529, 2, 1), (0x02023BCC, 2, 1), (0x02023BD7, 1, 1),
                    (0x02023BC8, 1, 4), (0x030030F4, 0x08011101, 4),
                    (0x03004FE0, 0x0802E439, 4), (0x02023FF8, 2, 1),
                    (0x02023BCE, 1, 2)]:
        put(a, v, n)
    for index, species in enumerate([6, 19]):
        base = 0x02023BE4 + index * 0x58
        for offset, value, size in [(0, species, 2), (0x28, 20, 2), (0x2C, 50, 2),
                                     (0x2A, 10, 1), (0x21, 10, 1), (0x22, 2, 1)]:
            put(base + offset, value, size)
    result = decode()
    assert result.active and result.battlers[1].party_slot == 2
    for offset, value in [(2, 11), (4, 22), (6, 33), (8, 44), (10, 55)]:
        put(0x02023BE4 + offset, value, 2)
    put(0x02023BE4 + 0x14, 0xFFFFFFFF, 4)
    put(0x02023F1C, 0x20, 2)
    put(0x02023F48, 4, 1)
    native = decode()
    assert list(native.battlers[1].stats.values()) == [11, 22, 44, 55, 33]
    assert native.battlers[1].ivs_raw == 0x3FFFFFFF
    assert native.weather_raw == 0x20 and native.weather_duration == 4
    assert result.battlers[2].side == "opponent"
    assert result.battlers[1].current_hp == 20
    assert result.menus[1].state == "command" and result.menus[1].cursor_raw == 2
    put(0x03004FE0, 0x0802EA11, 4)
    put(0x03004FF8, 3, 1)
    assert decode().menus[1].state == "move" and decode().menus[1].move_count == 3
    put(0x02023BC8, 0, 4)
    assert decode().menus[1].state == "busy_or_unknown"
    put(0x030030F4, 0x0811EBA1, 4)
    assert decode().menus[1].state == "party"
    # gPartyMenu.action SEND_OUT applies to a healthy Baton Pass user too.
    put(0x0203B0A8, 1, 1)
    put(0x0203B0AB, 1, 1)
    assert decode().menus[1].switch_required
    assert decode().menus[1].party_action_raw == 1
    for action in (0, 2, 3, 4):  # ordinary selection, trapping, item, ability
        put(0x0203B0AB, action, 1)
        assert not decode().menus[1].switch_required
    put(0x0203B0AB, 1, 1)
    put(0x0203B0A8, 0, 1)
    assert not decode().menus[1].switch_required  # not an in-battle party menu
    put(0x0203B0A9, 2, 1)
    put(0x03005090, 0x08122C5D, 4)
    put(0x03005094, 1, 1)
    put(0x0203B09C, 0x02010000, 4)
    put(0x02010017, 3, 1)
    put(0x0201000F, 11, 1)
    menu = decode().menus[1]
    assert menu.cursor_raw == 2 and menu.submenu == "pokemon_actions"
    assert menu.action_id == 11
    put(0x0203B0DC, 0x10, 1)
    put(0x0203B0DD, 0x23, 1)
    put(0x0203B0DE, 0x45, 1)
    put(0x0203B0A9, 1, 1)
    result = decode()
    assert result.party_order_valid
    assert result.menus[1].cursor_raw == 1 and result.menus[1].party_slot == 1
    assert result.party_order[1] == 2
    put(0x030030F4, 0x08107EE1, 4)
    assert decode().menus[1].state == "bag"
    assert decode().menus[1].switch_required is None
    put(0x03005090, 0x08108F0D, 4)
    put(0x0203AD02, 2, 2)
    put(0x0203AD0E, 4, 2)
    put(0x0203AD08, 1, 2)
    menu = decode().menus[1]
    assert menu.submenu == "item_list" and menu.pocket_raw == 2
    assert menu.scroll_raw == 4 and menu.row_raw == 1 and menu.item_index_raw == 5
    put(0x02037ABF, 128, 1)
    assert decode().menus[1].state == "busy_or_unknown"
