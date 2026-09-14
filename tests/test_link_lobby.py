"""Synthetic RAM snapshots: validate menu guards and frame-local key release."""
from pathlib import Path
import pytest

lupa = pytest.importorskip("lupa.lua54")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lobby():
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute('''
        emu={mem={}, keys=0, frame=0}
        function emu:read8(a) return self.mem[a] or 0 end
        function emu:read16(a) return self:read8(a) | (self:read8(a+1) << 8) end
        function emu:read32(a) return self:read16(a) | (self:read16(a+2) << 16) end
        function emu:setKeys(k) self.keys=k end
        function emu:currentFrame() return self.frame end
        function put(a, value, bytes)
            for i=0,bytes-1 do emu.mem[a+i]=(value >> (8*i)) & 255 end
        end
        function task(fn, id)
            put(0x03005090, fn | 1, 4); put(0x03005094, 1, 1)
            put(0x03005090+22, id or 0, 2)
        end
        C={GBA_KEY={A=0,B=1,UP=6,DOWN=7,LEFT=5,RIGHT=4}}
        put(0x03005008, 0x02025000, 4)
        put(0x02025004, 5, 1); put(0x02025005, 5, 1)
        put(0x02036E38, 1, 1); put(0x02036E38+2, 1, 1)
        put(0x02036E38+0x10, 17, 2); put(0x02036E38+0x12, 11, 2)
        put(0x02036E38+0x18, 2, 1)
        put(0x03000EA8, 2, 1)
        put(0x030030F4, 0x080565B5, 4)
    ''')
    lua.globals().profile_path = str(ROOT / "mgba/profiles/firered_us_v1_0.lua")
    lua.execute('''
        profile=dofile(profile_path)
        lobby=dofile(profile_path:gsub("%.lua$", "_lobby.lua"))(profile)
        read_link=dofile(profile_path:gsub("%.lua$", "_link.lua"))
        history={}
        function decision() return lobby.decide(lobby.observe(), read_link(), history) end
    ''')
    return lua


def test_real_menu_id_and_cursor_control_colosseum_then_single(lobby):
    g = lobby.globals()
    assert g.decision() == ("talk_to_attendant", "A")
    g.task(0x0809CC98, 39)
    assert g.decision() == ("select_colosseum", "DOWN")
    g.put(0x0203ADE6, 1, 1)
    assert g.decision() == ("select_colosseum", "A")
    g.task(0x0809CC98, 17)
    assert g.decision() == ("select_single_battle", "UP")
    g.put(0x0203ADE6, 0, 1)
    assert g.decision() == ("select_single_battle", "A")
    g.task(0x0809CC98, 999)
    assert g.decision()[0] == "error"


def test_save_yes_and_leader_confirmation_only_when_ready(lobby):
    g = lobby.globals()
    g.task(0x0806F6D4)
    g.put(0x03000FA4, 0x0806F8DD, 4)
    g.put(0x0203ADE6, 1, 1)
    assert g.decision() == ("confirm_save", "UP")
    g.put(0x0203ADE6, 0, 1)
    assert g.decision() == ("confirm_save", "A")
    g.put(0x03000FA9, 1, 1)
    assert g.decision() == "saving"
    g.task(0x08080B20)
    assert g.decision() == "confirm_players"
    g.put(0x03003F20, 0x48, 1)
    assert g.decision() == ("confirm_players", "A")
    g.task(0x08080CDC)
    assert g.decision() == "waiting_for_leader"


def test_leader_scroll_pause_requires_a_with_two_players(lobby):
    g = lobby.globals()
    # ConfirmWhenReady waits for the text printer to finish. Its message has a
    # scroll control after "A Button: Confirm", requiring a new A press.
    g.task(0x08080AD0)
    g.put(0x0203709C, 2, 1)  # FIELD_MESSAGE_BOX_AUTO_SCROLL is still printing
    g.put(0x03003F20, 0x44, 1)  # only one player
    assert g.decision() == "printing_confirmation"
    g.put(0x03003F20, 0x48, 1)  # two players
    assert g.decision() == ("printing_confirmation", "A")
    g.put(0x0203709C, 0, 1)  # printer finished; await the next task state
    assert g.decision() == "printing_confirmation"
    g.task(0x08080B20)
    assert g.decision() == ("confirm_players", "A")
    g.task(0x08080BC8)
    g.put(0x0203709C, 2, 1)  # the second confirmation has the same scroll pause
    assert g.decision() == ("confirm_connection", "A")
    g.task(0x08080CDC)  # partner must keep waiting for the leader
    assert g.decision() == "waiting_for_leader"


def test_leader_scroll_press_releases_before_handshake(lobby):
    g = lobby.globals()
    factory = lobby.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    ctl.start(ctl, 1)
    g.task(0x08080AD0)
    g.put(0x0203709C, 2, 1)
    g.put(0x03003F20, 0x48, 1)
    ctl.tick(ctl)
    assert ctl.phase == "printing_confirmation" and g.emu["keys"] == 1
    ctl.tick(ctl)
    assert g.emu["keys"] == 0  # a press, with a release before another A
    g.task(0x08080C6C)  # game now exchanges data without further input
    for _ in range(24):
        ctl.tick(ctl)
        assert ctl.phase == "confirming" and ctl.active and g.emu["keys"] == 0


def test_idle_room_is_not_connection_evidence_and_errors_stop(lobby):
    g = lobby.globals()
    g.put(0x02025004, 0, 2)
    assert g.decision() == "entering_colosseum"
    g.put(0x03003F20, 0x48, 1)
    g.put(0x03003F64, 1, 1)
    assert g.decision() == "complete"
    g.put(0x03003F3C, 1, 1)
    assert g.decision()[0] == "error"


def test_controller_releases_input_on_menu_change_and_completion(lobby):
    g = lobby.globals()
    factory = lobby.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    ctl.start(ctl, 1)
    ctl.tick(ctl)
    assert g.emu["keys"] == 1  # A, one frame
    g.task(0x0809CC98, 39)
    ctl.tick(ctl)
    assert g.emu["keys"] == 0  # old A cannot select Trade Center
    for _ in range(11):
        ctl.tick(ctl)
    assert g.emu["keys"] == 128  # DOWN
    g.put(0x02025004, 0, 2)
    g.put(0x03003F20, 0x48, 1)
    g.put(0x03003F64, 1, 1)
    ctl.tick(ctl)
    assert ctl.phase == "complete" and not ctl.active and g.emu["keys"] == 0


def test_wrong_checkpoint_and_bad_menu_never_receive_extra_keys(lobby):
    g = lobby.globals()
    factory = lobby.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    g.put(0x02036E38+0x10, 8, 2)
    with pytest.raises(lupa.LuaError, match="counter"):
        ctl.start(ctl, 1)
    g.put(0x02036E38+0x10, 17, 2)
    ctl.start(ctl, 1)
    g.task(0x0809CC98, 99)
    ctl.tick(ctl)
    assert ctl.phase == "error" and not ctl.active and g.emu["keys"] == 0


def test_attendant_adapter_probe_waits_without_keys_then_resumes(lobby):
    g = lobby.globals()
    factory = lobby.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    ctl.start(ctl, 1)
    ctl.tick(ctl)
    g.put(0x03000F9C, 1, 1)
    g.put(0x03000EB8, 0x081BBBC0, 4)
    g.put(0x03003F3C, 1, 1)
    for frame in range(1, 61):
        g.emu.frame = frame
        ctl.tick(ctl)
        assert ctl.phase == "detecting_adapter" and ctl.active
        assert g.emu["keys"] == 0
    g.put(0x03003F3C, 0, 1)
    g.task(0x0809CC98, 39)
    ctl.tick(ctl)
    assert ctl.phase == "select_colosseum" and ctl.active


def test_wireless_probe_is_bounded_and_real_wireless_menu_rejected(lobby):
    g = lobby.globals()
    g.put(0x03000F9C, 1, 1)
    g.put(0x03000EB8, 0x081BBBC0, 4)
    g.put(0x03003F3C, 1, 1)
    assert g.decision() == "detecting_adapter"
    g.emu.frame = 601
    assert g.decision()[0] == "error"
    g.emu.frame = 1
    g.put(0x03000EB8, 0x081BBBF0, 4)
    assert g.decision() == ("error", "wireless mode is unsupported by this wired lobby")


def test_staged_probe_parks_at_cable_menu_until_owner_continues(lobby):
    g = lobby.globals()
    factory = lobby.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    ctl.start(ctl, 7, True)
    g.task(0x0809CC98, 39)
    for _ in range(50):
        ctl.tick(ctl)
        assert g.emu["keys"] == 0 and ctl.active
    assert ctl.probe_complete and ctl.phase == "waiting_for_partner_probe"
    with pytest.raises(lupa.LuaError, match="does not own"):
        ctl["continue"](ctl, 8)
    ctl["continue"](ctl, 7)
    ctl.tick(ctl)
    assert ctl.phase == "select_colosseum" and g.emu["keys"] == 128


def linked_room(lua, player_id=0):
    g = lua.globals()
    g.put(0x02022718, 1, 1)
    g.put(0x02025004, 0, 2)
    g.put(0x03003F20, 0x48 | player_id, 1)
    g.put(0x03003F64, 1, 1)
    g.put(0x0300502C, player_id, 1)
    g.put(0x03005030, 2, 1)
    g.put(0x03000E80+player_id, 0x80, 1)
    link_object = 0x02031DEC+player_id*4
    g.put(link_object, 1, 1)
    g.put(link_object+1, player_id, 1)
    g.put(link_object+2, 3, 1)  # differs from the ordinary player-avatar index
    object_address = 0x02036E38+3*0x24
    g.put(object_address, 1, 1)
    g.put(object_address+0x10, 6+player_id+7, 2)
    g.put(object_address+0x12, 8+7, 2)
    factory = lua.execute((ROOT / "mgba/link_connect.lua").read_text())
    ctl = factory(g.profile, g.profile_path)
    ctl.start_battle(ctl, 7)
    return ctl, object_address, link_object


@pytest.mark.parametrize("player_id,target_x,direction", [(0, 3, 32), (1, 10, 16)])
def test_linked_players_walk_distinct_routes_from_real_link_objects(lobby, player_id, target_x, direction):
    g = lobby.globals()
    ctl, obj, link_obj = linked_room(lobby, player_id)
    actions, moving_until = [], -1
    for frame in range(200):
        g.emu.frame = frame
        g.put(link_obj+3, 1 if frame < moving_until else 2, 1)
        ctl.tick(ctl)
        key = g.emu["keys"]
        if key:
            actions.append(key)
            assert frame >= moving_until
            x, y = g.emu.read16(g.emu, obj+0x10), g.emu.read16(g.emu, obj+0x12)
            if key == 64:
                y -= 1
            elif key == 32:
                x -= 1
            elif key == 16:
                x += 1
            else:
                pytest.fail(f"unexpected input {key}")
            g.put(obj+0x10, x, 2)
            g.put(obj+0x12, y, 2)
            moving_until = frame+16
    assert actions == [64, direction, direction, direction, 64, 64]
    assert ctl.last.x == target_x and ctl.last.y == 5
    assert ctl.phase == "waiting_on_battle_pad" and g.emu["keys"] == 0


def test_battle_pad_wait_and_intentional_cable_restart(lobby):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    g.task(0x0808177C)
    g.put(0x0203709C, 1, 1)
    ctl.tick(ctl)
    assert ctl.phase == "battle_pad_text" and g.emu["keys"] == 1
    g.put(0x03005098, 2, 2)  # waiting for partner at pad
    ctl.tick(ctl)
    assert ctl.phase == "waiting_on_battle_pad" and g.emu["keys"] == 0
    g.task(0x08081318)
    g.put(0x03003F20, 0, 1)
    g.put(0x03003F64, 0, 1)  # intentional close before battle initialization
    ctl.tick(ctl)
    assert ctl.active and ctl.phase == "starting_link_battle" and g.emu["keys"] == 0
    g.put(0x03005094, 0, 1)  # task disappears as CB2_InitBattle begins
    ctl.tick(ctl)
    assert ctl.active and ctl.phase == "starting_link_battle" and g.emu["keys"] == 0


@pytest.mark.parametrize("battler_id", [0, 1])
def test_battle_intro_text_stops_at_local_command_menu(lobby, battler_id):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby, battler_id)
    g.put(0x03003529, 2, 1)
    g.put(0x02022B4C, 0xA, 4)
    g.put(0x030030F4, 0x08011101, 4)
    g.put(0x02023BCC, 2, 1)
    g.put(0x02023BD6+battler_id, 0, 1)
    g.put(0x02023BD6+(1-battler_id), 1, 1)
    g.put(0x02023BC8, 1 << battler_id, 4)
    controller_address = 0x03004FE0+battler_id*4
    g.put(controller_address, 0x08030611, 4)
    g.put(0x02020034+0x1B, 1, 1)
    g.put(0x02020034+0x1C, 3, 1)
    ctl.tick(ctl)
    assert ctl.phase == "battle_intro_text" and g.emu["keys"] == 1
    g.put(controller_address, 0x0802E439, 4)
    ctl.tick(ctl)
    assert ctl.phase == "battle_ready" and not ctl.active and g.emu["keys"] == 0
    assert ctl.status(ctl).diagnostics.command_ready
    g.put(controller_address, 0x0802EA11, 4)  # live status must detect later manual input
    assert not ctl.status(ctl).diagnostics.command_ready


def test_blocked_pad_movement_retries_bounded_holds_then_times_out(lobby):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    ctl.tick(ctl)
    assert g.emu["keys"] == 64
    pressed, released = [], []
    for frame in range(1, 602):
        g.emu.frame = frame
        ctl.tick(ctl)
        (pressed if g.emu["keys"] else released).append(frame)
    assert 1 in pressed and 24 in pressed and 8 in released
    assert ctl.phase == "error" and "no movement" in ctl.error


def test_pad_hold_stops_when_linked_movement_is_accepted(lobby):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    ctl.tick(ctl)
    assert g.emu["keys"] == 64
    g.emu.frame = 1
    g.put(0x02031DEC+3, 1, 1)  # local linked object is now walking
    ctl.tick(ctl)
    assert g.emu["keys"] == 0


def test_battle_start_ownership_and_disconnect_release(lobby):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    with pytest.raises(lupa.LuaError, match="already running"):
        ctl.start_battle(ctl, 8)
    ctl.tick(ctl)
    assert g.emu["keys"] == 64
    ctl.stop(ctl, "client_disconnected")
    assert not ctl.active and ctl.owner is None and g.emu["keys"] == 0


def test_closed_battle_transition_retains_checksum_flag_then_resumes(lobby):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    g.task(0x08081318)
    g.put(0x03005098, 5, 2)
    g.put(0x02022718, 0, 1)
    g.put(0x03003F64, 0, 1)
    g.put(0x03003F20, 8808, 4)  # supplied log: 0x2268, checksum bit set
    ctl.tick(ctl)
    assert ctl.active and ctl.phase == "waiting_for_battle_link_reopen"
    assert ctl.last.deferred_error_bits == 8192 and ctl.link.error_bits == 8192
    assert g.emu["keys"] == 0
    g.put(0x03005094, 0, 1)  # native startup task ends, status still stale
    ctl.tick(ctl)
    assert ctl.active and ctl.phase == "waiting_for_battle_link_reopen"
    g.put(0x02022718, 1, 1)
    g.put(0x03003F20, 0x48, 4)
    g.put(0x03003529, 2, 1)
    g.put(0x02022B4C, 0xA, 4)
    ctl.tick(ctl)
    assert ctl.active and ctl.phase == "battle_intro" and g.emu["keys"] == 0


@pytest.mark.parametrize("open_link,confirmed,task_state", [(1, 0, 5), (0, 1, 5), (0, 0, 0)])
def test_raw_errors_are_only_deferred_in_closed_unconfirmed_transition(lobby, open_link, confirmed, task_state):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    g.task(0x08081318)
    g.put(0x03005098, task_state, 2)
    g.put(0x02022718, open_link, 1)
    g.put(0x03003EAC, confirmed, 1)
    g.put(0x03003F64, 0, 1)
    g.put(0x03003F20, 8808, 4)
    ctl.tick(ctl)
    assert not ctl.active and ctl.phase == "error" and g.emu["keys"] == 0


@pytest.mark.parametrize("reopen", [False, True])
def test_closed_transition_error_wait_is_bounded_and_reopen_errors_still_fail(lobby, reopen):
    g = lobby.globals()
    ctl, _, _ = linked_room(lobby)
    g.task(0x08081318)
    g.put(0x03005098, 5, 2)
    g.put(0x02022718, 0, 1)
    g.put(0x03003F64, 0, 1)
    g.put(0x03003F20, 8808, 4)
    ctl.tick(ctl)
    if reopen:
        g.put(0x02022718, 1, 1)
    else:
        g.emu.frame = 601
    ctl.tick(ctl)
    assert not ctl.active and g.emu["keys"] == 0
    assert ("cable error" if reopen else "600 frames") in ctl.error
