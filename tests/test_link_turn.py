"""Exercise the real Lua controller against menu transitions and reordered slots."""
from pathlib import Path
import pytest

lupa = pytest.importorskip("lupa.lua54")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def turn():
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    factory = lua.execute((ROOT / "mgba/link_turn.lua").read_text())
    lua.execute('''
        C={GBA_KEY={A=0,B=1,UP=6,DOWN=7,LEFT=5,RIGHT=4}}
        emu={frame=0,keys=0}
        function emu:currentFrame() return self.frame end
        function emu:setKeys(k) self.keys=k end
        menu={state="command",cursor_raw=0,battler_id=0,move_count=4}
        player={side="player",valid=true,absent=false,party_slot=1,species_id=6,
                battler_id=0,current_hp=100,ability_id=66,type_ids={10,2},restrictions={}}
        player.moves={{move_id=53,current_pp=15},{move_id=19,current_pp=15},
                      {move_id=163,current_pp=20},{move_id=108,current_pp=20}}
        enemy={side="opponent",valid=true,absent=false,party_slot=1,species_id=9,
               current_hp=100,ability_id=67,type_ids={11,11},restrictions={}}
        battle={active=true,type_flags=10,outcome_raw=0,turn=0,menus={menu},
                battlers={player,enemy},party_order_valid=true,text_waiting=false,
                player_counts={valid=true,remaining=0,fainted=3},
                opponent_counts={valid=true,remaining=0,fainted=3,warnings={}}}
        report={battle=battle,warnings={},party={
            {valid=true,species_id=6,is_egg=false,current_hp=100},
            {valid=true,species_id=59,is_egg=false,current_hp=100},
            {valid=true,species_id=131,is_egg=false,current_hp=100}}}
        link={wireless=false,error_occurred=false,error_bits=0}
        function dofile(path)
            if path:match("_battle.lua$") then return function() return battle end end
            if path:match("_inspect.lua$") then return function() return report end end
            if path:match("_link.lua$") then return function() return link end end
            error("unexpected read " .. path)
        end
    ''')
    g = lua.globals()
    ctl = factory("profile.lua")
    g.ctl = ctl
    lua.execute('function tick() emu.frame=emu.frame+1; ctl:tick() end')
    return g, ctl


def next_key(g):
    for _ in range(40):
        g.tick()
        if g.emu["keys"]:
            return g.emu["keys"]
        if not g.ctl.active:
            raise AssertionError(g.ctl.error or g.ctl.phase)
    raise AssertionError(f"no input in phase {g.ctl.phase}")


@pytest.mark.parametrize("authoritative", [True, False])
@pytest.mark.parametrize("boundary", ["replacement", "command", "terminal"])
def test_wait_boundary_can_arrive_between_prepare_and_commit(turn, authoritative, boundary):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    g.menu.state = "busy_or_unknown"
    ctl.prepare(ctl, 7, 1, 1, 0, "wait", False, False)
    if boundary == "replacement":
        g.player.current_hp = g.report.party[1].current_hp = 0
        g.menu.state, g.menu.submenu = "party", "choose_pokemon"
        expected = "replacement_ready"
    elif boundary == "command":
        g.menu.state = "command"
        g.battle.turn = 1 if authoritative else 0
        expected = "turn_complete" if authoritative else "choice_ready"
    else:
        g.battle.outcome_raw = 1
        expected = "battle_complete"
    ctl.commit(ctl, 7, 1)
    g.tick()
    assert ctl.phase == expected and not ctl.active
    assert g.emu["keys"] == 0 and ctl.error is None


@pytest.mark.parametrize("kind", ["move", "switch"])
@pytest.mark.parametrize("changed", ["menu", "slot", "turn"])
def test_commit_still_rejects_changed_action_target_with_details(turn, kind, changed):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 2, 0, kind, False, False)
    if changed == "menu":
        g.menu.state = "move"
    elif changed == "slot":
        g.player.party_slot = 3
    else:
        g.battle.turn = 1
    with pytest.raises(lupa.LuaError, match="battle changed before commit") as error:
        ctl.commit(ctl, 7, 1)
    assert f"kind={kind}" in str(error.value)
    assert "expected_menu=command" in str(error.value)
    assert "actual_slot=" in str(error.value) and "actual_turn=" in str(error.value)
    assert g.emu["keys"] == 0


@pytest.mark.parametrize("authoritative", [True, False])
@pytest.mark.parametrize("menu", ["command", "party"])
def test_wait_selection_boundary_can_arrive_before_prepare(turn, authoritative, menu):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    g.menu.state, g.menu.submenu = menu, "choose_pokemon"
    if menu == "party":
        g.player.current_hp = g.report.party[1].current_hp = 0
    ctl.prepare(ctl, 7, 1, 1, 0, "wait", False, False)
    ctl.commit(ctl, 7, 1)
    g.tick()
    assert ctl.phase == ("replacement_ready" if menu == "party" else "choice_ready")
    assert not ctl.active and g.emu["keys"] == 0


@pytest.mark.parametrize("local_id", [0, 1])
def test_prepared_switch_reordered_party_and_one_turn_barrier(turn, local_id):
    g, ctl = turn
    g.player.battler_id = local_id
    g.menu.battler_id = local_id
    g.player.party_slot, g.player.species_id = 2, 59
    g.battle.turn = 254  # the real counter saturates at 255; it does not wrap
    ctl.prepare(ctl, 7, 12, 1, 254)
    for _ in range(20):
        g.tick()
        assert ctl.phase == "prepared" and g.emu["keys"] == 0
    ctl.commit(ctl, 7, 12)
    assert next_key(g) == 128  # command cursor: Fight -> Pokemon
    g.menu.cursor_raw = 2
    assert next_key(g) == 1
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.menu.cursor_raw, g.menu.party_slot = 0, 2  # display row 0 now contains Arcanine
    assert next_key(g) == 128
    g.menu.cursor_raw, g.menu.party_slot = 1, 1  # display row 1 contains Charizard
    assert next_key(g) == 1
    g.menu.submenu, g.menu.action_id = "pokemon_actions", 0
    assert next_key(g) == 128  # Summary -> Shift
    g.menu.action_id = 10
    assert next_key(g) == 1
    g.menu.state, g.menu.submenu = "busy_or_unknown", None
    g.battle.text_waiting = True
    assert next_key(g) == 1
    g.menu.state = "command"
    g.player.party_slot, g.player.species_id = 1, 6
    for _ in range(15):
        g.tick()
        assert ctl.active and g.emu["keys"] == 0  # same turn is not completion
    g.battle.turn = 255
    g.tick()
    assert ctl.phase == "turn_complete" and not ctl.active and ctl.owner is None
    assert ctl.turn_start == 254 and ctl.turn_end == 255 and g.emu["keys"] == 0
    assert ctl.turn_verified
    with pytest.raises(lupa.LuaError, match="below 255"):
        ctl.prepare(ctl, 7, 13, 2, 255)


def test_follower_uses_local_slot_and_menu_without_simulation_ram(turn):
    g, ctl = turn
    g.battle.simulation_authoritative = False
    g.player.valid, g.player.species_id = False, 0
    g.enemy.valid, g.enemy.species_id = False, 0
    g.battle.turn = 0
    ctl.prepare(ctl, 7, 8, 2, 7)  # shared turn is supplied by the peer coordinator
    ctl.commit(ctl, 7, 8)
    ctl.phase = "resolving"
    g.tick()
    assert ctl.active and g.emu["keys"] == 0  # old slot is not completion
    g.player.party_slot = 2
    g.menu.state = "busy_or_unknown"
    g.tick()
    assert ctl.active  # changed slot before next command menu is not completion
    g.menu.state = "command"
    g.tick()
    assert ctl.phase == "turn_complete" and not ctl.active
    assert ctl.turn_end is None and not ctl.turn_verified
    assert g.battle.turn == 0 and g.emu["keys"] == 0


def test_prepare_rejects_unavailable_targets_and_trapping(turn):
    g, ctl = turn
    for slot in (0, 1, 4, 7):
        with pytest.raises(lupa.LuaError):
            ctl.prepare(ctl, 7, 1, slot, 0)
        assert not ctl.active and g.emu["keys"] == 0
    g.enemy.ability_id = 23
    with pytest.raises(lupa.LuaError, match="ability prevents"):
        ctl.prepare(ctl, 7, 1, 2, 0)
    g.enemy.ability_id = 67
    g.report.party[2].current_hp = 0
    with pytest.raises(lupa.LuaError, match="fainted"):
        ctl.prepare(ctl, 7, 1, 2, 0)


def test_other_owner_double_commit_and_disconnect(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 2, 0)
    with pytest.raises(lupa.LuaError, match="does not own"):
        ctl.commit(ctl, 8, 1)
    ctl.commit(ctl, 7, 1)
    with pytest.raises(lupa.LuaError, match="already committed"):
        ctl.commit(ctl, 7, 1)
    assert next_key(g) == 128
    ctl.stop(ctl, "client_disconnected")
    assert not ctl.active and ctl.owner is None and g.emu["keys"] == 0


def test_wrong_pokemon_submenu_never_confirms_shift(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 2, 0)
    ctl.commit(ctl, 7, 1)
    assert next_key(g) == 128
    g.menu.cursor_raw = 2
    assert next_key(g) == 1
    g.menu.state, g.menu.submenu, g.menu.party_slot = "party", "choose_pokemon", 2
    assert next_key(g) == 1
    g.menu.submenu, g.menu.party_slot = "pokemon_actions", 1
    for _ in range(15):
        g.tick()
        assert g.emu["keys"] == 0
    assert ctl.phase == "error" and "wrong Pokemon" in ctl.error


def test_later_turn_with_wrong_active_slot_fails(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 2, 0)
    ctl.commit(ctl, 7, 1)
    ctl.phase = "resolving"
    g.battle.turn = 1
    g.tick()
    assert ctl.phase == "error" and "wrong active Pokemon" in ctl.error
    assert g.emu["keys"] == 0


@pytest.mark.parametrize("authoritative", [True, False])
def test_move_selection_waits_for_resolution_and_next_menu(turn, authoritative):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    g.menu.cursor_raw = 2
    ctl.prepare(ctl, 7, 1, 3, 0, "move", False, False)
    ctl.commit(ctl, 7, 1)
    assert next_key(g) == 64  # Pokemon -> Fight
    g.menu.cursor_raw = 0
    assert next_key(g) == 1
    g.menu.state = "move"
    assert next_key(g) == 128  # move 1 -> move 3
    g.menu.cursor_raw = 2
    assert next_key(g) == 1
    g.menu.state = "busy_or_unknown"
    for _ in range(15):
        g.tick()
        assert g.emu["keys"] == 0
    g.menu.state = "command"
    if authoritative:
        g.tick()
        assert ctl.active  # master verifies the shared counter
        g.battle.turn = 1
    g.tick()
    assert ctl.phase == "turn_complete" and not ctl.active


@pytest.mark.parametrize("authoritative", [True, False])
@pytest.mark.parametrize("count,start,target", [
    (n, start, target) for n in range(1, 5) for start in range(n) for target in range(n)])
def test_navigation_uses_only_existing_move_slots(turn, authoritative, count, start, target):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    ctl.prepare(ctl, 7, 1, target+1, 0, "move")
    ctl.commit(ctl, 7, 1)
    ctl.phase = "select_move"
    g.menu.state, g.menu.move_count, g.menu.cursor_raw = "move", count, start
    keys = []
    for _ in range(4):
        key = next_key(g)
        keys.append(key)
        current = g.menu.cursor_raw
        if key == 1:
            assert current == target and ctl.phase == "resolving"
            break
        # Match the game's input bounds, independently of routing policy.
        if key == 32:
            assert current & 1
            candidate = current ^ 1
        elif key == 16:
            assert not current & 1
            candidate = current ^ 1
        elif key == 64:
            assert current & 2
            candidate = current ^ 2
        else:
            assert key == 128 and not current & 2
            candidate = current ^ 2
        assert candidate < count, "attempted to move into an empty slot"
        g.menu.cursor_raw = candidate
    else:
        pytest.fail("did not confirm the requested move")
    if (count, start, target) == (3, 2, 1):
        assert keys == [64, 16, 1]  # Harden -> Poison Sting -> String Shot -> A


def test_target_outside_visible_move_menu_fails_promptly(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 4, 0, "move")
    ctl.commit(ctl, 7, 1)
    ctl.phase = "select_move"
    g.menu.state, g.menu.move_count = "move", 3
    g.tick()
    assert ctl.phase == "error" and "outside the visible move menu" in ctl.error
    assert not ctl.active and g.emu["keys"] == 0


def test_move_terminal_is_latched_and_keys_released(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "move", False, False)
    ctl.commit(ctl, 7, 1)
    g.battle.outcome_raw = 2
    g.emu["keys"] = 1
    g.tick()
    assert ctl.phase == "battle_complete" and not ctl.active
    assert ctl.terminal_inspection.battle.outcome_raw == 2
    assert g.emu["keys"] == 0 and ctl.owner is None


@pytest.mark.parametrize("boundary", ["tick", "commit"])
def test_terminal_checksum_is_retried_before_publishing_fresh_snapshot(turn, boundary):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "wait" if boundary == "commit" else "move")
    if boundary == "tick":
        ctl.commit(ctl, 7, 1)
    g.battle.outcome_raw = 2
    g.battle.opponent_counts.valid = False
    g.battle.opponent_counts.warnings[1] = "slot 6: checksum failure"
    g.emu["keys"] = 0 if boundary == "commit" else 1
    if boundary == "commit":
        ctl.commit(ctl, 7, 1)
    else:
        g.tick()
    assert ctl.active and ctl.phase == "capturing_terminal"
    assert ctl.terminal_inspection is None and g.emu["keys"] == 0
    assert ctl.terminal_rejected_inspection is not None
    for _ in range(3):
        g.tick()
        assert ctl.active and ctl.terminal_inspection is None
    # A new valid read supplies final HP; old party values are not substituted.
    g.report.party[2].current_hp = 17
    g.battle.opponent_counts.valid = True
    g.battle.opponent_counts.warnings[1] = None
    g.tick()
    assert ctl.phase == "battle_complete" and not ctl.active
    assert ctl.terminal_inspection.party[2].current_hp == 17
    assert ctl.terminal_capture.attempts == 5
    assert ctl.terminal_rejected_inspection is None


@pytest.mark.parametrize("side", ["player", "opponent"])
def test_persistent_invalid_terminal_data_fails_with_slot_diagnostics(turn, side):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "move")
    ctl.commit(ctl, 7, 1)
    g.battle.outcome_raw = 2
    g.battle[f"{side}_counts"].valid = False
    warnings = g.report.warnings if side == "player" else g.battle.opponent_counts.warnings
    warnings[1] = "slot 6: Bad Egg flag set"
    for _ in range(121):
        g.tick()
        assert g.emu["keys"] == 0
    assert not ctl.active and ctl.phase == "error"
    assert "120 emulated frames" in ctl.error
    assert f"{side}: slot 6: Bad Egg flag set" in ctl.error
    assert ctl.terminal_inspection is None
    assert ctl.status(ctl).terminal_rejected_inspection is not None


@pytest.mark.parametrize("change", ["active", "outcome", "authority", "deadline"])
def test_terminal_retry_rejects_teardown_changed_outcome_or_late_read(turn, change):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "move")
    ctl.commit(ctl, 7, 1)
    g.battle.outcome_raw = 2
    g.battle.opponent_counts.valid = False
    g.tick()
    if change == "active":
        g.battle.active = False
    elif change == "outcome":
        g.battle.outcome_raw = 0
    elif change == "authority":
        g.battle.type_flags = 14
    else:
        g.emu.frame += 121
        g.battle.opponent_counts.valid = True
    g.tick()
    assert ctl.phase == "error" and not ctl.active
    assert ctl.terminal_inspection is None and g.emu["keys"] == 0


def test_follower_does_not_require_its_unused_simulation_counts(turn):
    g, ctl = turn
    g.battle.simulation_authoritative = False
    ctl.prepare(ctl, 7, 1, 1, 0, "move")
    ctl.commit(ctl, 7, 1)
    g.battle.outcome_raw = 2
    g.battle.opponent_counts.valid = False
    g.tick()
    assert ctl.phase == "battle_complete" and not ctl.active


def test_faint_yields_replacement_without_consuming_another_move(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "move", False, False)
    ctl.commit(ctl, 7, 1)
    ctl.phase = "resolving"
    g.player.current_hp = 0
    g.report.party[1].current_hp = 0
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.tick()
    assert ctl.phase == "replacement_ready" and g.emu["keys"] == 0
    ctl.prepare(ctl, 7, 2, 2, 0, "switch", True, False)
    ctl.commit(ctl, 7, 2)
    assert ctl.phase == "await_party"


@pytest.mark.parametrize("authoritative", [True, False])
def test_baton_pass_yields_and_selects_recipient_without_opening_command_menu(turn, authoritative):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    g.player.volatile_status_raw = 0x2000
    ctl.prepare(ctl, 7, 1, 1, 0, "move")
    ctl.commit(ctl, 7, 1)
    ctl.phase = "resolving"
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.menu.switch_required, g.menu.party_slot = True, 2
    g.tick()
    assert ctl.phase == "replacement_ready" and g.player.current_hp == 100
    ctl.prepare(ctl, 7, 2, 2, 0, "switch", True)
    ctl.commit(ctl, 7, 2)
    assert ctl.phase == "await_party"
    assert next_key(g) == 1  # A on recipient; no command-menu navigation
    assert ctl.phase == "await_actions"
    g.menu.submenu, g.menu.action_id = "pokemon_actions", 11  # Send Out
    assert next_key(g) == 1
    assert ctl.phase == "resolving"
    g.menu.state, g.menu.submenu = "busy_or_unknown", None
    g.tick()
    assert ctl.active and g.emu["keys"] == 0
    g.player.party_slot, g.player.species_id = 2, 59
    g.menu.state = "command"
    if authoritative:
        g.battle.turn = 1
    g.tick()
    assert ctl.phase == "turn_complete" and not ctl.active
    assert g.emu["keys"] == 0


def test_living_regular_party_selection_cannot_claim_required_switch(turn):
    g, ctl = turn
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.player.volatile_status_raw = 0x2000
    with pytest.raises(lupa.LuaError, match="mandatory Send Out menu"):
        ctl.prepare(ctl, 7, 1, 2, 0, "switch", True)
    g.menu.switch_required = True
    ctl.prepare(ctl, 7, 1, 2, 0, "switch", True)
    g.menu.switch_required = False
    with pytest.raises(lupa.LuaError, match="required switch menu changed"):
        ctl.commit(ctl, 7, 1)
    assert g.emu["keys"] == 0


@pytest.mark.parametrize("authoritative", [True, False])
def test_fainted_player_can_wait_until_own_replacement_menu_opens(turn, authoritative):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    g.player.valid = authoritative  # follower simulation structs are unavailable
    g.player.current_hp = g.report.party[1].current_hp = 0
    g.menu.state = "busy_or_unknown"
    ctl.prepare(ctl, 7, 1, 1, 0, "wait", False, False)
    g.tick()
    assert ctl.phase == "prepared" and g.emu["keys"] == 0
    ctl.commit(ctl, 7, 1)
    g.tick()
    assert ctl.active and ctl.phase == "resolving" and g.emu["keys"] == 0
    g.battle.text_waiting = True
    assert next_key(g) == 1  # only advance text that requests confirmation
    g.battle.text_waiting = False
    g.tick()
    assert g.emu["keys"] == 0
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.tick()
    assert ctl.phase == "replacement_ready" and not ctl.active
    assert ctl.owner is None and g.emu["keys"] == 0
    ctl.prepare(ctl, 7, 2, 2, 0, "switch", True, False)
    ctl.commit(ctl, 7, 2)
    assert ctl.phase == "await_party"  # replacement is a separate legal choice


@pytest.mark.parametrize("kind", ["move", "switch"])
def test_fainted_player_still_cannot_attack_or_switch_voluntarily(turn, kind):
    g, ctl = turn
    g.player.current_hp = g.report.party[1].current_hp = 0
    with pytest.raises(lupa.LuaError, match="fainted active Pokemon") as error:
        ctl.prepare(ctl, 7, 1, 2, 0, kind, False, False)
    assert f"kind={kind} slot=1 menu=command" in str(error.value)
    assert "hp=0" in str(error.value)
    assert not ctl.active and ctl.owner is None


@pytest.mark.parametrize("record_problem", ["missing", "invalid", "egg"])
def test_wait_keeps_active_party_validation_and_reports_failed_record(turn, record_problem):
    g, ctl = turn
    g.menu.state = "busy_or_unknown"
    g.player.current_hp = g.report.party[1].current_hp = 0
    if record_problem == "missing":
        g.report.party[1] = None
    elif record_problem == "invalid":
        g.report.party[1].valid = False
    else:
        g.report.party[1].is_egg = True
    with pytest.raises(lupa.LuaError, match="local active party record is unavailable") as error:
        ctl.prepare(ctl, 7, 1, 1, 0, "wait", False, False)
    assert "kind=wait slot=1 menu=busy_or_unknown" in str(error.value)
    assert ("egg=true" if record_problem == "egg" else
            "valid=false" if record_problem == "invalid" else "valid=nil") in str(error.value)
    assert not ctl.active and ctl.owner is None and g.emu["keys"] == 0


@pytest.mark.parametrize("authoritative", [True, False])
def test_human_observer_never_writes_keys_even_during_replacement(turn, authoritative):
    g, ctl = turn
    g.battle.simulation_authoritative = authoritative
    # Catch every attempted key write, including prepare, commit and cleanup.
    g.emu["setKeys"] = lambda *args: (_ for _ in ()).throw(AssertionError("human keys were overwritten"))
    g.emu["keys"] = 64
    g.menu.state = "move"  # the user already opened Fight
    ctl.prepare(ctl, 7, 1, 1, 0, "human", False, False)
    g.menu.state = "party"  # GUI input continues between prepare and commit
    ctl.commit(ctl, 7, 1)
    g.emu.frame = 50000
    g.tick()
    assert ctl.active and ctl.phase == "observing" and g.emu["keys"] == 64
    g.menu.state = "busy_or_unknown"
    g.tick()
    g.menu.state, g.menu.submenu = "party", "choose_pokemon"
    g.player.current_hp = 0
    g.tick()
    assert ctl.active  # the human chooses their own replacement, at their own pace
    g.player.current_hp, g.player.party_slot = 100, 2
    g.menu.state = "command"
    g.battle.turn = 1
    g.tick()
    assert ctl.phase == "turn_complete" and not ctl.active
    ctl.stop(ctl, "client_disconnected")
    assert g.emu["keys"] == 64


def test_human_master_stays_observing_animations_and_latches_terminal(turn):
    g, ctl = turn
    g.battle.simulation_authoritative = True
    ctl.prepare(ctl, 7, 1, 1, 0, "human", False, False)
    ctl.commit(ctl, 7, 1)
    g.emu["keys"] = 1
    g.battle.turn, g.menu.state = 1, "busy_or_unknown"
    g.tick()
    assert ctl.active  # turn-counter advance during animations is not a new choice
    g.battle.outcome_raw = 1
    g.tick()
    assert ctl.phase == "battle_complete" and ctl.terminal_inspection.battle.outcome_raw == 1
    assert g.emu["keys"] == 1


def test_ai_waits_for_human_beyond_frame_budget_but_navigation_still_guarded(turn):
    g, ctl = turn
    ctl.prepare(ctl, 7, 1, 1, 0, "move", False, False, True)
    ctl.commit(ctl, 7, 1)
    ctl.phase = "resolving"
    g.menu.state = "busy_or_unknown"
    g.emu.frame = 100000
    g.tick()
    assert ctl.active and ctl.error is None
    ctl.stop(ctl)
    g.menu.state = "command"
    ctl.prepare(ctl, 7, 2, 1, 0, "move", False, False, True)
    ctl.commit(ctl, 7, 2)
    g.emu.frame += 40000
    g.tick()
    assert ctl.phase == "error" and "36000" in ctl.error
