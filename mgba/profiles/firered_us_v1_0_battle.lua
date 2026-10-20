-- Active-battler inspection; evaluator/debug data, not policy observations.
return function()
    local function task_active(fn)
        for i=0,15 do
            local task = 0x03005090+i*40
            if emu:read8(task+4) ~= 0 and (emu:read32(task) & 0xFFFFFFFE) == fn then return true end
        end
        return false
    end
    local active = (emu:read8(0x030030F0 + 0x439) & 2) ~= 0
    local result = {active=active, battlers={}, menus={}, evaluator_only=true}
    -- The message printer can await A during battle text. Menu controllers
    -- still take precedence; callers must never advance a selection menu here.
    local printer = 0x02020034
    local text_state = emu:read8(printer+0x1C)
    result.text_waiting = active and emu:read8(printer+0x1B) ~= 0
        and (text_state == 1 or text_state == 2 or text_state == 3)
    result.outcome_raw = emu:read8(0x02023E8A)
    local outcomes = {[0]="ongoing", [1]="won", [2]="lost", [3]="draw",
        [4]="ran", [5]="player_teleported", [6]="opponent_fled", [7]="caught",
        [8]="no_safari_balls", [9]="forfeited", [10]="opponent_teleported"}
    result.outcome = outcomes[result.outcome_raw & 0x7F] or "unknown"
    result.link_battle_ran = (result.outcome_raw & 0x80) ~= 0
    if result.link_battle_ran then result.outcome = "link_battle_ran" end
    result.outcome_scope = active and "current_battle" or "last_battle_or_stale"
    result.terminal = active and result.outcome_raw ~= 0 and result.outcome ~= "unknown"
    if not active then return result end
    local count = emu:read8(0x02023BCC)
    assert(count >= 1 and count <= 4, "battle structures are not initialized yet")
    result.type_flags = emu:read32(0x02022B4C)
    -- Only BATTLE_TYPE_IS_MASTER runs the simulation in a native link battle.
    -- The follower owns controller/party data, not gBattleMons or this turn count.
    result.simulation_authoritative = (result.type_flags & 2) == 0 or (result.type_flags & 4) ~= 0
    result.simulation_role = result.simulation_authoritative and "master_or_npc" or "link_follower"
    -- pret/pokefirered: gBattleWeather; gWishFutureKnock.weatherDuration at +0x28.
    result.weather_raw = emu:read16(0x02023F1C)
    result.weather_duration = emu:read8(0x02023F20+0x28)
    result.turn = emu:read8(0x03004F90 + 0x13)
    result.controller_exec_flags = emu:read32(0x02023BC8)
    result.callback2 = emu:read32(0x030030F4) & 0xFFFFFFFE
    result.screen = "battle_or_transition"
    if result.callback2 == 0x08107EE0 then result.screen = "bag"
    elseif result.callback2 == 0x0811EBA0 then result.screen = "party" end
    if result.screen == "party" then
        result.party_order = {}
        local seen = {}
        result.party_order_valid = true
        for i=0,5 do
            local packed = emu:read8(0x0203B0DC + i//2)
            local slot = (i%2 == 0) and (packed >> 4) or (packed & 15)
            if slot > 5 or seen[slot] then result.party_order_valid = false end
            seen[slot] = true
            result.party_order[i+1] = slot+1
        end
    end
    local absent = emu:read8(0x02023D70)
    for id=0,count-1 do
        local base = 0x02023BE4 + id*0x58
        local position = emu:read8(0x02023BD6+id)
        local mon = {
            battler_id=id, position=position,
            side=(position & 1) == 0 and "player" or "opponent",
            absent=(absent & (1 << id)) ~= 0,
            party_slot=emu:read16(0x02023BCE+id*2)+1,
            species_id=emu:read16(base), level=emu:read8(base+0x2A),
            current_hp=emu:read16(base+0x28), max_hp=emu:read16(base+0x2C),
            type_ids={emu:read8(base+0x21),emu:read8(base+0x22)},
            held_item_id=emu:read16(base+0x2E), ability_id=emu:read8(base+0x20),
            status_raw=emu:read32(base+0x4C), volatile_status_raw=emu:read32(base+0x50),
            stat_stages={}, moves={},
            stats={emu:read16(base+2),emu:read16(base+4),emu:read16(base+8),
                   emu:read16(base+10),emu:read16(base+6)},
            ivs_raw=emu:read32(base+0x14) & 0x3FFFFFFF,
        }
        local disable = 0x02023E0C + id*0x1C
        local bs = emu:read32(0x02023FE8)
        mon.restrictions = {
            disabled_move=emu:read16(disable+4), encored_move=emu:read16(disable+6),
            encore_timer=emu:read8(disable+0xE) & 15, encore_slot=emu:read8(disable+0xC),
            taunt_timer=emu:read8(disable+0x13) & 15,
            last_move=emu:read16(0x02023D98+id*2), locked_move=emu:read16(0x02023DB8+id*2),
            status3=emu:read32(0x02023DFC+id*4),
        }
        if bs >= 0x02000000 and bs < 0x0203F000 then
            mon.restrictions.choice_move = emu:read16(bs+0xC8+id*2)
        end
        if mon.held_item_id < 375 then
            mon.restrictions.hold_effect = emu:read8(0x083DB028+mon.held_item_id*44+18)
        end
        mon.valid = mon.species_id >= 1 and mon.species_id <= 411 and mon.max_hp > 0
            and mon.current_hp <= mon.max_hp and mon.party_slot <= 6
        mon.data_source = result.simulation_authoritative and "local_simulation" or "unavailable_on_link_follower"
        if not result.simulation_authoritative then mon.valid = false end
        for stat=0,7 do mon.stat_stages[#mon.stat_stages+1] = emu:read8(base+0x18+stat) end
        for slot=0,3 do
            local move = {slot=slot+1, move_id=emu:read16(base+0xC+slot*2),
                          current_pp=emu:read8(base+0x24+slot)}
            if move.move_id >= 1 and move.move_id <= 354 then
                local entry = 0x08250C04 + move.move_id*12
                local pp = emu:read8(entry+4)
                local bonus = (emu:read8(base+0x3B) >> (slot*2)) & 3
                move.max_pp = pp + (pp*bonus)//5
                move.base_type_id = emu:read8(entry+2)
                move.power = emu:read8(entry+1)
            end
            mon.moves[#mon.moves+1] = move
        end
        result.battlers[#result.battlers+1] = mon
        if mon.side == "player" and (not mon.absent or (result.type_flags & 3) == 2) then
            local fn = emu:read32(0x03004FE0+id*4) & 0xFFFFFFFE
            local menu = {battler_id=id, controller=fn, state="busy_or_unknown"}
            local executing = (result.controller_exec_flags & (1 << id)) ~= 0
            -- In link battles the buffer execution bit can clear after dispatch
            -- while HandleInputChooseAction/Move still owns the local controller.
            -- BattleMainCB1 calls that controller every frame without this bit.
            -- Keep the callback, fade and exact input-handler checks below.
            local input_controller_live = executing or (result.type_flags & 2) ~= 0
            local fading = (emu:read8(0x02037ABF) & 0x80) ~= 0
            if not fading and result.screen == "party" then
                menu.state = "party"
                -- gPartyMenu: menuType at +8 (low nibble), action at +11.
                -- PARTY_ACTION_SEND_OUT is requested by both faint replacement
                -- and Baton Pass. It is independent of the active Pokemon's HP.
                menu.party_action_raw = emu:read8(0x0203B0A0+11)
                menu.switch_required = (emu:read8(0x0203B0A0+8) & 15) == 1
                    and menu.party_action_raw == 1
                menu.cursor_raw = emu:read8(0x0203B0A0+9)
                if result.party_order_valid and menu.cursor_raw < 6 then
                    menu.party_slot = result.party_order[menu.cursor_raw+1]
                end
                menu.submenu = "other_or_transition"
                if task_active(0x0811FB28) then menu.submenu = "choose_pokemon"
                elseif task_active(0x08122C5C) then
                    menu.submenu = "pokemon_actions"
                    local internal = emu:read32(0x0203B09C)
                    local cursor = emu:read8(0x0203ADE6)
                    menu.action_cursor_raw = cursor
                    if internal >= 0x02000000 and internal < 0x0203FC00 then
                        local n = emu:read8(internal+23)
                        if n <= 8 and cursor < n then
                            menu.action_id = emu:read8(internal+15+cursor)
                        end
                    end
                end
            elseif not fading and result.screen == "bag" then
                menu.state = "bag"
                menu.pocket_raw = emu:read16(0x0203ACFC+6)
                menu.submenu = task_active(0x08108F0C) and "item_list" or "other_or_transition"
                if menu.pocket_raw <= 2 then
                    local pocket = menu.pocket_raw
                    menu.scroll_raw = emu:read16(0x0203ACFC+14+pocket*2)
                    menu.row_raw = emu:read16(0x0203ACFC+8+pocket*2)
                    menu.item_index_raw = menu.scroll_raw+menu.row_raw
                end
            elseif not fading and input_controller_live and result.callback2 == 0x08011100 then
                if fn == 0x0802E438 then
                    menu.state = "command"
                    menu.cursor_raw = emu:read8(0x02023FF8+id)
                elseif fn == 0x0802EA10 then
                    menu.state = "move"
                    menu.cursor_raw = emu:read8(0x02023FFC+id)
                    menu.move_count = emu:read8(0x03004FF8) -- gNumberOfMovesToChoose, local controller
                elseif fn == 0x0802E674 then menu.state = "target" end
            end
            result.menus[#result.menus+1] = menu
        end
    end
    return result
end
