-- FireRed v1.0 wired 2P Colosseum -> first command menu, using ordinary input.
-- Provenance: overworld.c linked objects, cable_club.c seat/start tasks,
-- BattleColosseum_2P/map.json + map.bin, and the pinned v1.0 symbols.
return function(profile, profile_path, lobby)
    local read_battle = dofile(profile_path:gsub("%.lua$", "_battle.lua"))
    local function observe()
        local s = lobby.observe()
        s.link_open = emu:read8(0x02022718) ~= 0 -- sLinkOpen; CheckErrorStatus guard
        s.field_player_id = emu:read8(0x0300502C)
        s.field_player_count = emu:read8(0x03005030)
        s.link_object_valid = false
        if s.field_player_id < 2 then
            local link_object = 0x02031DEC+s.field_player_id*4
            local object_id = emu:read8(link_object+2)
            s.movement_mode = emu:read8(link_object+3)
            s.player_link_state = emu:read8(0x03000E80+s.field_player_id)
            if emu:read8(link_object) ~= 0 and object_id < 16
                and emu:read8(link_object+1) == s.field_player_id then
                local object = profile.capture.object_events+object_id*0x24
                s.link_object_valid = (emu:read8(object) & 1) ~= 0
                s.x, s.y = emu:read16(object+0x10)-7, emu:read16(object+0x12)-7
            end
        end
        for i=0,15 do
            local task = profile.capture.tasks+i*40
            if emu:read8(task+4) ~= 0 then
                local fn = emu:read32(task) & 0xFFFFFFFE
                if fn == 0x0808177C then s.seat_state = emu:read16(task+8) end
                if fn == 0x08081318 then s.battle_start_state = emu:read16(task+8) end
            end
        end
        s.command_ready, s.text_waiting = false, false
        if (emu:read8(0x03003529) & 2) ~= 0 then
            local ok, battle = pcall(read_battle)
            if ok and battle.active then
                s.battle_turn, s.battle_outcome = battle.turn, battle.outcome_raw
                for _, menu in ipairs(battle.menus) do
                    s.battler_id, s.battle_menu, s.battle_controller = menu.battler_id, menu.state, menu.controller
                    if menu.state == "command" then
                        s.command_ready = true
                        s.command_cursor = menu.cursor_raw
                    end
                    -- Only advance a recognized battle text printer, never an
                    -- unknown menu or the first action-selection screen.
                    local printer = 0x02020034
                    local state = emu:read8(printer+0x1C)
                    if menu.controller == 0x08030610 and emu:read8(printer+0x1B) ~= 0
                        and (state == 1 or state == 2 or state == 3) then
                        s.text_waiting = true
                    end
                end
            end
        end
        return s
    end
    local function decide(s, link, history)
        if link.wireless then return "error", "battle startup requires wired mode" end
        if link.error_occurred then return "error", "game reported a cable error" end
        if link.error_bits ~= 0 then
            -- CloseLink leaves gLinkStatus intact. CheckErrorStatus itself only
            -- acts on its raw flags while sLinkOpen is true. During the known
            -- lobby close/reopen, retain the flags in diagnostics and wait.
            local closing = s.battle_start_state == 4 or s.battle_start_state == 5
                or history.transition
            if closing and not s.link_open and not link.received_remote_players
                and not s.command_ready then
                history.closed_error_frame = history.closed_error_frame or s.frame
                s.deferred_error_bits = link.error_bits
                if s.frame-history.closed_error_frame > 600 then
                    return "error", "cable remained closed with stale error flags for 600 frames"
                end
                history.transition = true
                return "waiting_for_battle_link_reopen"
            end
            return "error", "game reported a cable error"
        end
        history.closed_error_frame = nil
        if link.battle_active then
            if not link.link_battle then return "error", "expected a link battle" end
            if s.command_ready then return "battle_ready" end
            if s.fading then return "battle_intro" end
            if s.text_waiting then return "battle_intro_text", "A" end
            return "battle_intro"
        end
        -- The game intentionally closes the lobby connection before CB2_InitBattle
        -- reopens it. Wait through this transition without movement or buttons.
        if s.battle_start_state ~= nil then
            history.transition = true
            return "starting_link_battle"
        end
        if history.transition then return "starting_link_battle" end
        if s.map_group ~= 0 or s.map_number ~= 0 then
            return "error", "battle startup requires the two-player Colosseum"
        end
        if s.fading or s.cb2 ~= profile.capture.cb2_overworld then return "waiting_for_colosseum" end
        if s.seat_state ~= nil then
            if s.seat_state == 3 then return "error", "battle pad readiness was cancelled" end
            if s.seat_state <= 1 and s.field_message ~= 0 then return "battle_pad_text", "A" end
            return "waiting_on_battle_pad"
        end
        if not link.connection_established or not link.received_remote_players or link.player_count ~= 2 then
            return "error", "Colosseum connection lost before entering the battle pads"
        end
        if not s.link_object_valid or s.field_player_count ~= 2 then return "waiting_for_link_player" end
        if s.field_player_id ~= link.local_id then return "error", "linked player identity mismatch" end
        if s.locked or s.player_link_state ~= 0x80 then return "waiting_for_player_control" end
        if s.movement_mode == 1 then return "walking_to_battle_pad" end
        if s.movement_mode ~= 0 and s.movement_mode ~= 2 then
            return "error", "unknown linked player movement mode"
        end
        if history.move then
            if s.x == history.move.x and s.y == history.move.y then
                if s.frame-history.move.first_frame > 600 then
                    return "error", "no movement after battle-pad input; check linked object diagnostics"
                end
                -- Linked overworld input is exchanged asynchronously. A one-frame
                -- pulse can miss an input exchange; retry only while still idle
                -- at the same coordinates. Never queue keys while moving/locked.
                if s.frame-history.move.frame >= 24 then
                    return "walking_to_battle_pad", history.move.key
                end
                return "walking_to_battle_pad"
            end
            history.move = nil
        end
        local target_x = s.field_player_id == 0 and 3 or 10
        if s.x == target_x and s.y == 5 then return "waiting_on_battle_pad" end
        if s.x < 3 or s.x > 10 or s.y < 5 or s.y > 8 then
            return "error", "linked player is outside the supported Colosseum approach"
        end
        local key
        if s.y == 8 then key = "UP"
        elseif s.x ~= target_x then
            if s.y ~= 7 then return "error", "unexpected position before battle-pad approach" end
            key = s.x < target_x and "RIGHT" or "LEFT"
        elseif s.y > 5 then key = "UP" end
        if key then
            return "walking_to_battle_pad", key
        end
        return "waiting_on_battle_pad"
    end
    local function after_input(s, history, key)
        if key ~= "A" then
            local first = history.move and history.move.first_frame or s.frame
            history.move = {x=s.x, y=s.y, frame=s.frame, first_frame=first, key=key}
        end
    end
    return {observe=observe, decide=decide, after_input=after_input}
end
