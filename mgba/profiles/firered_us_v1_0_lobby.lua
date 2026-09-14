-- Wired Single Battle lobby, FireRed v1.0 only. All navigation uses ordinary keys.
-- Source: pret/pokefirered cable_club.inc, cable_club.c, script_menu.c,
-- start_menu.c, and symbols commit 3b401eaf14f193946899eb7c056d5ac02a470126.
return function(profile)
    local p = assert(profile.capture)
    local function u8(a) return emu:read8(a) end
    local function u16(a) return emu:read16(a) end
    local function ptr(a) return emu:read32(a) & 0xFFFFFFFE end
    local link_tasks = {
        [0x080809F8]="starting_cable", [0x08080A4C]="waiting_for_partner",
        [0x08080AD0]="printing_confirmation", [0x08080B20]="confirm_players",
        [0x08080BC8]="confirm_connection", [0x08080C6C]="confirming",
        [0x08080CDC]="waiting_for_leader", [0x08080DC0]="checking_connection",
        [0x08080E6C]="exchanging_cards", [0x08080F78]="stopping_link",
        [0x08080FB4]="link_failed", [0x08080FF0]="link_error",
    }
    local function observe()
        local s = {frame=emu:currentFrame(),
            cb1=ptr(p.main), cb2=ptr(p.main+4),
            locked=u8(p.field_locked) ~= 0, script_status=u8(p.script_status),
            script_pc=emu:read32(0x03000EB0+8),
            script_native=ptr(0x03000EB0+4),
            fading=(u8(p.palette_fade+7) & 0x80) ~= 0,
            cursor=u8(0x0203ADE4+2), menu_delay=u8(0x02039988),
            save_callback=ptr(0x03000FA4), save_printing=u8(0x03000FA9) ~= 0,
            field_message=u8(0x0203709C),
        }
        local save = emu:read32(p.save_block1_ptr)
        if save >= 0x02000000 and save <= 0x0203C000 then
            s.map_group, s.map_number = u8(save+4), u8(save+5)
        end
        local index = u8(p.player_avatar+5)
        if index < 16 then
            local object = p.object_events+index*0x24
            s.x, s.y = u16(object+0x10)-7, u16(object+0x12)-7
            s.facing = u8(object+0x18) & 15
            s.player_active = (u8(object) & 1) ~= 0 and (u8(object+2) & 1) ~= 0
            s.moving = (u8(object) & 2) ~= 0
        end
        for i=0,15 do
            local task = p.tasks+i*40
            if u8(task+4) ~= 0 then
                local fn = ptr(task)
                if fn == 0x0809CC98 then s.menu_id = u16(task+8+7*2) end
                if fn == 0x0809CE54 then s.yes_no = true end
                if fn == 0x0806F6D4 then s.saving = true end
                if link_tasks[fn] then s.link_task = link_tasks[fn] end
            end
        end
        return s
    end
    local function counter(s)
        return s.map_group == p.map_group and s.map_number == p.map_number
            and s.x == p.x and s.y == p.y and s.player_active and not s.locked
            and s.script_status == 2 and s.cb2 == p.cb2_overworld
            and not s.fading and not s.moving
    end
    local function choice(s, target)
        if s.cursor == target then return "A" end
        return s.cursor < target and "DOWN" or "UP"
    end
    local function decide(s, link, history)
        if link.wireless then
            -- IsWirelessAdapterConnected temporarily selects RFU while probing,
            -- then restores cable mode when no wireless adapter answers. Frame
            -- callbacks can observe that intermediate state inside the special.
            local probing = s.map_group == p.map_group and s.map_number == p.map_number
                and s.locked and s.script_pc >= 0x081BBB9C and s.script_pc < 0x081BBBE7
                and s.menu_id == nil and not s.link_task and not link.received_remote_players
                and not link.battle_active
            if probing then
                history.probe_frame = history.probe_frame or s.frame
                if s.frame - history.probe_frame <= 600 then return "detecting_adapter" end
                return "error", "attendant adapter detection did not return to cable mode within 600 frames"
            end
            return "error", "wireless mode is unsupported by this wired lobby"
        end
        history.probe_frame = nil
        if link.error_occurred or link.error_bits ~= 0 then return "error", "game reported a cable error" end
        if link.battle_active then return "error", "unexpected battle during lobby connection" end
        if s.fading then return "waiting_for_fade" end
        if s.map_group == 0 and s.map_number == 0 then
            if s.cb2 == p.cb2_overworld and link.connection_established
                and link.received_remote_players and link.player_count == 2 then
                return "complete"
            end
            return "entering_colosseum"
        end
        if s.map_group ~= p.map_group or s.map_number ~= p.map_number then
            return "error", "expected the generated Viridian counter checkpoint"
        end
        if s.menu_id ~= nil then
            if s.menu_id ~= 39 and s.menu_id ~= 17 then
                return "error", "unexpected multichoice menu " .. s.menu_id
            end
            if s.menu_delay ~= 0 then return "waiting_for_menu" end
            if s.menu_id == 39 then return "select_colosseum", choice(s, 1) end
            return "select_single_battle", choice(s, 0)
        end
        if s.saving then
            if not s.save_printing and (s.save_callback == 0x0806F7F8 or s.save_callback == 0x0806F8DC) then
                return "confirm_save", choice(s, 0)
            end
            if s.save_callback == 0x0806FA04 or s.save_callback == 0x0806FA2C then
                return "error", "in-game save failed"
            end
            if s.save_callback == 0x0806F9E0 then return "saved", "A" end
            return "saving"
        end
        if s.link_task then
            if s.link_task == "link_failed" or s.link_task == "link_error" or s.link_task == "stopping_link" then
                return "error", "Cable Club rejected the connection: " .. s.link_task
            end
            -- ConfirmWhenReady waits for the printer to finish, but its message
            -- contains a scroll pause after "A Button: Confirm". Despite the
            -- ShowFieldAutoScrollMessage name, that pause can need JOY_NEW(A).
            -- Advance the leader's text while both players are present; the
            -- frame controller releases A and reobserves before each next press.
            if s.link_task == "printing_confirmation" and s.field_message ~= 0
                and link.player_count == 2 then return s.link_task, "A" end
            if (s.link_task == "confirm_players" or s.link_task == "confirm_connection")
                and link.player_count == 2 then return s.link_task, "A" end
            return s.link_task
        end
        -- Abort paths include requirements failures; don't dismiss their useful text.
        if s.script_pc >= 0x081BB63C and s.script_pc < 0x081BB8A7 then
            return "error", "Cable Club refused entry; check the visible message and Parcel completion"
        end
        if s.script_pc >= 0x081BB5B3 and s.script_pc < 0x081BB63C then
            return "walking_into_colosseum"
        end
        if counter(s) then
            if history.talked then return "error", "attendant returned to overworld without connecting" end
            if s.facing ~= 2 then return "face_attendant", "UP" end
            return "talk_to_attendant", "A"
        end
        local lobby_script = (s.script_pc >= 0x081BB44A and s.script_pc < 0x081BB5B3)
            or (s.script_pc >= 0x081BBB9C and s.script_pc < 0x081BBBE7)
        if lobby_script then
            history.talked = true
            if s.field_message ~= 0 or s.script_native == 0x0806B898 then
                return "attendant_dialogue", "A"
            end
            return "attendant_dialogue"
        end
        return "waiting_for_dialogue"
    end
    return {observe=observe, counter_ready=counter, decide=decide}
end
