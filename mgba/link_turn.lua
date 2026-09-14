-- One prepared switch per linked player. Python commits both sides and waits
-- for both results; neither shared mGBA process nor either core is suspended.
return function(profile_path, log)
    log = log or function() end
    local read_battle = dofile(profile_path:gsub("%.lua$", "_battle.lua"))
    local inspect = dofile(profile_path:gsub("%.lua$", "_inspect.lua"))
    local read_link = dofile(profile_path:gsub("%.lua$", "_link.lua"))
    local events_ok, make_events = pcall(dofile, profile_path:gsub("%.lua$", "_events.lua"))
    local self = {active=false, owner=nil, phase="idle", cooldown=0, peer_simulation_supported=true,
        move_supported=true, human_supported=true, battle_events_supported=events_ok}
    local TERMINAL_RETRY_FRAMES = 120
    function self:sample_events()
        if self.combat then
            local ok, err = pcall(function() self.combat:sample() end)
            if not ok then self.combat.error=tostring(err) end
        end
    end
    local function player(battle)
        for _, mon in ipairs(battle.battlers or {}) do
            if mon.side == "player"
                and (mon.valid or (battle.simulation_authoritative == false
                    and mon.party_slot >= 1 and mon.party_slot <= 6)) then return mon end
        end
        error("local active Pokemon is unavailable")
    end
    local function current_menu(battle)
        assert(#battle.menus == 1, "switch controller requires a single local battler")
        return battle.menus[1]
    end
    local function check_battle(battle)
        assert(battle.active and (battle.type_flags & 3) == 2, "requires an active singles link battle")
        assert(battle.outcome_raw == 0, "battle ended before the switch completed")
        local link = read_link()
        assert(not link.wireless and not link.error_occurred and link.error_bits == 0,
            "game reported a link error during the turn")
    end
    local function switch_allowed(battle, mon)
        -- The follower has no simulation structs. The paired Python coordinator
        -- checks BOTH players' legality from the master's fresh inspection.
        if battle.simulation_authoritative == false then return end
        local volatile = mon.volatile_status_raw or 0
        local status3 = (mon.restrictions or {}).status3 or 0
        assert((volatile & (0xE000 | 0x4000000 | 0x1000 | 0x400000)) == 0
            and (status3 & 0x400) == 0, "active Pokemon cannot switch now")
        for _, enemy in ipairs(battle.battlers) do
            if enemy.side == "opponent" and not enemy.absent and enemy.current_hp > 0 then
                local ability = enemy.ability_id
                local flying = mon.type_ids[1] == 2 or mon.type_ids[2] == 2
                local steel = mon.type_ids[1] == 8 or mon.type_ids[2] == 8
                assert(ability ~= 23 and not (ability == 71 and not flying and mon.ability_id ~= 26)
                    and not (ability == 42 and steel), "opponent ability prevents switching")
            end
        end
    end
    local function phase(value)
        if self.phase ~= value then
            log("switch request=" .. tostring(self.request_id) .. " phase=" .. value)
        end
        self.phase = value
    end
    function self:stop(reason)
        self.active, self.owner = false, nil
        if self.kind ~= "human" then emu:setKeys(0) end
        phase(reason or "cancelled")
    end
    function self:status(events_after)
        local events = {}
        if self.combat then
            for _, event in ipairs(self.combat.events) do
                if event.seq > (events_after or 0) then events[#events+1]=event end
            end
        end
        return {active=self.active, phase=self.phase, request_id=self.request_id,
            target_slot=self.target_slot, turn_start=self.turn_start, turn_end=self.turn_end,
            frame_start=self.frame_start, frame_end=self.frame_end, error=self.error,
            simulation_authoritative=self.authoritative, turn_verified=self.turn_verified,
            diagnostics=self.last, kind=self.kind, terminal_inspection=self.terminal_inspection,
            terminal_capture=self.terminal_capture,
            terminal_rejected_inspection=self.terminal_rejected_inspection,
            combat=self.combat and {events=events, seq=self.combat.seq, error=self.combat.error} or nil}
    end
    function self:capture_terminal(battle)
        -- A frame callback can catch party data during an in-place update.
        -- Publish only a validated master snapshot; never reuse old party HP.
        local frame = emu:currentFrame()
        if not self.terminal_capture then
            self.terminal_capture = {frame_start=frame, attempts=0,
                outcome_raw=battle.outcome_raw, type_flags=battle.type_flags}
        end
        local capture = self.terminal_capture
        local context = " (" .. (capture.issue or "no valid terminal snapshot") .. ")"
        assert(battle.active and (battle.type_flags & 7) == (capture.type_flags & 7)
            and battle.outcome_raw == capture.outcome_raw and battle.outcome_raw ~= 0,
            "battle changed during terminal capture" .. context)
        assert(frame-capture.frame_start <= TERMINAL_RETRY_FRAMES,
            "terminal capture exceeded 120 emulated frames" .. context)
        capture.attempts, capture.frame_end = capture.attempts+1, frame
        local snapshot = inspect(profile_path)
        local state = snapshot.battle
        assert(state and state.active and state.outcome_raw == capture.outcome_raw
            and (state.type_flags & 7) == (capture.type_flags & 7),
            "terminal inspection changed battle identity" .. context)
        local issues = {}
        if self.authoritative then
            for _, side in ipairs({"player", "opponent"}) do
                local counts = state[side .. "_counts"] or {}
                if not counts.valid then
                    local warnings = side == "player" and snapshot.warnings or counts.warnings
                    issues[#issues+1] = side .. ": " ..
                        (warnings and #warnings > 0 and table.concat(warnings, "; ") or "counts unavailable")
                end
            end
        end
        if #issues > 0 then
            capture.issue = table.concat(issues, "; ")
            self.terminal_rejected_inspection = snapshot
            phase("capturing_terminal")
            assert(frame-capture.frame_start < TERMINAL_RETRY_FRAMES,
                "terminal party counts are invalid after 120 emulated frames: " .. capture.issue)
            return
        end
        self.terminal_rejected_inspection = nil
        self.terminal_inspection, self.frame_end = snapshot, frame
        self:stop("battle_complete")
    end
    function self:prepare(owner, request_id, slot, expected_turn, kind, replacement, skip_move_menu, human_opponent, battle_log)
        kind = kind or "switch"
        assert(not self.active, "a linked turn already owns input")
        assert(request_id and request_id >= 1 and request_id % 1 == 0, "invalid request_id")
        assert(kind == "switch" or kind == "move" or kind == "wait" or kind == "human", "unknown command kind")
        assert(slot and slot >= 1 and slot <= (kind == "move" and 4 or 6) and slot % 1 == 0,
            "invalid command slot")
        assert(expected_turn and expected_turn >= 0 and expected_turn < 255, "expected_turn must be below 255")
        local report = inspect(profile_path)
        local battle = report.battle
        check_battle(battle)
        local menu = current_menu(battle)
        if kind == "human" then
            -- The human may already be browsing Fight, Bag or the party screen.
            -- This controller observes boundaries and never writes their keys.
        elseif replacement then
            assert(kind == "switch" and menu.state == "party" and menu.submenu == "choose_pokemon",
                "replacement requires the party selection screen")
        elseif kind == "wait" then
            assert(menu.state == "busy_or_unknown" or menu.state == "command"
                or (menu.state == "party" and menu.submenu == "choose_pokemon"),
                "wait requires a busy, command, or replacement selection screen; got " .. tostring(menu.state))
        else assert(menu.state == "command", "prepare requires the command menu") end
        self.authoritative = battle.simulation_authoritative ~= false
        if battle_log and self.authoritative and not self.combat then
            assert(events_ok, "battle event profile is unavailable")
            self.combat=make_events()
            self:sample_events()
        end
        assert(not self.authoritative or battle.turn == expected_turn
            or (kind == "wait" and battle.turn > expected_turn), "turn changed before command preparation")
        local mon = player(battle)
        if replacement then
            local stored = report.party[mon.party_slot]
            assert(menu.switch_required == true or (stored and stored.valid and stored.current_hp == 0),
                "replacement requires a fainted Pokemon or a mandatory Send Out menu")
        end
        local target
        if kind == "switch" then
            if not replacement then switch_allowed(battle, mon) end
            target = report.party[slot]
            assert(target and target.valid and not target.is_egg and target.current_hp > 0,
                "target slot is empty, invalid, an Egg, or fainted")
            assert(mon.party_slot ~= slot, "target Pokemon is already active")
        elseif kind == "move" and self.authoritative and not skip_move_menu then
            local move = mon.moves[slot]
            assert(move and move.move_id > 0 and move.current_pp > 0, "move slot is empty or out of PP")
        end
        local current = report.party[mon.party_slot]
        local current_details = string.format(
            "kind=%s slot=%s menu=%s valid=%s egg=%s hp=%s active_hp=%s authoritative=%s",
            kind, tostring(mon.party_slot), tostring(menu.state),
            tostring(current and current.valid), tostring(current and current.is_egg),
            tostring(current and current.current_hp), tostring(mon.current_hp), tostring(self.authoritative))
        assert(current and current.valid and not current.is_egg,
            "local active party record is unavailable (" .. current_details .. ")")
        -- A fainted player can still be waiting for its replacement screen while
        -- the partner chooses. WAIT observes that interval; it does not attack
        -- or choose a replacement on the player's behalf.
        assert(kind == "wait" or kind == "human" or replacement or current.current_hp > 0,
            "fainted active Pokemon cannot take requested action (" .. current_details .. ")")
        self.kind, self.replacement, self.skip_move_menu = kind, replacement, skip_move_menu
        self.human_opponent = human_opponent == true
        self.request_id, self.target_slot = request_id, slot
        self.target_species = target and target.species_id or nil
        self.terminal_inspection, self.left_selection = nil, false
        self.terminal_capture, self.terminal_rejected_inspection = nil, nil
        self.prepared_menu = menu.state
        self.prepared_required_switch = menu.switch_required == true
        self.initial_slot = mon.party_slot
        self.turn_start, self.turn_end, self.turn_verified = expected_turn, nil, false
        self.frame_start, self.frame_end = emu:currentFrame(), nil
        self.error, self.last, self.cooldown = nil, nil, 0
        self.active, self.owner = true, owner
        if kind ~= "human" then emu:setKeys(0) end
        phase("prepared")
    end
    function self:commit(owner, request_id)
        assert(self.active and self.owner == owner and self.request_id == request_id,
            "this client does not own the prepared request")
        assert(self.phase == "prepared", "request was already committed")
        local battle = read_battle()
        if self.kind == "wait" and battle.active and battle.outcome_raw ~= 0 then
            self:capture_terminal(battle)
            return
        end
        check_battle(battle)
        local menu, mon = current_menu(battle), player(battle)
        -- WAIT submitted no choice. Its busy interval may finish while Python
        -- commits the peer. Observe that boundary instead of requiring it to
        -- remain busy, and do not press A on the newly opened selection menu.
        local waiting = self.kind == "wait"
        local menu_ok = menu.state == self.prepared_menu or (waiting and
            (menu.state == "command" or (menu.state == "party" and menu.submenu == "choose_pokemon")))
        local turn_ok = not self.authoritative or battle.turn == self.turn_start
            or (waiting and battle.turn > self.turn_start)
        assert(self.kind == "human" or (menu_ok and turn_ok and mon.party_slot == self.initial_slot),
            string.format("battle changed before commit (kind=%s expected_menu=%s actual_menu=%s " ..
                "submenu=%s expected_slot=%s actual_slot=%s expected_turn=%s actual_turn=%s authoritative=%s)",
                self.kind, self.prepared_menu, tostring(menu.state), tostring(menu.submenu),
                tostring(self.initial_slot), tostring(mon.party_slot), tostring(self.turn_start),
                tostring(battle.turn), tostring(self.authoritative)))
        if waiting then self.left_selection = true end
        if self.kind == "switch" and not self.replacement then switch_allowed(battle, player(battle)) end
        if self.replacement then
            assert(menu.submenu == "choose_pokemon"
                and (not self.prepared_required_switch or menu.switch_required == true),
                "required switch menu changed before commit")
        end
        self.frame_start = emu:currentFrame()
        phase(self.kind == "human" and "observing" or self.kind == "wait" and "resolving"
            or self.replacement and "await_party" or "open_party")
    end
    function self:tick()
        if not self.active then return end
        if self.kind ~= "human" then emu:setKeys(0) end
        if self.phase == "prepared" then return end
        local ok, err = pcall(function()
            -- A human can take as long as needed, even at unlimited speed.
            -- Keep the navigation guard on the AI's own menu operations.
            if self.kind ~= "human" and not (self.human_opponent and self.phase == "resolving") then
                assert(emu:currentFrame()-self.frame_start <= 36000, "switch exceeded 36000 emulated frames")
            end
            local battle = read_battle()
            if battle.outcome_raw ~= 0 or self.terminal_capture then
                self:capture_terminal(battle)
                return
            end
            check_battle(battle)
            local mon, menu = player(battle), current_menu(battle)
            self.last = {frame=emu:currentFrame(), turn=battle.turn, battler_id=mon.battler_id,
                active_slot=mon.party_slot, species_id=mon.species_id, menu=menu}
            if self.kind == "human" then
                if menu.state == "busy_or_unknown" then self.left_selection = true end
                -- Human replacement selection proceeds in the GUI without an
                -- extra policy decision. Python still yields for AI replacements.
                local choosing = menu.state == "command" or menu.state == "move" or menu.state == "bag"
                    or (menu.state == "party" and mon.current_hp > 0)
                if (self.authoritative and battle.turn > self.turn_start and choosing)
                    or (not self.authoritative and self.left_selection and menu.state == "command") then
                    self.turn_end = self.authoritative and battle.turn or nil
                    self.turn_verified = self.authoritative
                    self.frame_end = emu:currentFrame()
                    self:stop("turn_complete")
                end
                return
            end
            if self.phase == "resolving" then
                if menu.state == "party" and menu.submenu == "choose_pokemon" then
                    -- The old party screen may still be closing after Shift.
                    if self.kind ~= "switch" or (self.left_selection and mon.party_slot == self.target_slot) then
                        self:stop("replacement_ready")
                    end
                    return
                end
                if menu.state == "busy_or_unknown" then self.left_selection = true end
            end
            if self.phase == "resolving" and menu.state == "command" then
                if self.kind == "wait" and (not self.authoritative or battle.turn == self.turn_start) then
                    -- A previously busy side needs a choice now. The peer may
                    -- still be submitting its action; Python waits for that
                    -- submission before exposing this new decision to PPO.
                    self.frame_end = emu:currentFrame()
                    self:stop("choice_ready")
                    return
                end
                if not self.authoritative then
                    -- A new command menu and the changed local party index prove
                    -- local completion. Only the peer can verify the shared turn.
                    if (self.kind ~= "switch" and self.left_selection)
                        or (self.kind == "switch" and mon.party_slot == self.target_slot) then
                        if self.kind == "switch" then
                            local current = inspect(profile_path).party[self.target_slot]
                            assert(current and current.valid and current.species_id == self.target_species,
                                "completed local switch has the wrong Pokemon")
                        end
                        self.frame_end = emu:currentFrame()
                        self:stop("turn_complete")
                    end
                    return
                end
                local elapsed = battle.turn-self.turn_start
                if elapsed ~= 0 then
                    assert(elapsed > 0 and (self.kind ~= "switch" or elapsed == 1),
                        "unexpected battle turn change")
                    assert(self.kind ~= "switch" or
                        (mon.party_slot == self.target_slot and mon.species_id == self.target_species),
                        "completed turn has the wrong active Pokemon")
                    self.turn_end, self.frame_end = battle.turn, emu:currentFrame()
                    self.turn_verified = true
                    self:stop("turn_complete")
                end
                return
            end
            if self.cooldown > 0 then self.cooldown = self.cooldown-1; return end
            local key
            if self.phase == "open_party" then
                if menu.state == "busy_or_unknown" then return end
                assert(menu.state == "command", "command menu changed before opening party")
                local cursor = menu.cursor_raw
                assert(cursor and cursor >= 0 and cursor <= 3, "command cursor unavailable")
                local target = self.kind == "move" and 0 or 2
                if cursor == target then
                    key = "A"
                    phase(self.kind == "move" and (self.skip_move_menu and "resolving" or "await_move") or "await_party")
                elseif (cursor & 1) ~= 0 then key = "LEFT"
                else key = target == 0 and "UP" or "DOWN" end
            elseif self.phase == "await_move" then
                if menu.state == "move" then phase("select_move") end
                return
            elseif self.phase == "select_move" then
                if menu.state == "busy_or_unknown" then return end
                assert(menu.state == "move", "move menu changed before selection")
                local cursor, target = menu.cursor_raw, self.target_slot-1
                local count = menu.move_count
                assert(count and count >= 1 and count <= 4, "visible move count unavailable")
                assert(cursor and cursor >= 0 and cursor < count and target < count,
                    "requested move or cursor is outside the visible move menu")
                if cursor == target then key = "A"; phase("resolving")
                elseif (cursor & 1) ~= (target & 1) and (cursor ~ 1) < count then
                    key = (target & 1) == 0 and "LEFT" or "RIGHT"
                -- With three moves, bottom-left -> top-right must go UP
                -- before RIGHT; the game refuses input into empty slot four.
                else key = target < 2 and "UP" or "DOWN" end
            elseif self.phase == "await_party" then
                if menu.state == "party" and menu.submenu == "choose_pokemon" then phase("select_party") end
                return
            elseif self.phase == "select_party" then
                if menu.state == "busy_or_unknown" or menu.submenu == "other_or_transition" then return end
                assert(menu.state == "party" and menu.submenu == "choose_pokemon",
                    "party menu changed before selection")
                assert(battle.party_order_valid, "party display order unavailable")
                if menu.party_slot == self.target_slot then key = "A"; phase("await_actions")
                else key = "DOWN" end
            elseif self.phase == "await_actions" then
                if menu.state == "party" and menu.submenu == "pokemon_actions" then
                    assert(menu.party_slot == self.target_slot, "wrong Pokemon actions opened")
                    phase("select_shift")
                end
                return
            elseif self.phase == "select_shift" then
                if menu.state == "busy_or_unknown" or menu.submenu == "other_or_transition" then return end
                assert(menu.state == "party" and menu.submenu == "pokemon_actions"
                    and menu.party_slot == self.target_slot, "selected Pokemon changed before Shift")
                if menu.action_id == 10 or menu.action_id == 11 then key = "A"; phase("resolving")
                else key = "DOWN" end
            elseif self.phase == "resolving" then
                if menu.state == "busy_or_unknown" and battle.text_waiting then key = "A" end
            else error("unknown switch phase") end
            if key then
                emu:setKeys(1 << assert(C.GBA_KEY[key]))
                self.cooldown = 11
            end
        end)
        if not ok then
            self.error = tostring(err)
            self:stop("error")
            log("switch error: " .. self.error)
        end
    end
    return self
end
