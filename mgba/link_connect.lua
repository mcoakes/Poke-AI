-- Frame-driven key controller. No recursive runFrame, memory writes or timed macros.
return function(profile, profile_path, log)
    log = log or function() end
    local lobby = dofile(profile_path:gsub("%.lua$", "_lobby.lua"))(profile)
    local read_link = dofile(profile_path:gsub("%.lua$", "_link.lua"))
    local battle_start = dofile(profile_path:gsub("%.lua$", "_battle_start.lua"))(profile, profile_path, lobby)
    local self = {active=false, phase="idle", history={}, owner=nil, ticks=0, cooldown=0,
        staged_supported=true, battle_start_supported=true, probe_complete=false, mode="lobby"}
    function self:status()
        if self.mode == "battle" and self.phase == "battle_ready" then
            self.last, self.link = battle_start.observe(), read_link()
        end
        return {active=self.active, phase=self.phase, error=self.error,
            frames_elapsed=self.ticks, diagnostics=self.last, link=self.link,
            probe_complete=self.probe_complete, mode=self.mode}
    end
    function self:stop(reason)
        self.active = false
        self.owner = nil
        self.phase = reason or "cancelled"
        self.held_direction = nil
        emu:setKeys(0)
        log("lobby stop: " .. self.phase)
    end
    function self:start(owner, hold_at_service)
        assert(not self.active, "link connection automation is already running")
        local s, link = lobby.observe(), read_link()
        assert(lobby.counter_ready(s), "auto-connect requires the generated Viridian link-counter .ss1 (10,4), idle in the overworld")
        assert(not link.connection_established and not link.wireless and not link.battle_active,
            "auto-connect requires a pre-connection checkpoint")
        self.owner, self.active, self.phase = owner, true, "starting"
        self.mode = "lobby"
        self.ticks, self.cooldown, self.history = 0, 0, {}
        self.error, self.last, self.link = nil, s, link
        self.probe_complete, self.hold_at_service = false, hold_at_service == true
        emu:setKeys(0)
        log("lobby start; staged=" .. tostring(self.hold_at_service))
    end
    function self:start_battle(owner)
        assert(not self.active, "link connection automation is already running")
        local s, link = battle_start.observe(), read_link()
        assert(s.map_group == 0 and s.map_number == 0 and not link.battle_active,
            "battle startup requires the connected two-player Colosseum")
        assert(link.connection_established and link.received_remote_players
            and link.player_count == 2 and not link.wireless, "both cable players must be connected")
        self.owner, self.active, self.phase, self.mode = owner, true, "starting_battle", "battle"
        self.ticks, self.cooldown, self.history = 0, 0, {}
        self.error, self.last, self.link = nil, s, link
        emu:setKeys(0)
        log("battle startup: approaching assigned pads")
    end
    function self:continue(owner)
        assert(self.owner == owner and self.active, "this client does not own active auto-connect")
        assert(self.probe_complete, "adapter probe is not complete")
        self.hold_at_service = false
        log("partner probe complete; continuing dialogue")
    end
    function self:tick()
        if not self.active then return end
        emu:setKeys(0)
        local ok, err = pcall(function()
            self.ticks = self.ticks+1
            assert(self.ticks <= 72000, "lobby exceeded 72000 emulated frames")
            local controller = self.mode == "battle" and battle_start or lobby
            self.last, self.link = controller.observe(), read_link()
            local phase, key = controller.decide(self.last, self.link, self.history)
            if phase == "select_colosseum" then
                self.probe_complete = true
                if self.hold_at_service then phase, key = "waiting_for_partner_probe", nil end
            end
            if phase ~= self.phase then
                log(string.format("phase=%s frame=%d script=0x%08x wireless=%s task=%s menu=%s x=%s y=%s seat=%s",
                    phase, self.last.frame, self.last.script_pc, tostring(self.link.wireless),
                    tostring(self.last.link_task), tostring(self.last.menu_id),
                    tostring(self.last.x), tostring(self.last.y), tostring(self.last.seat_state)))
            end
            self.phase = phase
            if phase == "error" then error(key) end
            if phase == "complete" or phase == "battle_ready" then self:stop(phase); return end
            -- Keep a direction visible across linked input exchanges, but release
            -- immediately once movement starts, coordinates change, or control locks.
            local move = self.history.move
            if self.mode == "battle" and self.held_direction and move
                and self.last.frame < self.hold_until and phase == "walking_to_battle_pad"
                and not self.last.locked and self.last.movement_mode ~= 1
                and self.last.x == move.x and self.last.y == move.y then
                emu:setKeys(1 << assert(C.GBA_KEY[self.held_direction]))
                return
            end
            self.held_direction = nil
            -- Reobserve before EVERY input; released frames generate distinct presses.
            if self.cooldown > 0 then self.cooldown = self.cooldown-1; return end
            if key then
                emu:setKeys(1 << assert(C.GBA_KEY[key]))
                self.cooldown = 11
                if controller.after_input then controller.after_input(self.last, self.history, key) end
                if self.mode == "battle" and key ~= "A" then
                    self.held_direction, self.hold_until = key, self.last.frame+8
                end
            end
        end)
        if not ok then
            self:stop("error")
            self.error = tostring(err)
            log("lobby error: " .. self.error)
        end
    end
    return self
end
