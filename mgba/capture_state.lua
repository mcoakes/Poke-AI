-- Dedicated unattended builder. Do not load alongside adapter.lua.
local profile = dofile(CAPTURE_PROFILE_PATH)
local p = assert(profile.capture, "profile has no capture support")
local count, stable, done = 0, 0, false
local last = {}
local function u8(a) return emu:read8(a) end
local function u16(a) return emu:read16(a) end
local function u32(a) return emu:read32(a) end
local function callback(a) return u32(a) & 0xFFFFFFFE end

local function encode(value)
    if type(value) == "boolean" or type(value) == "number" then return tostring(value) end
    if type(value) == "table" then
        local fields = {}
        for key, item in pairs(value) do
            fields[#fields + 1] = encode(key) .. ":" .. encode(item)
        end
        return "{" .. table.concat(fields, ",") .. "}"
    end
    local text = tostring(value):gsub('[%z\1-\31\\"]', function(c)
        return string.format("\\u%04x", c:byte())
    end)
    return '"' .. text .. '"'
end

local function finish(report)
    done = true
    emu:setKeys(0)
    report.diagnostics = last
    local file = assert(io.open(CAPTURE_REPORT_PATH, "wb"))
    file:write(encode(report), "\n")
    file:close()
end

local function fingerprint()
    local hash = 2166136261
    for _, region in ipairs(profile.reset_regions) do
        local data = assert(emu:readRange(region[1], region[2]))
        assert(#data == region[2], "incomplete RAM read")
        for i = 1, #data do hash = ((hash ~ data:byte(i)) * 16777619) & 0xFFFFFFFF end
    end
    return string.format("%08x", hash)
end

local function observe()
    local state = {
        cb1 = callback(p.main), cb2 = callback(p.main + 4),
        locked = u8(p.field_locked), script_status = u8(p.script_status),
        fading = (u8(p.palette_fade + 7) & 0x80) ~= 0,
        party_count = u8(p.party_count),
    }
    local save = u32(p.save_block1_ptr)
    if save < 0x02000000 or save > 0x0203C000 then return state end
    state.map_group, state.map_number = u8(save + 4), u8(save + 5)
    local index = u8(p.player_avatar + 5)
    if index >= 16 then return state end
    local object = p.object_events + index * 0x24
    local flags = u8(object)
    state.x, state.y = u16(object + 0x10) - 7, u16(object + 0x12) - 7
    state.facing = u8(object + 0x18) & 15
    state.player_active = (flags & 1) ~= 0 and (u8(object + 2) & 1) ~= 0
    state.moving = (flags & 2) ~= 0 or ((flags & 0x40) ~= 0 and (flags & 0x80) == 0)
    state.prevent_step = u8(p.player_avatar + 6)
    state.tile_transition = u8(p.player_avatar + 3)
    return state
end

local function ready(s)
    return s.cb1 == p.cb1_overworld and s.cb2 == p.cb2_overworld
        and s.map_group == p.map_group and s.map_number == p.map_number
        and s.x == p.x and s.y == p.y and s.player_active and not s.moving
        and s.prevent_step == 0 and s.tile_transition == 0
        and s.locked == 0 and s.script_status == 2 and not s.fading
        and s.party_count == CAPTURE_PARTY_COUNT
end

local function save_verified()
    emu:setKeys(0)
    local before, frame = fingerprint(), emu:currentFrame()
    assert(emu:saveStateFile(CAPTURE_STATE_PATH, C.SAVESTATE.ALL), "saveStateFile failed")
    for _ = 1, 2 do
        assert(emu:loadStateFile(CAPTURE_STATE_PATH, C.SAVESTATE.ALL), "state reload failed")
        emu:setKeys(0)
        assert(emu:currentFrame() == frame and fingerprint() == before,
               "state round-trip frame/RAM mismatch")
        local restored = observe()
        assert(ready(restored) and restored.facing == 2, "restored state is not counter-ready")
    end
    finish({ok = true, frame = frame, ram_fnv1a = before, reloads_verified = 2,
            state_ready = false, profile = profile.id})
end

local function tick()
    if done then return end
    count = count + 1
    emu:setKeys(0)
    last = observe()
    if CAPTURE_PROGRESS_PATH and count % 60 == 0 then
        local file = assert(io.open(CAPTURE_PROGRESS_PATH, "wb"))
        file:write(encode({frames = count, diagnostics = last}), "\n")
        file:close()
    end
    if count > CAPTURE_MAX_FRAMES then error("timed out waiting for a ready counter") end
    if ready(last) then
        if last.facing ~= 2 then
            stable = 0
            if count % 20 == 0 then emu:setKeys(1 << C.GBA_KEY.UP) end
        elseif u16(p.main + 0x28) == 0 and u16(p.main + 0x2A) == 0 then
            stable = stable + 1
            if stable >= 12 then save_verified() end
        else stable = 0 end
        return
    end
    stable = 0
    -- Only navigate recognized title/menu callbacks. Never mash A in the overworld.
    if last.cb2 == p.cb2_title then
        if count % 30 == 0 then emu:setKeys(1 << C.GBA_KEY.START) end
    elseif last.cb2 == p.cb2_menu and not last.fading then
        for i = 0, 15 do
            local task = p.tasks + i * 40
            if u8(task + 4) ~= 0 and callback(task) == p.task_menu_input then
                assert(u16(p.save_status) == 1, "battery save did not load cleanly")
                assert(u16(task + 8) >= 1 and u16(task + 10) == 0,
                       "Continue is not the selected menu option")
                if count % 30 == 0 then emu:setKeys(1 << C.GBA_KEY.A) end
            end
        end
    end
end

local raw = emu:getGameCode() or ""
local checksum = emu:checksum(C.CHECKSUM.CRC32)
local hex = checksum:gsub(".", function(c) return string.format("%02x", c:byte()) end)
assert(raw:match("BPRE$") and hex == profile.crc32, "unsupported capture ROM")
callbacks:add("frame", function()
    local ok, err = pcall(tick)
    if not ok and not done then finish({ok = false, error = tostring(err)}) end
end)
