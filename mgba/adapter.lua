-- FireRed RL mGBA adapter, party/bag inspection first pass (2D).
-- The emulator runs normally between requests; this is not a paused RL scheduler.

local ADAPTER_VERSION = "0.18.1"
local DEFAULT_PORT = 55355
local MAX_STEP_FRAMES = 600

-- Qt's scripting console is not forwarded to the process's stderr log.
local function diagnostic(message)
    if not FIRERED_RL_LOG_PATH then return end
    local file, err = io.open(FIRERED_RL_LOG_PATH, "a")
    if not file then console:error("Cannot write adapter log: " .. tostring(err)); return end
    file:write(tostring(message), "\n")
    file:flush()
    file:close()
end
diagnostic("adapter " .. ADAPTER_VERSION .. " loading")

local profile_id = FIRERED_RL_PROFILE or "firered_us_v1_0"
local profile_path = FIRERED_RL_PROFILE_PATH or
    ("mgba/profiles/" .. profile_id .. ".lua")
local loaded, profile = pcall(dofile, profile_path)
if not loaded then
    error("FireRed RL adapter could not load profile " .. profile_path .. ": " .. profile)
end
if type(profile) ~= "table" or profile.id ~= profile_id then
    error("FireRed RL adapter loaded an invalid profile: " .. profile_path)
end

local observer_ok, observer = pcall(dofile, profile_path:gsub("%.lua$", "_revealed.lua"))
if not observer_ok then observer = nil end
local function attach_revealed(report)
    if observer and report.battle then
        report.battle.revealed_opponents = observer.records
        report.battle.revealed_active_index = observer.active_index
    end
    return report
end
local clients = {}
local next_client_id = 1
local server = nil
local send_response
local pending_step = nil
local pending_reset = nil
local state_path = FIRERED_RL_STATE_PATH
local link_connect = nil
local link_connect_error = nil
local link_turn, link_turn_error = nil, nil
if FIRERED_RL_NATIVE_LINK_SESSION then
    local ok, result = pcall(function()
        local directory = profile_path:match("^(.*[/\\])") or "mgba/profiles/"
        return dofile(directory .. "../link_connect.lua")(profile, profile_path, diagnostic)
    end)
    if ok then link_connect = result else link_connect_error = tostring(result) end
    ok, result = pcall(function()
        local directory = profile_path:match("^(.*[/\\])") or "mgba/profiles/"
        return dofile(directory .. "../link_turn.lua")(profile_path, diagnostic)
    end)
    if ok then link_turn = result else link_turn_error = tostring(result) end
end

local ACTION_KEYS = {
    [1] = C.GBA_KEY.A,
    [2] = C.GBA_KEY.B,
    [3] = C.GBA_KEY.UP,
    [4] = C.GBA_KEY.DOWN,
    [5] = C.GBA_KEY.LEFT,
    [6] = C.GBA_KEY.RIGHT,
    [7] = C.GBA_KEY.START,
    [8] = C.GBA_KEY.SELECT,
    [9] = C.GBA_KEY.L,
    [10] = C.GBA_KEY.R,
}

local function json_escape(value)
    return tostring(value)
        :gsub("\\", "\\\\")
        :gsub('"', '\\"')
        :gsub("\b", "\\b")
        :gsub("\f", "\\f")
        :gsub("\n", "\\n")
        :gsub("\r", "\\r")
        :gsub("\t", "\\t")
end

local function json_string(value)
    return '"' .. json_escape(value) .. '"'
end

local function crc32_hex()
    local checksum = emu:checksum(C.CHECKSUM.CRC32)
    if not checksum or #checksum < 4 then return "" end
    local a, b, c, d = checksum:byte(1, 4)
    return string.format("%02x%02x%02x%02x", a, b, c, d)
end

local function identity()
    local raw_game_code = emu:getGameCode() or ""
    local game_code = raw_game_code:match("([%w][%w][%w][%w])$") or raw_game_code
    local checksum = crc32_hex()
    local game_code_supported = false
    for _, supported_code in ipairs(profile.game_codes) do
        if game_code == supported_code then
            game_code_supported = true
            break
        end
    end
    return game_code, raw_game_code, checksum,
        game_code_supported and checksum == profile.crc32
end

local function status_json()
    local game_code, raw_game_code, checksum, supported = identity()
    return table.concat({
        '{"ok":true',
        ',"adapter_version":', json_string(ADAPTER_VERSION),
        ',"profile":', json_string(profile.id),
        ',"game_code":', json_string(game_code),
        ',"raw_game_code":', json_string(raw_game_code),
        ',"rom_crc32":', json_string(checksum),
        ',"supported":', supported and "true" or "false",
        ',"reset_configured":', state_path and "true" or "false",
        ',"native_link_session":', FIRERED_RL_NATIVE_LINK_SESSION and "true" or "false",
        ',"link_session_id":', json_string(FIRERED_RL_LINK_SESSION_ID or ""),
        ',"link_player":', tostring(FIRERED_RL_LINK_PLAYER or 0),
        ',"auto_connect_supported":', link_connect and "true" or "false",
        ',"staged_connect_supported":', link_connect and link_connect.staged_supported and "true" or "false",
        ',"battle_start_supported":', link_connect and link_connect.battle_start_supported and "true" or "false",
        ',"link_turn_supported":', link_turn and "true" or "false",
        ',"link_peer_simulation_supported":', link_turn and link_turn.peer_simulation_supported and "true" or "false",
        ',"link_move_supported":', link_turn and link_turn.move_supported and "true" or "false",
        ',"link_human_supported":', link_turn and link_turn.human_supported and "true" or "false",
        ',"battle_events_supported":', link_turn and link_turn.battle_events_supported and "true" or "false",
        "}\n",
    })
end

local function error_json(message)
    return '{"ok":false,"error":' .. json_string(message) .. '}\n'
end

local function json_value(value)
    if type(value) == "table" then
        local parts = {}
        if value[1] ~= nil or next(value) == nil then
            for _, item in ipairs(value) do parts[#parts+1] = json_value(item) end
            return "[" .. table.concat(parts, ",") .. "]"
        end
        for key, item in pairs(value) do parts[#parts+1] = json_string(key) .. ":" .. json_value(item) end
        return "{" .. table.concat(parts, ",") .. "}"
    elseif type(value) == "number" or type(value) == "boolean" then return tostring(value)
    else return json_string(value) end
end

local function placeholder_state_json()
    return table.concat({
        '{"player_hp":0,"player_max_hp":1',
        ',"opponent_hp":0,"opponent_max_hp":1',
        ',"player_active":0,"opponent_active":0',
        ',"player_fainted":0,"opponent_fainted":0',
        ',"menu_state":0,"turn":0,"outcome":0}',
    })
end

local function inspection_json()
    local ok, report = pcall(function()
        return dofile(profile_path:gsub("%.lua$", "_inspect.lua"))(profile_path)
    end)
    if ok then return ',"inspection":' .. json_value(attach_revealed(report)) end
    return ',"inspection_error":' .. json_string(tostring(report))
end

local function integer_field(line, name)
    local value = line:match('"' .. name .. '"%s*:%s*(-?%d+)')
    if not value then return nil end
    return tonumber(value)
end

local function strict_integer_field(line, name)
    local raw = line:match('"' .. name .. '"%s*:%s*([^,%}%s]+)')
    if not raw or not raw:match("^%d+$") then error(name .. " must be an unsigned integer") end
    return assert(tonumber(raw))
end

local function step_response(action, frames, frame_start, frame_end)
    return table.concat({
        '{"ok":true,"state_ready":false',
        ',"action":', tostring(action),
        ',"frames_requested":', tostring(frames),
        ',"frame_start":', tostring(frame_start),
        ',"frame_end":', tostring(frame_end),
        ',"state":', placeholder_state_json(),
        inspection_json(),
        "}\n",
    })
end

local function text_input_safe()
    local ok, battle = pcall(function()
        return dofile(profile_path:gsub("%.lua$", "_battle.lua"))()
    end)
    if not ok or not battle.active or battle.outcome_raw ~= 0 then return false end
    for _, menu in ipairs(battle.menus) do
        if menu.state ~= "busy_or_unknown" then return false end
    end
    return true
end

local function apply_step(id, line)
    if link_turn and link_turn.active then
        send_response(id, error_json("a linked turn owns input; cancel it before stepping"))
        return
    end
    if link_connect and link_connect.active then
        send_response(id, error_json("link auto-connect owns input; cancel it before stepping"))
        return
    end
    local action = integer_field(line, "action")
    local frames = integer_field(line, "frames")
    if action == nil or action < 0 or action > 10 then
        send_response(id, error_json("action must be an integer from 0 through 10"))
        return
    end
    if frames == nil or frames < 1 or frames > MAX_STEP_FRAMES then
        send_response(id, error_json(
            "frames must be an integer from 1 through " .. MAX_STEP_FRAMES
        ))
        return
    end
    if pending_step or pending_reset then
        send_response(id, error_json("another step command is still pending"))
        return
    end

    local key = ACTION_KEYS[action]
    local expected_slot = integer_field(line, "expected_party_slot")
    if expected_slot ~= nil then
        local expected_submenu = line:match('"expected_party_submenu"%s*:%s*"([%w_]+)"')
        local ok, battle = pcall(function()
            return dofile(profile_path:gsub("%.lua$", "_battle.lua"))()
        end)
        local menu = ok and battle.menus[1] or nil
        if not menu or menu.state ~= "party" or menu.party_slot ~= expected_slot+1
            or menu.submenu ~= expected_submenu
            or (expected_submenu == "pokemon_actions" and menu.action_id ~= 10 and menu.action_id ~= 11) then
            emu:setKeys(0)
            send_response(id, error_json("party selection changed: expected slot " .. (expected_slot+1)
                .. " / " .. tostring(expected_submenu) .. "; observed cursor "
                .. tostring(menu and menu.cursor_raw) .. " / " .. tostring(menu and menu.submenu)))
            return
        end
    end
    local keys = 0
    if key then keys = 1 << key end
    local guard_menu = line:match('"guard_menu"%s*:%s*true') ~= nil
    if guard_menu and not text_input_safe() then keys = 0 end

    emu:setKeys(0)
    local frame_start = emu:currentFrame()
    emu:setKeys(keys)
    pending_step = {
        client_id = id,
        action = action,
        frames = frames,
        frame_start = frame_start,
        guard_menu = guard_menu,
    }
end

local function advance_pending_step()
    if not pending_step then return end
    if pending_step.guard_menu and not text_input_safe() then emu:setKeys(0) end
    local frame_end = emu:currentFrame()
    if frame_end - pending_step.frame_start < pending_step.frames then return end

    local completed = pending_step
    pending_step = nil
    emu:setKeys(0)
    send_response(completed.client_id, step_response(
        completed.action,
        completed.frames,
        completed.frame_start,
        frame_end
    ))
end

local function reset_fingerprint()
    -- Full EWRAM and IWRAM, defined in the selected platform profile.
    local hash = 2166136261
    for _, region in ipairs(profile.reset_regions) do
        local data = emu:readRange(region[1], region[2])
        if not data or #data ~= region[2] then error("incomplete reset RAM read") end
        for i = 1, #data do hash = ((hash ~ data:byte(i)) * 16777619) & 0xFFFFFFFF end
    end
    return string.format("%08x", hash)
end

local function advance_control()
    if observer then observer:sample() end
    if link_turn then link_turn:sample_events() end
    if link_connect and link_connect.active then link_connect:tick(); return end
    if link_turn and link_turn.active then link_turn:tick(); return end
    if not pending_reset then advance_pending_step(); return end
    local id = pending_reset
    pending_reset = nil
    emu:setKeys(0)
    local ok, response = pcall(function()
        -- Include the state's battery data so resetting a team restores its save too.
        if not emu:loadStateFile(state_path, C.SAVESTATE.ALL) then
            error("could not load configured emulator state: " .. state_path)
        end
        emu:setKeys(0)
        if observer then observer:reset(); observer:sample() end
        return '{"ok":true,"state_ready":false,"seed_applied":false' ..
            ',"frame":' .. tostring(emu:currentFrame()) ..
            ',"ram_fnv1a":' .. json_string(reset_fingerprint()) ..
            ',"state_path":' .. json_string(state_path) .. inspection_json() .. '}\n'
    end)
    send_response(id, ok and response or error_json(tostring(response)))
end

local function command_from_json(line)
    if not line:match("^%s*{%s*.-}%s*$") then
        return nil, "request must be one JSON object"
    end
    local command = line:match('"command"%s*:%s*"([^"\\]+)"')
    if not command then
        return nil, "request must contain a string command"
    end
    return command, nil
end

local function stop_client(id)
    local client = clients[id]
    if not client then return end
    if pending_step and pending_step.client_id == id then
        pending_step = nil
        emu:setKeys(0)
    end
    if pending_reset == id then pending_reset = nil; emu:setKeys(0) end
    if link_connect and link_connect.owner == id then link_connect:stop("client_disconnected") end
    if link_turn and link_turn.owner == id then link_turn:stop("client_disconnected") end
    clients[id] = nil
    client.socket:close()
end

send_response = function(id, payload)
    local client = clients[id]
    if not client then return false end
    local _, err = client.socket:send(payload)
    if err and err ~= socket.ERRORS.AGAIN then
        diagnostic("socket send error client=" .. id .. " error=" .. tostring(err))
        console:error("FireRed RL adapter send error: " .. tostring(err))
        stop_client(id)
        return false
    end
    return true
end

local function handle_line(id, line)
    local command, err = command_from_json(line)
    if not command then
        send_response(id, error_json(err))
        return
    end

    if command == "reset" or command == "step" or command == "inspect" or command == "link_status"
        or command == "link_connect" or command == "link_connect_continue" or command == "link_battle_start"
        or command == "link_turn_prepare" or command == "link_turn_commit" then
        local _, _, _, supported = identity()
        if not supported then send_response(id, error_json("unsupported ROM")); return end
    end
    if command == "ping" or command == "status" then
        send_response(id, status_json())
    elseif command == "close" then
        send_response(id, status_json())
        stop_client(id)
    elseif command == "link_turn_prepare" or command == "link_turn_commit"
        or command == "link_turn_status" or command == "link_turn_cancel" then
        local ok, result = pcall(function()
            assert(link_turn, link_turn_error or "turn control requires a native link session")
            if command == "link_turn_prepare" then
                assert(not pending_step and not pending_reset and not (link_connect and link_connect.active),
                    "another input command is pending")
                local kind = line:match('"kind"%s*:%s*"([a-z]+)"')
                assert(kind, "kind is required")
                link_turn:prepare(id, strict_integer_field(line, "request_id"),
                    strict_integer_field(line, "slot"), strict_integer_field(line, "expected_turn"), kind,
                    line:match('"replacement"%s*:%s*true') ~= nil,
                    line:match('"skip_move_menu"%s*:%s*true') ~= nil,
                    line:match('"human_opponent"%s*:%s*true') ~= nil,
                    line:match('"battle_log"%s*:%s*true') ~= nil)
            elseif command == "link_turn_commit" then
                link_turn:commit(id, strict_integer_field(line, "request_id"))
            elseif command == "link_turn_cancel" then
                assert(not link_turn.owner or link_turn.owner == id, "another client owns the linked turn")
                link_turn:stop()
            end
            local state = link_turn:status(integer_field(line, "events_after"))
            state.ok = true
            return state
        end)
        send_response(id, ok and (json_value(result) .. "\n") or error_json(tostring(result)))
    elseif command == "link_connect" or command == "link_connect_status" or command == "link_connect_cancel"
        or command == "link_connect_continue" or command == "link_battle_start" then
        local ok, result = pcall(function()
            assert(link_connect, link_connect_error or "auto-connect requires launch_link.py or launch_pool.py")
            if command == "link_connect" or command == "link_battle_start" or command == "link_connect_continue" then
                assert(not (link_turn and link_turn.active), "a linked turn owns input")
            end
            if command == "link_battle_start" then
                assert(not pending_step and not pending_reset, "another input command is pending")
                link_connect:start_battle(id)
            elseif command == "link_connect" then
                assert(not pending_step and not pending_reset, "another input command is pending")
                link_connect:start(id, line:match('"hold_at_service"%s*:%s*true') ~= nil)
            elseif command == "link_connect_continue" then
                link_connect:continue(id)
            elseif command == "link_connect_cancel" then
                assert(not link_connect.owner or link_connect.owner == id, "another client owns auto-connect")
                link_connect:stop()
            end
            local state = link_connect:status()
            state.ok = true
            return state
        end)
        send_response(id, ok and (json_value(result) .. "\n") or error_json(tostring(result)))
    elseif command == "link_status" then
        local ok, result = pcall(function()
            return dofile(profile_path:gsub("%.lua$", "_link.lua"))()
        end)
        send_response(id, ok and (json_value(result) .. "\n") or error_json(tostring(result)))
    elseif command == "inspect" then
        if pending_step or pending_reset then
            send_response(id, error_json("another control command is pending"))
            return
        end
        local ok, result = pcall(function()
            local inspect = dofile(profile_path:gsub("%.lua$", "_inspect.lua"))
            return attach_revealed(inspect(profile_path))
        end)
        send_response(id, ok and (json_value(result) .. "\n") or error_json(tostring(result)))
    elseif command == "step" then
        apply_step(id, line)
    elseif command == "reset" then
        if FIRERED_RL_NATIVE_LINK_SESSION then
            send_response(id, error_json("individual reset is disabled in a link session; restart both windows through launch_link.py"))
        elseif not state_path then
            send_response(id, error_json("no reset state configured; launch with --states"))
        elseif pending_step or pending_reset then
            send_response(id, error_json("another control command is pending"))
        elseif not profile.reset_regions then
            send_response(id, error_json("profile has no reset fingerprint regions"))
        else
            pending_reset = id
        end
    else
        send_response(id, error_json("unknown command: " .. command))
    end
end

local function receive_client(id)
    local client = clients[id]
    if not client then return end

    while true do
        local chunk, err = client.socket:receive(4096)
        if chunk then
            client.buffer = client.buffer .. chunk
            while true do
                local newline = client.buffer:find("\n", 1, true)
                if not newline then break end
                local line = client.buffer:sub(1, newline - 1):gsub("\r$", "")
                client.buffer = client.buffer:sub(newline + 1)
                if #line > 0 then handle_line(id, line) end
                if not clients[id] then return end
            end
            if #client.buffer > 65536 then
                send_response(id, error_json("request line exceeds 65536 bytes"))
                stop_client(id)
                return
            end
        elseif err == socket.ERRORS.AGAIN then
            return
        else
            diagnostic("socket receive ended client=" .. id .. " error=" .. tostring(err))
            stop_client(id)
            return
        end
    end
end

local function accept_client()
    local client_socket, err = server:accept()
    if not client_socket then
        if err ~= socket.ERRORS.AGAIN then
            console:error("FireRed RL adapter accept error: " .. tostring(err))
        end
        return
    end

    local id = next_client_id
    next_client_id = next_client_id + 1
    clients[id] = { socket = client_socket, buffer = "" }
    client_socket:add("received", function() receive_client(id) end)
    client_socket:add("error", function(_, socket_err)
        diagnostic("socket error client=" .. id .. " error=" .. tostring(socket_err))
        console:error("FireRed RL adapter client error: " .. tostring(socket_err))
        stop_client(id)
    end)
    console:log("FireRed RL adapter: client " .. id .. " connected")
    diagnostic("client connected: " .. id)
end

local function shutdown(reason)
    diagnostic("adapter shutdown: " .. tostring(reason))
    pending_step = nil
    pending_reset = nil
    if link_connect then link_connect:stop("shutdown") end
    if link_turn then link_turn:stop("shutdown") end
    emu:setKeys(0)
    for id, _ in pairs(clients) do stop_client(id) end
    if server then
        server:close()
        server = nil
    end
end

local port = tonumber(FIRERED_RL_PORT) or DEFAULT_PORT
server = socket.bind("127.0.0.1", port)
if not server then
    error("FireRed RL adapter could not bind to 127.0.0.1:" .. port)
end

local ok, listen_error = server:listen(1)
if not ok then
    server:close()
    server = nil
    error("FireRed RL adapter could not listen: " .. tostring(listen_error))
end

server:add("received", accept_client)
callbacks:add("frame", advance_control)
callbacks:add("stop", function() shutdown("emulator stopped") end)
callbacks:add("crashed", function() shutdown("emulated core crashed") end)

local game_code, raw_game_code, checksum, supported = identity()
console:log(string.format(
    "FireRed RL adapter %s listening on 127.0.0.1:%d; game=%s raw_game=%s crc32=%s supported=%s",
    ADAPTER_VERSION, port, game_code, raw_game_code, checksum, tostring(supported)
))
