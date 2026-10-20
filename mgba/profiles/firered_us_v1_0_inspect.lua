-- Read-only first-pass party/bag decoder. All layout details stay in this profile.
local orders = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,3,1,2},{0,2,3,1},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{2,0,1,3},{3,0,1,2},{2,0,3,1},{3,0,2,1},
    {1,2,0,3},{1,3,0,2},{2,1,0,3},{3,1,0,2},{2,3,0,1},{3,2,0,1},
    {1,2,3,0},{1,3,2,0},{2,1,3,0},{3,1,2,0},{2,3,1,0},{3,2,1,0},
}
local pockets = {
    {"items",0x310,42},{"key_items",0x3B8,30},{"poke_balls",0x430,13},
    {"tm_hm",0x464,58},{"berries",0x54C,43},
}
local function byte(s, offset) return s:byte(offset + 1) end
local function word(s, offset) return (string.unpack("<I2", s, offset + 1)) end
local function dword(s, offset) return (string.unpack("<I4", s, offset + 1)) end
local function pointer(address, size)
    local value = emu:read32(address)
    assert(value >= 0x02000000 and value + size <= 0x02040000 and value % 4 == 0,
           "save blocks not ready; load Continue or a valid state first")
    return value
end

local function decode_party(address, count, scan_slots)
    local party = emu:readRange(address, 600)
    local report = {ok=true, profile="firered_us_v1_0", state_ready=false,
        frame=emu:currentFrame(), native_v2=1, party_count=count, party={}, bag={}, warnings={},
        remaining=0, fainted=0, counts_valid=true}
    if count < 1 or count > 6 then
        report.counts_valid = false
        report.warnings[1] = "party count not initialized"
        return report
    end
    for slot = 1, 6 do
        local mon = {slot=slot, occupied=slot <= count}
        report.party[slot] = mon
        if mon.occupied then
            local raw = party:sub((slot-1)*100+1, slot*100)
            local pid, ot = dword(raw,0), dword(raw,4)
            local decrypted = {}
            for i=0,11 do decrypted[#decrypted+1] = string.pack("<I4", dword(raw,32+i*4) ~ pid ~ ot) end
            local data = table.concat(decrypted)
            local sum = 0
            for i=0,23 do sum = (sum + word(data, i*2)) & 0xFFFF end
            mon.checksum_valid = sum == word(raw,28)
            mon.bad_egg = (byte(raw,19) & 1) ~= 0
            if not mon.checksum_valid or mon.bad_egg then
                mon.valid = false
                report.counts_valid = false
                local issues = {}
                if not mon.checksum_valid then
                    issues[#issues+1] = string.format("checksum failure (expected=%04x actual=%04x)", word(raw,28), sum)
                end
                if mon.bad_egg then issues[#issues+1] = "Bad Egg flag set" end
                report.warnings[#report.warnings+1] = "slot " .. slot .. ": " .. table.concat(issues, "; ")
            else
                local order = orders[pid % 24 + 1]
                local growth, attacks, misc = order[1]*12, order[2]*12, order[4]*12
                mon.species_id = word(data,growth)
                if scan_slots and mon.species_id == 0 then
                    mon.occupied = false
                    mon.valid = false
                else
                mon.held_item_id = word(data,growth+2)
                mon.is_egg = (dword(data,misc+4) & 0x40000000) ~= 0
                mon.level, mon.current_hp, mon.max_hp = byte(raw,84), word(raw,86), word(raw,88)
                mon.status_raw = dword(raw,80)
                mon.valid = mon.species_id >= 1 and mon.species_id <= 411
                    and mon.level >= 1 and mon.level <= 100 and mon.current_hp <= mon.max_hp
                if not mon.valid then
                    report.counts_valid = false
                    report.warnings[#report.warnings+1] = "slot " .. slot .. ": implausible species/level/HP"
                else
                    local base = 0x08254784 + mon.species_id*28
                    mon.base_type_ids = {emu:read8(base+6), emu:read8(base+7)}
                    mon.ivs_raw = dword(data,misc+4) & 0x3FFFFFFF
                    mon.ability_id = emu:read8(base+22+((dword(data,misc+4) >> 31) & 1))
                    mon.stats = {word(raw,90),word(raw,92),word(raw,96),word(raw,98),word(raw,94)}
                    mon.moves = {}
                    for move_slot=0,3 do
                        local id = word(data,attacks+move_slot*2)
                        local move = {slot=move_slot+1, move_id=id, current_pp=byte(data,attacks+8+move_slot)}
                        mon.moves[#mon.moves+1] = move
                        if id >= 1 and id <= 354 then
                            local entry = 0x08250C04 + id*12
                            local base_pp = emu:read8(entry+4)
                            local bonus = (byte(data,growth+8) >> (move_slot*2)) & 3
                            move.type_id = emu:read8(entry+2)
                            move.max_pp = base_pp + (base_pp*bonus)//5
                        elseif id ~= 0 then
                            report.warnings[#report.warnings+1] = "slot " .. slot .. ": unknown move ID " .. id
                        end
                    end
                    if not mon.is_egg then
                        if mon.current_hp > 0 then report.remaining = report.remaining + 1
                        else report.fainted = report.fainted + 1 end
                    end
                end
                end
            end
        end
    end
    return report
end

return function(profile_path)
    local count = emu:read8(0x02024029)
    assert(count >= 1 and count <= 6, "party not ready; load into the overworld first")
    local sb1, sb2 = pointer(0x03005008, 0x3D68), pointer(0x0300500C, 0xF24)
    local bag_key = emu:read16(sb2 + 0xF20)
    local report = decode_party(0x02024284, count)
    for _, pocket in ipairs(pockets) do
        local entries = {}
        report.bag[pocket[1]] = entries
        local bytes = emu:readRange(sb1+pocket[2], pocket[3]*4)
        for i=0,pocket[3]-1 do
            local id = word(bytes,i*4)
            if id ~= 0 then
                local quantity = word(bytes,i*4+2) ~ bag_key
                entries[#entries+1] = {slot=i+1, item_id=id, quantity=quantity}
                if quantity == 0 or quantity > 999 then
                    report.warnings[#report.warnings+1] = pocket[1] .. ": unusual quantity at slot " .. (i+1)
                end
            end
        end
    end
    if profile_path then
        report.battle = dofile(profile_path:gsub("%.lua$", "_battle.lua"))()
        if report.battle.active then
            if report.battle.screen == "party" then
                if report.battle.party_order_valid then
                    local canonical = {}
                    for display_slot, original_slot in ipairs(report.battle.party_order) do
                        local mon = report.party[display_slot]
                        mon.slot = original_slot
                        canonical[original_slot] = mon
                    end
                    report.party = canonical
                else
                    report.counts_valid = false
                    report.warnings[#report.warnings+1] = "battle party order is not initialized"
                end
            end
            -- gEnemyPartyCount is cached and need not be populated for NPC battles.
            -- Decode all six slots; species zero denotes an empty slot.
            local enemy = decode_party(0x0202402C, 6, true)
            report.battle.opponent_party = enemy.party
            -- Active HP can precede the copy back to the party during animations.
            for _, mon in ipairs(report.battle.battlers) do
                local side = mon.side == "player" and report or enemy
                local stored = side.party[mon.party_slot]
                if mon.valid and not mon.absent and stored and stored.valid and not stored.is_egg then
                    if stored.current_hp > 0 and mon.current_hp == 0 then
                        side.remaining = side.remaining - 1
                        side.fainted = side.fainted + 1
                    elseif stored.current_hp == 0 and mon.current_hp > 0 then
                        side.remaining = side.remaining + 1
                        side.fainted = side.fainted - 1
                    end
                end
            end
            report.battle.player_counts = {remaining=report.remaining, fainted=report.fainted, valid=report.counts_valid}
            report.battle.opponent_counts = {remaining=enemy.remaining, fainted=enemy.fainted, valid=enemy.counts_valid}
            report.battle.opponent_counts.warnings = enemy.warnings
            report.battle.opponent_counts.cached_party_count = emu:read8(0x0202402A)
        end
    end
    return report
end
