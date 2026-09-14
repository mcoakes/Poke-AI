-- Observe issued "used move" messages, not selected moves or hidden movesets.
local observer = {records={}, slots={}, active_index=0}
function observer:reset()
    self.records, self.slots, self.active_index = {}, {}, 0
    self.skip_existing = true
    self.was_active = false
end
function observer:sample()
    if (emu:read8(0x03003529) & 2) == 0 then self.active_index=0; self.was_active=false; return end
    if not self.was_active then self:reset(); self.was_active=true end
    local count = emu:read8(0x02023BCC)
    if count < 1 or count > 4 then return end
    local exec = emu:read32(0x02023BC8)
    for id=0,count-1 do
        if (emu:read8(0x02023BD6+id) & 1) == 1 and (emu:read8(0x02023D70) & (1 << id)) == 0 then
            local slot = emu:read16(0x02023BCE+id*2)
            local species = emu:read16(0x02023BE4+id*0x58)
            if slot < 6 and species > 0 and species <= 411 then
                local index = self.slots[slot]
                if not index then
                    index = #self.records+1
                    self.slots[slot] = index
                    self.records[index] = {species_id=species, moves={}}
                end
                self.active_index = index
                local record = self.records[index]
                record.species_id = species
                local buffer = 0x02022BC4+id*0x200
                if not self.skip_existing and (exec & (1 << id)) ~= 0
                    and emu:read8(buffer) == 16 and emu:read16(buffer+2) == 4 then
                    local move = emu:read16(buffer+4)
                    if move > 0 and move <= 354 then
                        local found = false
                        for _, known in ipairs(record.moves) do if known.move_id == move then found=true end end
                        if not found then
                            if #record.moves == 4 then table.remove(record.moves, 1) end
                            record.moves[#record.moves+1] = {move_id=move,
                                type_id=emu:read8(0x08250C04+move*12+2)}
                        end
                    end
                end
            end
        end
    end
    self.skip_existing = false
end
observer:reset()
return observer
