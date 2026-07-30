--[[
  shortcaps.lua -- short List-of-Tables entries.

  A caption written as "Short Title: full descriptive caption." keeps its full
  text under the table but contributes only "Short Title" to the List of
  Tables, via LaTeX's optional caption argument: \caption[Short]{Short: ...}.

  Figures are handled in tex/preamble.tex instead: Quarto generates their
  \caption call after every Lua filter has run, so it can only be patched in
  LaTeX. Tables cannot be handled there, because longtable redefines \caption
  locally inside its own environment, which is why they are done here.

  Must run AFTER Quarto's own filters (see the `filters:` key in _quarto.yml).
]]

local MAX_SCAN = 30   -- inlines to look through before giving up on a colon

-- The part of a caption before the first colon, as plain text, or nil.
local function short_text(inlines)
  local parts = {}
  for j = 1, math.min(#inlines, MAX_SCAN) do
    local e = inlines[j]
    if e.t == "Str" then
      if e.text:sub(-1) == ":" then
        parts[#parts + 1] = e.text:sub(1, -2)
        local short = table.concat(parts)
        if short ~= "" then return short end
        return nil
      end
      parts[#parts + 1] = e.text
    elseif e.t == "Space" then
      parts[#parts + 1] = " "
    else
      return nil   -- markup before the colon: not a short-title caption
    end
  end
  return nil
end

function Table(t)
  local cap = t.caption
  if not cap or not cap.long or #cap.long == 0 then return nil end
  local first = cap.long[1]
  if first.t ~= "Plain" and first.t ~= "Para" then return nil end
  local short = short_text(first.content)
  if not short then return nil end
  cap.short = { pandoc.Str(short) }
  t.caption = cap
  return t
end
