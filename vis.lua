colors = {
  HEADER = '\27[95m',
  OKBLUE = '\27[94m',
  OKGREEN = '\27[92m',
  WARNING = '\27[93m',
  FAIL = '\27[91m',
  ENDC = '\27[0m',
  BOLD = '\27[1m',
  UNDERLINE = '\27[4m',
  RESET = '\27[0m'
}

local decimalPlaces = 2

function lines(str)
  local t = {}
  local function helper(line) table.insert(t, line) return "" end
  helper((str:gsub("(.-)\r?\n", helper)))
  return t
end

function flatten(tensor)
  return tensor:reshape(1, tensor:nElement())
end

function round(tensor, places)
  places = places or decimalPlaces
  local tensorClone = tensor:clone()
  local offset = 0
  if tensor:sum() ~= 0 then
    offset = - math.floor(math.log10(torch.abs(tensorClone):mean())) + (places - 1)
  end

  tensorClone = tensorClone * (10 ^ offset)
  tensorClone:round()
  tensorClone = tensorClone / (10 ^ offset)

  if tostring(tensorClone[1]) == tostring(0/0) then
    print(tensor)
    print(math.floor(math.log10(torch.abs(tensorClone):mean())))
    print(offset)
    error("got nan")
  end

  return tensorClone
end

function simplestr(tensor)
  local decimalPlaces = 2
  local rounded = round(tensor)
  -- local rounded = tensor:clone()

  local strTable = lines(tostring(flatten(rounded)))
  table.remove(strTable, #strTable)
  table.remove(strTable, #strTable)

  local str = ""
  for i, line in ipairs(strTable) do
    str = str..line
  end
  return str
end

function prettySingleError(number)
  local str = tostring(number)
  if math.abs(number) < 1e-10 then
    return '0.0000'
  else
    return colors.FAIL..str..colors.RESET
  end
end

function prettyError(err)
  if type(err) == 'number' then
    return prettySingleError(err)
  elseif type(err) == 'table' then
    local str = ''
    for i, val in ipairs(err) do
      str = str .. ' ' .. prettySingleError(val)
    end
    return str
  elseif err.size then -- assume tensor
    local rounded = round(err)
    if rounded:nDimension() ~= 1 then
      error("Only able to pretty-print 1D tensors.")
    else
      local str = ''
      for i = 1, rounded:size(1) do
        str = str .. ' ' .. prettySingleError(rounded[i])
      end
      return str
    end
  else
    error("Not sure what to do with this object.")
  end
end




















