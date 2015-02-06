colors = {
  HEADER = '\27[95m',
  OKBLUE = '\27[94m',
  OKGREEN = '\27[92m',
  WARNING = '\27[93m',
  FAIL = '\27[91m',
  ENDC = '\27[0m',
  BOLD = '\27[1m',
  UNDERLINE = '\27[4m'
}

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
  local tensorClone = tensor:clone()
  local offset = - math.floor(math.log10(torch.abs(tensorClone):mean())) + (places - 1)

  tensorClone = tensorClone * (10 ^ offset)
  tensorClone:round()
  tensorClone = tensorClone / (10 ^ offset)
  return tensorClone
end

function simplestr(tensor)
  local decimalPlaces = 2
  local rounded = round(tensor, decimalPlaces)

  local strTable = lines(tostring(flatten(rounded)))
  table.remove(strTable, #strTable)
  table.remove(strTable, #strTable)

  local str = ""
  for i, line in ipairs(strTable) do
    str = str..line
  end
  return str
end