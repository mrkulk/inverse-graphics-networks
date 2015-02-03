require 'nn'
require 'parallel'

local ParallelParallel, parent = torch.class('nn.ParallelParallel', 'nn.Container')

function moduleWrapper()
  require 'nn'
  require 'ACR'
  require 'INTM'
  require 'Bias'
  require 'PrintModule'
  require 'INTMReg'
  require 'gradACRWrapper'

  torch.setnumthreads(1)

  local modelKeys = {'output', 'weight', 'bias', 'gradInput', 'gradWeight', 'gradBias', 'test'}
  function sanitize(m)
    if torch.typename(m) then
      local current = {}
      for _, key in ipairs(modelKeys) do
        current[key] = m[key]
      end

      if m.modules then
        local modules = {}
        for index, child in ipairs(m.modules) do
          table.insert(modules, sanitize(child))
        end
        current.modules = modules
      end
      return current
    elseif type(m) == 'table' then
      local current = {}
      for index, elem in ipairs(m) do
        table.insert(current, sanitize(elem))
      end
    else
      error("Must be a single torch object or a list of them!")
    end
  end

  function updateIn(m, tbl)
    -- recursively update any changed modules
    if tbl.modules then
      for moduleIndex, updates in pairs(tbl.modules) do
        updateIn(m.modules[moduleIndex], updates)
      end
    end

    -- set all the updated keys in this module
    for key, value in pairs(tbl) do
      if key ~= 'modules' then
        m[key] = value
      end
    end
  end

  function fetchFrom(m, keyList)
    if type(keyList) == 'table' then
      if #keyList > 1 then
        local key = table.remove(keyList, 1)
        return fetchFrom(m[key], keyList)
      else
        return m[keyList[1]]
      end
    else
      return m[keyList]
    end
  end

  parallel.yield()
  local mod = nil
  while true do
    local message = parallel.parent:receive()
    local messageType = message[1]
    local data = message[2]

    if messageType == 'module' then
      mod = data
      -- use with CPU parallel in the grad calculation
      -- for i, m in pairs(mod:findModules('nn.ACR')) do
      --   m:makeThreads()
      -- end
    elseif messageType == 'updateOutput' then
      parallel.parent:send(mod:updateOutput(data))
    elseif messageType == 'updateGradInput' then
      parallel.parent:send(mod:updateGradInput(data[1], data[2]))
    elseif messageType == 'accGradParameters' then
      mod:accGradParameters(data[1], data[2], data[3])
    elseif messageType == 'accUpdateGradParameters' then
      mod:accUpdateGradParameters(data[1], data[2], data[3])
    elseif messageType == 'zeroGradParameters' then
      mod:zeroGradParameters()
    elseif messageType == 'updateParameters' then
      mod:updateParameters(data)
    elseif messageType == 'training' then
      mod:training()
    elseif messageType == 'evaluate' then
      mod:evaluate()
    elseif messageType == 'share' then
      mod:evaluate(data[1], data[2])
    elseif messageType == 'reset' then
      mod:reset(data)
    elseif messageType == 'update' then
      updateIn(mod, data)
      parallel.parent:send(true)
    elseif messageType == 'dump' then
      parallel.parent:send(sanitize(mod))
    elseif messageType == 'fetch' then
      parallel.parent:send(fetchFrom(mod, data))
    else
      error("Not a valid message type!")
    end
  end
end

-- if I want to bring back caching
-- local modelKeys = {'output', 'weight', 'bias', 'gradInput', 'gradWeight', 'gradBias'}
function buildCache(module, parent, childIndex)
  -- print(module)
  local cache = {_hiddenData = {}}
  cache.__typename = torch.typename(module)
  if parent then
    cache._parent = parent
    cache._childIndex = childIndex
  end

  -- also for caching
  -- for _, key in ipairs(modelKeys) do
  --   cache._hiddenData[key] = module[key]
  -- end
  if module.modules then
    cache._hiddenData.modules = {}
    for i, submodule in ipairs(module.modules) do
      table.insert(cache._hiddenData.modules, buildCache(submodule, cache, i))
    end
  end
  return cache
end

function setmetatable_recursive(tbl, mt)
  if tbl._hiddenData.modules then
    for _, module in ipairs(tbl._hiddenData.modules) do
      setmetatable_recursive(module, mt)
    end
  end
  setmetatable(tbl, mt)
end

function ParallelParallel:__init(inputDimension,outputDimension)
  parent.__init(self)
  self.modules = {}
  self.size = torch.LongStorage()
  self.inputDimension = inputDimension
  self.outputDimension = outputDimension
  self.childThreads = {}

  -- make self.childThreads act just like parallel.children
  parallel._fill(self.childThreads)
end

function setval_remote(tbl, key, value)
  if tbl._threadConnection then
    tbl._threadConnection:send({ 'update', {[key] = value}  })
    tbl._threadConnection:receive()
  elseif tbl._parent then
    setval_remote(tbl._parent, 'modules', {[tbl._childIndex] = {[key] = value}})
  else
    error("No parent or threadConnection!")
  end
end

function ParallelParallel:add(mod)
  local child = parallel.fork()
  child:exec(moduleWrapper)
  table.insert(self.childThreads, child)

  local localmod = buildCache(mod)
  localmod._threadConnection = child

  setmetatable_recursive(localmod, {
    __index = function(tbl, key)
      if type(key) == 'string' then
        if string.sub(key, 1, 1) == '_' then
          -- if the table had that key, this query wouldn't be here
          return nil
        elseif key == 'modules' then
          return tbl._hiddenData[key]
        end
      end

      if tbl._threadConnection then
        if type(key) ~= 'table' then key = {key} end
        tbl._threadConnection:send({'fetch', key})
        return tbl._threadConnection:receive()
      elseif tbl._parent then
        if type(key) == 'table' then
          table.insert(key, 1, tbl._childIndex)
          table.insert(key, 1, 'modules')
          return tbl._parent[key]
        else
          return tbl._parent[{'modules', tbl._childIndex, key}]
        end
      else
        error("No parent or threadConnection found!")
      end

    end,
    __newindex = function(tbl, key, value)
      setval_remote(tbl, key, value)
      -- #caching
      -- tbl._hiddenData[key] = value
    end
    })

  table.insert(self.modules, localmod)
  child:join()
  child:send({'module', mod})
  return self
end

function ParallelParallel:updateOutput(input)
  local nModule=input:size(self.inputDimension)

  for i = 1, #self.modules do
    local currentInput = input:select(self.inputDimension,i)
    self.childThreads[i]:send({'updateOutput', currentInput})
  end

  local outputs = self.childThreads:receive()

  for i = 1, #outputs do
    -- #caching
    -- self.modules[i].output = outputs[i]
    local currentOutput = outputs[i]
    local outputSize = currentOutput:size(self.outputDimension)

    if i == 1 then
      self.size:resize(currentOutput:dim()):copy(currentOutput:size())
    else
      self.size[self.outputDimension] = self.size[self.outputDimension] + outputSize
    end

  end
  self.output:resize(self.size)

  local offset = 1
  for i=1,nModule do
    local currentOutput = outputs[i]
    local outputSize = currentOutput:size(self.outputDimension)
    self.output:narrow(self.outputDimension, offset, outputSize):copy(currentOutput)
    offset = offset + currentOutput:size(self.outputDimension)
  end
  return self.output
end

function ParallelParallel:updateGradInput(input, gradOutput)
  local nModule=input:size(self.inputDimension)
  self.gradInput:resizeAs(input)
  local offset = 1

  for i = 1, #self.modules do
    -- #caching
    -- self.childThreads[i]:join()
    local currentInput = input:select(self.inputDimension,i)
    local currentOutput = self.modules[i].output
    local outputSize = currentOutput:size(self.outputDimension)
    local currentGradOutput = gradOutput:narrow(self.outputDimension, offset, outputSize)
    self.childThreads[i]:send({'updateGradInput', {currentInput, currentGradOutput}})
  end

  local gradInputs = parallel.children:receive()

  for i=1,nModule do
    local currentGradInput = gradInputs[i]
    local currentOutput = self.modules[i].output
    local outputSize = currentOutput:size(self.outputDimension)
    self.gradInput:select(self.inputDimension,i):copy(currentGradInput)
    offset = offset + outputSize
  end
  return self.gradInput
end

function ParallelParallel:accGradParameters(input, gradOutput, scale)
  local nModule=input:size(self.inputDimension)

  local offset = 1
  for i=1,nModule do
    local mod = self.modules[i]
    local currentOutput = mod.output
    local outputSize = currentOutput:size(self.outputDimension)

    self.childThreads[i]:send({
      'accGradParameters',
      {input:select(self.inputDimension,i),
        gradOutput:narrow(self.outputDimension, offset,outputSize),
        scale}})

    offset = offset + outputSize
  end
end

function ParallelParallel:accUpdateGradParameters(input, gradOutput, lr)
  local nModule=input:size(self.inputDimension)

  local offset = 1
  for i=1,nModule do
    local mod = self.modules[i];
    local currentOutput = mod.output

    self.childThreads[i]:send({
      'accUpdateGradParameters',
      {input:select(self.inputDimension,i),
        gradOutput:narrow(self.outputDimension, offset,
                   currentOutput:size(self.outputDimension)),
        lr}})

    offset = offset + currentOutput:size(self.outputDimension)
  end
end

function ParallelParallel:zeroGradParameters()
  for i = 1, #self.modules do
    self.childThreads[i]:send({'zeroGradParameters'})
  end
end

function ParallelParallel:training()
  for i = 1, #self.modules do
    self.childThreads[i]:send({'training'})
  end
end

function ParallelParallel:evaluate()
  for i = 1, #self.modules do
    self.childThreads[i]:send({'evaluate'})
  end
end

function ParallelParallel:reset(stdv)
  for i = 1, #self.modules do
    self.childThreads[i]:send({'reset', stdv})
  end
end

function ParallelParallel:updateParameters(learningRate)
  for i = 1, #self.modules do
    self.childThreads[i]:send({'updateParameters', learningRate})
  end
end

function ParallelParallel:share(mlp, ...)
  for i = 1, #self.modules do
    self.childThreads[i]:send({'share', {mlp.modules[i], ...}})
  end
end

-- function ParallelParallel:findModules(name, container)
--   container = container or self
--   local nodes = {}
--   local containers = {}
--   local mod_type = torch.typename(self)
--   if mod_type == typename then
--     nodes[#nodes+1] = self
--     containers[#containers+1] = container
--   end
--   -- Recurse on nodes with 'modules'
--   if (self.modules ~= nil) then
--     if (torch.type(self.modules) == 'table') then
--       for i = 1, #self.modules do
--         local child = self.modules[i]
--         local cur_nodes, cur_containers = child:findModules(typename, self)
--         assert(#cur_nodes == #cur_containers,
--           'Internal error: incorrect return length')  -- This shouldn't happen
--         -- add the list items from our child to our list (ie return a
--         -- flattened table of the return nodes).
--         for j = 1, #cur_nodes do
--           nodes[#nodes+1] = cur_nodes[j]
--           containers[#containers+1] = cur_containers[j]
--         end
--       end
--     end
--   end
--   -- return nodes, containers


--   container = container or self

--   for i = 1, #self.modules do
--     self.childThreads[i]:send({'findModules', {}})
--   end
--   local results = self.childThreads:receive()
--   -- return self.modules
-- end

function ParallelParallel:__tostring__()
  local tab = '  '
  local line = '\n'
  local next = '  |`-> '
  local ext = '  |    '
  local extlast = '       '
  local last = '   ... -> '
  local str = torch.type(self)
  str = str .. ' {' .. line .. tab .. 'input'
  for i=1,#self.modules do
    if i == self.modules then
      str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
    else
      str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
    end
  end
  str = str .. line .. tab .. last .. 'output'
  str = str .. line .. '}'
  return str
end