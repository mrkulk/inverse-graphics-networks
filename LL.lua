Threads = require 'threads'

local LL = {}
function LL:new(nThreads, imports)
  local new = {}
  imports = imports or {'nn'}
  self.nThreads = nThreads or 4
  self.__index = self
  self.threads = Threads(nThreads,
    function()
      torch.setnumthreads(1)
      for i = 1,#imports do
        require(imports[i])
      end
    end
  )
  return setmetatable(new, self)
end

function LL:pmap(indexable, fn)
  local results = {}
  local size = 0
  local flatIndexable = nil
  if indexable.torch ~= nil then
    size = indexable:nElement()
    flatIndexable = indexable:view(indexable:nElement())
  elseif indexable.size ~= nil then
    size = indexable:size()
    flatIndexable = indexable
  elseif type(indexable) == 'table' then
    size = #indexable
    flatIndexable = indexable
  end

  for i = 1, size do
    local element = flatIndexable[i]
    self.threads:addjob(
      function()
        local res = nil
        local status, err = pcall(function()
          local index = i
          local arg = element
          local mFn = fn
          res = {index, mFn(index, arg)}
        end)
        if not status then
          print("Error in a thread. Check imports in the initialization!")
          print(err)
        end
        return res
      end,

      function(result)
        results[result[1]] = result[2]
      end
    )
  end

  self.threads:synchronize()
  return results
end

function LL:pmap_mut(indexable, fn)
  local results = {}
  -- local indexable_mut = {}

  local size = 0
  local flatIndexable = nil
  if indexable.torch ~= nil then
    size = indexable:nElement()
    flatIndexable = indexable:view(indexable:nElement())
  elseif indexable.size ~= nil then
    size = indexable:size()
    flatIndexable = indexable
  elseif type(indexable) == 'table' then
    size = #indexable
    flatIndexable = indexable
  end

  for i = 1, size do
    local element = flatIndexable[i]
    self.threads:addjob(
      function()
        local res = nil
        local status, err = pcall(function()
          local index = i
          local arg = element
          local mFn = fn
          res = {index, arg, mFn(index, arg)}
        end)
        if not status then
          print("Error in a thread. Check imports in the initialization!")
          print(err)
        end
        return res
      end,

      function(result)
        indexable[result[1]] = result[2]
        results[result[1]] = result[3]
      end
    )
  end

  self.threads:synchronize()
  return results
end

return LL
