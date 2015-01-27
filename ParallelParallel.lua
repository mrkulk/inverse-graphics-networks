require 'nn'
LL = require 'LL'
moses = require 'moses'
require 'parallel'

local ParallelParallel, parent = torch.class('nn.ParallelParallel', 'nn.Container')

function moduleWrapper()
   require 'nn'
   require 'ACR'
   require 'INTM'
   require 'Bias'

   torch.setnumthreads(1)

   -- parallel.parent:send("started")
   -- parallel.print("I'M ALIIIIVE")
   local mod = nil
   while true do
      -- parallel.print("Yielding...")
      m = parallel.yield()
      -- parallel.print("Join received.")
      if m == 'break' then break end

      local message = parallel.parent:receive()
      local messageType = message[1]
      local data = message[2]

      -- parallel.print(message)
      if messageType == 'module' then
         mod = data
      elseif messageType == 'updateOutput' then
         mod:updateOutput(data)
         parallel.parent:send(mod)
      elseif messageType == 'updateGradInput' then
         -- parallel.print(data)
         mod:updateGradInput(data[1], data[2])
         parallel.parent:send(mod)
      end
   end
end

function ParallelParallel:__init(inputDimension,outputDimension)
   parent.__init(self)
   self.modules = {}
   self.size = torch.LongStorage()
   self.inputDimension = inputDimension
   self.outputDimension = outputDimension

   -- self.ll = LL:new(nThreads, {'nn', 'ParallelParallel', 'Bias', 'ACR', 'INTM'})
end

function ParallelParallel:add(mod)
   local child = parallel.fork()
   child:exec(moduleWrapper)
   table.insert(self.modules, mod)
   -- os.execute("sleep " .. tonumber(1))
   return self
end

function ParallelParallel:updateOutput(input)
   local nModule=input:size(self.inputDimension)

   -- parallel.print(parallel.children)
   -- parallel.print("issuing join")

   for i = 1, #self.modules do
      parallel.children[i]:join()
      -- parallel.print("sending module")
      parallel.children[i]:send({'module', self.modules[i]})
   end

   -- parallel.print("issuing join #2")
   -- parallel.children:join()
   for i = 1, #self.modules do
      parallel.children[i]:join()
      local currentInput = input:select(self.inputDimension,i)
      -- parallel.print("sending updateOutput")
      parallel.children[i]:send({'updateOutput', currentInput})
   end

   local updatedModules = parallel.children:receive()

   -- local inputDimension = self.inputDimension
   -- local runModule = function(inputTensor, i, module)
   --    local currentInput = inputTensor:select(inputDimension,i)
   --    return module:updateOutput(currentInput)
   -- end
   -- local outputs = self.ll:pmap_mut(self.modules, moses.bind(runModule, input))

   for i = 1, #updatedModules do
      self.modules[i] = updatedModules[i]
      local currentOutput = updatedModules[i].output
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
      local currentOutput = self.modules[i].output
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
      parallel.children[i]:join()
      parallel.children[i]:send({'module', self.modules[i]})
   end

   for i = 1, #self.modules do
      parallel.children[i]:join()
      local currentInput = input:select(self.inputDimension,i)
      local currentOutput = self.modules[i].output
      local outputSize = currentOutput:size(self.outputDimension)
      local currentGradOutput = gradOutput:narrow(self.outputDimension, offset, outputSize)
      -- parallel.print(currentGradOutput)
      parallel.children[i]:send({'updateGradInput', {currentInput, currentGradOutput}})
   end

   local updatedModules = parallel.children:receive()
   -- print(updatedModules)

   -- local inputDimension = self.inputDimension
   -- local outputDimension = self.outputDimension
   -- local offset = 1
   -- local runModule = function(inputTensor, gradOutputTensor, i, module)
   --    local currentInput = inputTensor:select(inputDimension,i)
   --    local currentOutput = module.output
   --    local outputSize = currentOutput:size(outputDimension)
   --    local currentGradOutput = gradOutputTensor:narrow(outputDimension, offset, outputSize)

   --    return module:updateGradInput(currentInput, currentGradOutput)
   -- end

   -- local gradInputs = self.ll:pmap_mut(self.modules, moses.bind(moses.bind(runModule, input), gradOutput))
   -- print(self.modules)

   for i=1,nModule do
      self.modules[i] = updatedModules[i]
      local currentGradInput = updatedModules[i].gradInput
      local currentOutput = updatedModules[i].output
      local outputSize = currentOutput:size(self.outputDimension)
      -- print(self.gradInput)
      -- print(currentGradInput)
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

      mod:accGradParameters(
          input:select(self.inputDimension,i),
          gradOutput:narrow(self.outputDimension, offset,outputSize),
          scale
      )

      offset = offset + outputSize
   end
end

function ParallelParallel:accUpdateGradParameters(input, gradOutput, lr)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do
      local module = self.modules[i];
      local currentOutput = module.output
      module:accUpdateGradParameters(

          input:select(self.inputDimension,i),
          gradOutput:narrow(self.outputDimension, offset,
                            currentOutput:size(self.outputDimension)),
          lr)

      offset = offset + currentOutput:size(self.outputDimension)
   end
end

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