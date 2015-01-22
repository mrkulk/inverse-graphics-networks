require 'nn'
LL = require 'LL'
moses = require 'moses'

local ParallelParallel, parent = torch.class('nn.ParallelParallel', 'nn.Container')

function ParallelParallel:__init(nThreads, inputDimension,outputDimension)
   parent.__init(self)
   self.modules = {}
   self.size = torch.LongStorage()
   self.inputDimension = inputDimension
   self.outputDimension = outputDimension

   self.ll = LL:new(nThreads, {'nn', 'ParallelParallel', 'Bias', 'ACR', 'INTM'})
end

function ParallelParallel:updateOutput(input)
   local nModule=input:size(self.inputDimension)

   local runModule = function(inputTensor, i, module)
      local currentInput = inputTensor:select(self.inputDimension,i)
      return module:updateOutput(currentInput)
   end
   local outputs = self.ll:pmap(self.modules, moses.bind(runModule, input))

   for i = 1, #outputs do
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
   for i=1,nModule do
      local module=self.modules[i]
      local currentInput = input:select(self.inputDimension,i)
      local currentOutput = module.output
      local outputSize = currentOutput:size(self.outputDimension)
      local currentGradOutput = gradOutput:narrow(self.outputDimension, offset, outputSize)

      local currentGradInput = module:updateGradInput(currentInput, currentGradOutput)

      self.gradInput:select(self.inputDimension,i):copy(currentGradInput)
      offset = offset + outputSize
   end
   return self.gradInput
end

function ParallelParallel:accGradParameters(input, gradOutput, scale)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do
      local module = self.modules[i]
      local currentOutput = module.output
      local outputSize = currentOutput:size(self.outputDimension)

      module:accGradParameters(
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