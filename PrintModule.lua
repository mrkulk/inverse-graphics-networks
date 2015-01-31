-- require 'nn'
local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function PrintModule:__init(name)
  self.name = name
end

function PrintModule:updateOutput(input)
  print(self.name.." input: ")
  print(input:sum())
  self.output = input
  return input
end

function PrintModule:updateGradInput(input, gradOutput)
  print(self.name.." gradInput:")
  print(gradOutput:sum())
  self.gradInput = gradOutput
  return self.gradInput
end

