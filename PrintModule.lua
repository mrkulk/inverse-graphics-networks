-- require 'nn'
local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function PrintModule:updateOutput(input)
  -- print(input)
  self.output = input
  return input
end

function PrintModule:updateGradInput(input, gradOutput)
  print(gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

