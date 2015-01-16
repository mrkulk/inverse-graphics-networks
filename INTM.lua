-- INTM module
local INTM, Parent = torch.class('nn.INTM', 'nn.Module')
 
function INTM:__init(input_dim, output_dim)
   Parent.__init(self)
   self.input_dim = input_dim
   self.output_dim = output_dim
end
 
function INTM:updateOutput(input)
  self.output = input[{{1,self.output_dim}}]*10
  return self.output
end
 
function INTM:updateGradInput(input, gradOutput)
  a=1
end
 