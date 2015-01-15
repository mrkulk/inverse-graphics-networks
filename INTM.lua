-- INTM module
local INTM, Parent = torch.class('nn.INTM', 'nn.Module')
 
function INTM:__init()
   parent.__init(self)
end
 
function INTM:updateOutput(input)
  self.output = input
  return self.output
end
 
function INTM:updateGradInput(input, gradOutput)
  
end
 