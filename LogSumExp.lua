local LogSumExp, parent = torch.class('nn.LogSumExp', 'nn.Module')

function LogSumExp:__init()
  parent.__init(self)
end

function LogSumExp:updateOutput(input)
  self.output = input:clone()
  self.max_val = torch.max(self.output)

  print("MAXVAL:", self.max_val,  ' MINVAL:', torch.min(input))

  self.output = self.output - self.max_val
  self.output = torch.log(torch.sum(torch.exp(self.output), 2))

  self.output = self.output + self.max_val
  return self.output
end

function LogSumExp:updateGradInput(input, gradOutput)
  -- print("MAXVAL:", self.max_val,  ' MINVAL:', torch.min(input))
  self.gradInput = torch.Tensor(input:size())
  local t1 = input - self.max_val
  local normalization = torch.sum(t1, 2)
  for i=1, input:size()[2] do
    local t2 = torch.reshape(torch.exp(t1[{{}, i, {}, {}}]), input:size()[1], 1, input:size()[3], input:size()[4])
    self.gradInput[{{}, i, {}, {}}] = torch.cdiv(t2, normalization)
    self.gradInput[{{}, i, {}, {}}] = torch.cmul(self.gradInput[{{}, i, {}, {}}], gradOutput)
  end

  return self.gradInput
end

