local Bias, parent = torch.class('nn.Bias', 'nn.Module')

function Bias:__init(bsize, outputSize)
  parent.__init(self)
  self.output = torch.Tensor(bsize, outputSize)
  self.bsize = bsize
  self.bias = torch.rand(bsize, outputSize)
  self.gradBias = torch.Tensor(bsize, outputSize)
end

function Bias:updateOutput(input)
  self.output:copy(self.bias)
  return self.output
end

function Bias:updateGradInput(input, gradOutput)
  self.gradInput = torch.zeros(input:size())
  return self.gradInput
end

function Bias:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  self.gradBias:add(scale, gradOutput)
end

function Bias:updateParameters(learningRate)
  self.bias:add(-learningRate, self.gradBias)
end

function Bias:accUpdateGradParameters(input, gradOutput, learningRate)
  local gradBias = self.gradBias
  self.gradBias = self.bias
  self:accGradParameters(input, gradOutput, -learningRate)
  self.gradBias = gradBias
end



