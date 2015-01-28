require 'nn'
require 'ParallelParallel'

pp = nn.ParallelParallel(1, 1)
-- pp = nn.Parallel(1, 1)
for i = 1,4 do
  seq = nn.Sequential()
  seq:add(nn.Linear(10,9))
  seq:add(nn.Tanh())
  seq:add(nn.Reshape(3,3))
  pp:add(seq)
end

input = torch.rand(4,10)
target = torch.rand(3,12)
criterion = nn.MSECriterion()

-- print(criterion:forward(pp:forward(input), target))
-- pp:backward(input, criterion:backward(pp.output, target))
for i=1,1000 do
  -- print(pp:forward(input))
  print(criterion:forward(pp:forward(input), target))
  pp:zeroGradParameters()
  pp:backward(input, criterion:backward(pp.output, target))
  pp:updateParameters(0.01)
end
