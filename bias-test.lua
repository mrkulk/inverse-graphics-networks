require 'nn'
dofile 'Bias.lua'
mlp = nn.Sequential()
b = nn.Bias(10)
mlp:add(b)
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01

dataset = {}
function dataset:size() return 100 end
for i=1, dataset:size() do
  local input = torch.Tensor()
  local output = torch.Tensor({1,6,7,3,2,9,2,4,10,-5})
  dataset[i] = {input, output}
end

trainer:train(dataset)
