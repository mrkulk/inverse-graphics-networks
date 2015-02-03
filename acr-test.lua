require 'nn'
dofile 'INTM.lua'
dofile 'ACR.lua'
dofile 'Bias.lua'

require 'optim'

torch.manualSeed(1)

--[[
intm = nn.INTM(1,7,10)
enc = torch.Tensor({0,0,1,1,0,0*math.pi/4,  1})
enc = enc:reshape(1,7)
print(enc:size())
pose = intm:forward(enc)
print(pose)
]]
twidth = 3
imwidth = 10
acr = nn.ACR(1,imwidth)
intm = nn.INTM(1,7,10)
bias = nn.Bias(1, twidth*twidth)


net = nn.Sequential()
acr_wrapper = nn.Sequential()
acr_wrapper:add(nn.Replicate(2))
acr_wrapper:add(nn.SplitTable(1))
  
acr_in = nn.ParallelTable()
acr_in:add(bias)
acr_in:add(intm)

acr_wrapper:add(acr_in)
net:add(acr_wrapper)
net:add(acr)
--net:add(nn.Sigmoid())

criterion = nn.MSECriterion()

data = torch.Tensor({-1.5,-1.5,1,1,0,0*math.pi/4,  1})
data = data:reshape(1,7)
targets = torch.zeros(imwidth,imwidth):reshape(1,imwidth,imwidth)
targets = targets * 0.5

outputs = net:forward(data)
targets[{{}, {4,6}, {4,6}}] = outputs[{{}, {1,3}, {1,3}}] 
f = criterion:forward(outputs, targets)



df_do = criterion:backward(outputs, targets)
back = net:backward(data, df_do)

print('pred:',outputs)
print('target:', targets)

--test gradients for template
EPSILON = 1e-7
grad_diff = 0
ac_bias = acr_in.modules[1]

ac_bias_truegrad = torch.zeros(ac_bias.bias:size())

for ii = 1, ac_bias.bias:size()[2] do

	--local t = ac_bias.bias[{1, ii }]
	ac_bias.bias[{1,ii}] = ac_bias.bias[{1,ii}] + EPSILON
	J_pos = criterion:forward(net:forward(data), targets)

	ac_bias.bias[{1,ii}] = ac_bias.bias[{1,ii}] - 2*EPSILON
	J_neg = criterion:forward(net:forward(data), targets)
	
	ac_bias_truegrad[{1,ii}] = (J_pos - J_neg)/(2*EPSILON)

	--ac_bias.bias[{1,ii}] = t
	ac_bias.bias[{1,ii}] = ac_bias.bias[{1,ii}] + EPSILON
end
local diff = torch.sum(torch.pow(ac_bias_truegrad - ac_bias.gradBias,2))
print('ACR error:', diff)
print('True', ac_bias_truegrad)
print('Calc', ac_bias.gradBias)
grad_diff = grad_diff + diff


print('[GRADIENT CHECKER: TEMPLATE] Error: ', grad_diff)


