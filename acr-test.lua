
require 'nn'
dofile 'INTM.lua'
dofile 'ACR.lua'
dofile 'Bias.lua'

require 'optim'

--torch.manualSeed(1)

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

data = torch.Tensor({-0.50,-1.51,.4,1.4,0.2,1*math.pi/4,  0.3})
data = data:reshape(1,7)
targets = torch.zeros(imwidth,imwidth):reshape(1,imwidth,imwidth)
targets = targets * 0.5

outputs = net:forward(data):clone()

shiftedOutputs = net:forward(torch.Tensor({-2.0,-2.0,1,1,0,0*math.pi/4,  1}):reshape(1,7))
targets[{{}, {1,7}, {1,7}}] = shiftedOutputs[{{}, {1,7}, {1,7}}]

-- targets[{{}, {2,7}, {2,7}}] = outputs[{{}, {1,6}, {1,6}}]
f = criterion:forward(outputs, targets)



df_do = criterion:backward(outputs, targets)

net:forward(data)
back = net:backward(data, df_do)

--
--------------------------------------------------------------------------
-------------------test gradients for template----------------------------
--------------------------------------------------------------------------
if true then
	-- print('pred:',outputs)
	-- print('target:', targets)

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
	print('True', ac_bias_truegrad:reshape(twidth, twidth))
	print('Calc', ac_bias.gradBias:reshape(twidth, twidth))
	grad_diff = grad_diff + diff


	print('[GRADIENT CHECKER: TEMPLATE] Error: ', grad_diff)

else
--------------------------------------------------------------------------
-------------------test gradients for pose--------------------------------
--------------------------------------------------------------------------
	print('pred:',outputs)
	print('target:', targets)

	EPSILON = 1e-4
	grad_diff = 0
	intm = acr_in.modules[2]
	intm_calcgrads = intm.gradInput:clone()
	intm_truegrad = torch.zeros(intm.gradInput:size())

	for ii = 1, intm_truegrad:size()[2] do
		data[{1,ii}] = data[{1,ii}] + EPSILON
		J_pos = criterion:forward(net:forward(data), targets)

		data[{1,ii}] = data[{1,ii}] - 2*EPSILON
		J_neg = criterion:forward(net:forward(data), targets)

		intm_truegrad[{1,ii}] = (J_pos - J_neg)/(2*EPSILON)

		data[{1,ii}] = data[{1,ii}] + EPSILON
	end
	local diff = torch.sum(torch.pow(intm_truegrad - intm_calcgrads,2))
	print('intm gradient error:', diff)
	print('True', intm_truegrad)
	print('Calc', intm_calcgrads)
end