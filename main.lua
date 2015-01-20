-- Unsupervised Capsule Deep Network
-- require('mobdebug').start()

require 'nn'
require 'nngraph'
require 'image'
require 'dataset-mnist'
require 'INTM'
require 'math'
require 'torch'
require 'Bias'
require 'ACR'

--number of acr's
num_acrs = 9
image_width = 28
h1size = 2000
outsize = 7*num_acrs --affine variables
intm_out_dim = 10

architecture = nn.Sequential()
architecture:add(nn.Linear(image_width * image_width,h1size))
architecture:add(nn.Tanh())
architecture:add(nn.Linear(h1size, outsize))

architecture:add(nn.Reshape(num_acrs,7))
--architecture:add(nn.SplitTable(1))

-- Creating intm and acr's
decoder = nn.Parallel(1,2)
for ii=1,num_acrs do
  local acr_wrapper = nn.Sequential()
    acr_wrapper:add(nn.Replicate(2))

    acr_wrapper:add(nn.SplitTable(1))

    local acr_in = nn.ParallelTable()
    acr_in:add(nn.Bias(11*11))
    acr_in:add(nn.INTM(7,intm_out_dim))
  acr_wrapper:add(acr_in)
  acr_wrapper:add(nn.ACR(image_width)) --]]
  decoder:add(acr_wrapper)
end

architecture:add(decoder)
architecture:add(nn.Reshape(num_acrs, image_width,image_width))


architecture:add(nn.Mul(100))
architecture:add(nn.Exp())
architecture:add(nn.Sum(1))
architecture:add(nn.Log())
architecture:add(nn.Mul(1/100))

for i = 1, 10 do
  print "Forward ..."
  architecture:forward(torch.rand(image_width * image_width))
  -- print(res)

  print('Backward ...')
  architecture:backward(torch.rand(image_width * image_width), torch.rand(image_width ,image_width))
end
