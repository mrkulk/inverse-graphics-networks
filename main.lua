-- Unsupervised Capsule Deep Network
-- require('mobdebug').start()

require 'nn'
require 'nngraph'
require 'image'
require 'dataset-mnist'
require 'INTM'
require 'math'
require 'torch'

--number of acr's
num_acrs = 2
imsize = 12
h1size = 20
outsize = 7*num_acrs --affine variables
intm_out_dim = 10

architecture = nn.Sequential()
architecture:add(nn.Linear(imsize,h1size))
architecture:add(nn.Tanh())
architecture:add(nn.Linear(h1size, outsize))

architecture:add(nn.Reshape(num_acrs,7))
architecture:add(nn.SplitTable(1))

-- Creating intm and acr's
decoder = nn.ParallelTable()
for ii=1,num_acrs do
  local t = nn.Sequential()
    t:add(nn.INTM(7,intm_out_dim)) --intm
    t:add(nn.Linear(intm_out_dim,imsize)) --acr [TODO]
  decoder:add(t)
end
architecture:add(decoder)
--TODO: Add OM and MSE Criterion
  
print(architecture:forward(torch.rand(imsize))[1])
print('Backward ...')
print(architecture:backward(torch.rand(imsize), torch.rand(2,imsize)))  

