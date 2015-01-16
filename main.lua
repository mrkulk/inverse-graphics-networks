-- Unsupervised Capsule Deep Network
-- require('mobdebug').start()

require 'nn'
require 'nngraph'
require 'image'
require 'dataset-mnist'

--number of acr's
num_acrs = 2
imsize = 10
h1size = 20
outsize = 7*num_acrs --affine variables

architecture = nn.Sequential()
architecture:add(nn.Linear(imsize,h1size))
architecture:add(nn.Tanh())
architecture:add(nn.Linear(h1size, outsize))

architecture:add(nn.Reshape(num_acrs,7))
architecture:add(nn.SplitTable(1))

architecture:add(nn.ParallelTable():add(nn.Linear()):add())

print(architecture:forward(torch.rand(imsize))[1])
  
--input = nn.Identity()()

