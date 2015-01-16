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

-- Creating intm and acr's
decoder = nn.ParallelTable()
for ii=1,num_acrs do
  decoder:add(nn.Linear(7,10))
end

--architecture:add(nn.ParallelTable():add(nn.Linear(7,10)):add(nn.Linear(7,10)))

print(architecture:forward(torch.rand(imsize))[2])
  
--input = nn.Identity()()

