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
outsize = 7 --affine variables

architecture = nn.Sequential()

-- encoder network
input = nn.Identity()()
h1 = nn.Linear(imsize,h1size)
h1_act = nn.Tanh(h1)
architecture:add(h1_act)


-- decoder network
intm_network = nn.Sequential()
igeonArray = {}

local t=nn.Sequential()
tmp = nn.Linear(h1size, outsize)(h1_act)

--[[
for i=1,num_acrs do
  local t=nn.Sequential()
  igeonArray[i] = nn.Linear(h1size,outsize)(h1)
  t:add(igeonArray[i])
  intm_network:add(t)
end
architecture:add(intm_network)
]]
--]]

--network=nn.gModule({input},{igeonArray[1]})

--print(network:forward(torch.rand(imsize)))
-- graph.dot(network.fg,'Big MLP')


