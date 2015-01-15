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

-- encoder network
input = nn.Identity()()
h1 = nn.Tanh()(nn.Linear(imsize,h1size)(input))

-- decoder network
igeonArray = {}
for i=1,num_acrs do
  igeonArray[i] = nn.Linear(h1size,outsize)(h1)
end

--intm


graph=nn.gModule({input},{igeonArray[1]})

print(graph:forward(torch.rand(imsize)))



