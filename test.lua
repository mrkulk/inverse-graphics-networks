require 'nn'
require 'nngraph'
require 'INTM'

require('mobdebug').start()

function intm_test()
  imsize=10; h1size=20; igeo_outsize=7;
  input = nn.Identity()()
  h1 = nn.Tanh()(nn.Linear(imsize,h1size)(input))
  
  igeopose = nn.INTM(h1size, igeo_outsize)(h1)
  cost = nn.Mean()(igeopose)
  graph=nn.gModule({input},{igeopose})
  
  print('Forward pass ... ')
  indata = torch.rand(imsize)
  print(graph:forward(indata))
  
  print('Backward pass ...')
  dummy_target = torch.rand(igeo_outsize)
  print(graph:backward(indata, dummy_target))
end


function basic_torch()
  x1=nn.Linear(20,20)()
  x2=nn.Linear(10,10)()
  m0=nn.Linear(20,1)(nn.Tanh()(x1))
  m1=nn.Linear(10,1)(nn.Tanh()(x2))
  madd=nn.CAddTable()({m0,m1})
  m2=nn.Sigmoid()(madd)
  m3=nn.Tanh()(madd)
  gmod = nn.gModule({x1,x2},{m2,m3})

  x = torch.rand(20)
  y = torch.rand(10)

  gmod:updateOutput({x,y})
  gmod:updateGradInput({x,y},{torch.rand(1),torch.rand(1)})

  out=gmod:forward({x,y})
  grad=gmod:backward({x,y},{torch.rand(1),torch.rand(1)})
  print(grad[2])
end


intm_test()