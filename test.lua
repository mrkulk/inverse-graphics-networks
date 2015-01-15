require 'nn'
require 'nngraph'
require 'INTM'

function intm_test()
  
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