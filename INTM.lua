-- INTM module
-- require('mobdebug').start()

local INTM, parent = torch.class('nn.INTM', 'nn.Module')

function INTM:__init(bsize, input_dim, output_dim)
   parent.__init(self)
   self.input_dim = input_dim
   self.output_dim = output_dim
   self.output = torch.zeros(bsize, output_dim)
   self.bsize = bsize
end


--[[
transform =
[ s1*cos(theta), -s2*(sin(theta) - z*cos(theta)), s1*t1*cos(theta) - s2*t2*(sin(theta) - z*cos(theta))]
[ s1*sin(theta),  s2*(cos(theta) + z*sin(theta)), s1*t1*sin(theta) + s2*t2*(cos(theta) + z*sin(theta))]
[             0,                               0,                                                    1]

input = [t1, t2, s1,s2, z, theta, intensity]
--]]

function INTM:updateOutput(input)
  
  --self.output = input[{{1,self.output_dim}}]*10
  -- print(input[{{},7}]:sum())
  t1=input[{{},1}]
  t2=input[{{},2}]
  s1=input[{{},3}]
  s2=input[{{},4}]
  z=input[{{},5}]
  theta=input[{{},6}]

  self.output[{{},1}]= torch.cmul(s1, torch.cos(theta)) --s1*math.cos(theta)
  self.output[{{},2}]= torch.cmul(s2, torch.cmul(z, torch.cos(theta)) - torch.sin(theta))  --s2*(math.sin(theta) - z*math.cos(theta))
  self.output[{{},3}]= torch.cmul(s1, torch.cmul(t1, torch.cos(theta))) - torch.cmul(s2,torch.cmul(t2,torch.sin(theta) - torch.cmul(z, torch.cos(theta))))  --s1*t1*math.cos(theta) - s2*t2*(math.sin(theta) - z*math.cos(theta))
  self.output[{{},4}]= torch.cmul(s1, torch.sin(theta)) --s1*math.sin(theta)
  self.output[{{},5}]= torch.cmul(s2, torch.cos(theta) +  torch.cmul(z, torch.sin(theta))) --s2*(math.cos(theta) + z*math.sin(theta))
  self.output[{{},6}]= torch.cmul(s1, torch.cmul(t1, torch.sin(theta))) + torch.cmul(s2, torch.cmul(t2, torch.cos(theta) + torch.cmul(z, torch.sin(theta)))) --s1*t1*math.sin(theta) + s2*t2*(math.cos(theta) + z*math.sin(theta))
  self.output[{{},7}]=0;
  self.output[{{},8}]=0;
  self.output[{{},9}]=1;
  self.output[{{},10}]=input[{{},7}]


  return self.output
end


--[[
d(transform)/ds1
[ cos(theta), 0, t1*cos(theta)]
[ sin(theta), 0, t1*sin(theta)]
[          0, 0,             0]

d(transform)/ds2
[ 0, z*cos(theta) - sin(theta), -t2*(sin(theta) - z*cos(theta))]
[ 0, cos(theta) + z*sin(theta),  t2*(cos(theta) + z*sin(theta))]
[ 0,                         0,                               0]

d(transform)/dt1
[ 0, 0, s1*cos(theta)]
[ 0, 0, s1*sin(theta)]
[ 0, 0,             0]

d(transform)/dt2
[ 0, 0, -s2*(sin(theta) - z*cos(theta))]
[ 0, 0,  s2*(cos(theta) + z*sin(theta))]
[ 0, 0,                               0]

d(transform)/dz
[ 0, s2*cos(theta), s2*t2*cos(theta)]
[ 0, s2*sin(theta), s2*t2*sin(theta)]
[ 0,             0,                0]

d(transform)/dtheta
[ -s1*sin(theta), -s2*(cos(theta) + z*sin(theta)), - s1*t1*sin(theta) - s2*t2*(cos(theta) + z*sin(theta))]
[  s1*cos(theta), -s2*(sin(theta) - z*cos(theta)),   s1*t1*cos(theta) - s2*t2*(sin(theta) - z*cos(theta))]
[              0,                               0,                                                      0]

--]]
function INTM:updateGradInput(input, gradOutput)
  bsize = self.bsize
  --gradOutput_reshaped = gradOutput[{{1,9}}]:reshape(3,3) --last is intensity
  local gradOutput_reshaped = torch.zeros(bsize, 3, 3)
  for i=1, bsize do
    gradOutput_reshaped[{i,{},{}}] = gradOutput[{i,{1,9}}]:reshape(3,3) --last is intensity
  end

  t1=input[{{},1}]
  t2=input[{{},2}]
  s1=input[{{},3}]
  s2=input[{{},4}]
  z=input[{{},5}]
  theta=input[{{},6}]

  grad_s1 = torch.zeros(bsize, 3,3)
  grad_s1[{{},1,1}]=torch.cos(theta); grad_s1[{{},1,3}]=torch.cmul(t1,torch.cos(theta))
  grad_s1[{{},2,1}]=torch.sin(theta); grad_s1[{{},2,3}]=torch.cmul(t1,torch.sin(theta))

  grad_s2 = torch.zeros(bsize, 3,3)
  grad_s2[{{},1,2}]= torch.cmul(z,torch.cos(theta))-torch.sin(theta); grad_s2[{{},1,3}]=torch.cmul(-t2,(torch.sin(theta)- torch.cmul(z,torch.cos(theta))))
  grad_s2[{{},2,2}]=torch.cos(theta)+ torch.cmul(z,torch.sin(theta)); grad_s2[{{},2,3}]= torch.cmul(t2,(torch.cos(theta)+torch.cmul(z,torch.sin(theta))))

  grad_t1 = torch.zeros(bsize, 3,3)
  grad_t1[{{},1,3}]=torch.cmul(s1,torch.cos(theta)); grad_t1[{{},2,3}]=torch.cmul(s1,torch.sin(theta))

  grad_t2 = torch.zeros(bsize, 3,3)
  grad_t2[{{},1,3}]=torch.cmul(-s2,(torch.sin(theta) - torch.cmul(z,torch.cos(theta)))); grad_t2[{{},2,3}]=torch.cmul(s2,(torch.cos(theta) + torch.cmul(z,torch.sin(theta))))

  grad_z = torch.zeros(bsize, 3,3)
  grad_z[{{},1,2}]=torch.cmul(s2,torch.cos(theta)); grad_z[{{},1,3}]=torch.cmul(s2,torch.cmul(t2,torch.cos(theta)))
  grad_z[{{},2,2}]=torch.cmul(s2,torch.sin(theta)); grad_z[{{},2,3}]=torch.cmul(s2,torch.cmul(t2,torch.sin(theta)))

  grad_theta = torch.zeros(bsize, 3,3)
  grad_theta[{{},1,1}] = torch.cmul(-s1,torch.sin(theta)); grad_theta[{{},1,2}]=torch.cmul(-s2,(torch.cos(theta) + torch.cmul(z,torch.sin(theta))));
  grad_theta[{{},1,3}] = torch.cmul(-s1,torch.cmul(t1,torch.sin(theta))) - torch.cmul(s2,torch.cmul(t2,(torch.cos(theta) + torch.cmul(z,torch.sin(theta)))))
  grad_theta[{{},2,1}] = torch.cmul(s1,torch.cos(theta)); grad_theta[{{},2,2}]=torch.cmul(-s2,(torch.sin(theta) - torch.cmul(z,torch.cos(theta))));
  grad_theta[{{},2,3}] = torch.cmul(s1,torch.cmul(t1,torch.cos(theta))) - torch.cmul(s2, torch.cmul(t2,(torch.sin(theta) - torch.cmul(z,torch.cos(theta)))))

  self.gradInput = torch.zeros(bsize, self.input_dim)

  for i=1,bsize do
    self.gradInput[i][1] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_t1[{i,{},{}}]))
    self.gradInput[i][2] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_t2[{i,{},{}}]))
    self.gradInput[i][3] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_s1[{i,{},{}}]))
    self.gradInput[i][4] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_s2[{i,{},{}}]))
    self.gradInput[i][5] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_z[{i,{},{}}]))
    self.gradInput[i][6] = torch.sum(torch.cmul(gradOutput_reshaped[{i,{},{}}], grad_theta[{i,{},{}}]))

    self.gradInput[i][7] = gradOutput[{i,10}] -- intensity is unchanged in this module
  end

  --print("intm gradinput", self.gradInput:sum())
  return self.gradInput
end
