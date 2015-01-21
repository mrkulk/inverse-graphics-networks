-- INTM module
-- require('mobdebug').start()

local INTM, parent = torch.class('nn.INTM', 'nn.Module')

function INTM:__init(input_dim, output_dim)
   parent.__init(self)
   self.input_dim = input_dim
   self.output_dim = output_dim
   self.output = torch.zeros(output_dim)
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
  t1=input[1];t2=input[2];s1=input[3];s2=input[4];z=input[5];theta=input[6]
  self.output[1]=s1*math.cos(theta)
  self.output[2]=-s2*(math.sin(theta) - z*math.cos(theta))
  self.output[3]=s1*t1*math.cos(theta) - s2*t2*(math.sin(theta) - z*math.cos(theta))
  self.output[4]=s1*math.sin(theta)
  self.output[5]=s2*(math.cos(theta) + z*math.sin(theta))
  self.output[6]=s1*t1*math.sin(theta) + s2*t2*(math.cos(theta) + z*math.sin(theta))
  self.output[7]=0; self.output[8]=0;self.output[9]=1;
  self.output[10]=input[7]
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
  gradOutput_reshaped = gradOutput[{{1,9}}]:reshape(3,3) --last is intensity
  s1=input[2];s2=input[3];t1=input[4];t2=input[5];z=input[6];theta=input[7]

  grad_s1 = torch.zeros(3,3)
  grad_s1[1][1]=math.cos(theta); grad_s1[1][3]=t1*math.cos(theta)
  grad_s1[2][1]=math.sin(theta); grad_s1[1][2]=t1*math.sin(theta)

  grad_s2 = torch.zeros(3,3)
  grad_s2[1][2]=z*math.cos(theta)-math.sin(theta); grad_s2[1][2]=-t2*(math.sin(theta)-z*math.cos(theta))
  grad_s2[2][2]=math.cos(theta)+z*math.sin(theta); grad_s2[2][3]= t2*(math.cos(theta)+z*math.sin(theta))

  grad_t1 = torch.zeros(3,3)
  grad_t1[1][3]=s1*math.cos(theta); grad_t1[2][3]=s1*math.sin(theta)

  grad_t2 = torch.zeros(3,3)
  grad_t2[1][3]=-s2*(math.sin(theta) - z*math.cos(theta)); grad_t2[2][3]=s2*(math.cos(theta) + z*math.sin(theta))

  grad_z = torch.zeros(3,3)
  grad_z[1][2]=s2*math.cos(theta); grad_z[1][3]=s2*t2*math.cos(theta)
  grad_z[2][2]=s2*math.sin(theta); grad_z[2][3]=s2*t2*math.sin(theta)

  grad_theta = torch.zeros(3,3)
  grad_theta[1][1]=-s1*math.sin(theta); grad_theta[1][2]=-s2*(math.cos(theta) + z*math.sin(theta));
  grad_theta[1][3]=- s1*t1*math.sin(theta) - s2*t2*(math.cos(theta) + z*math.sin(theta))
  grad_theta[2][1]=s1*math.cos(theta); grad_theta[2][2]=-s2*(math.sin(theta) - z*math.cos(theta));
  grad_theta[2][3]=s1*t1*math.cos(theta) - s2*t2*(math.sin(theta) - z*math.cos(theta))

  self.gradInput = torch.zeros(self.input_dim)
  self.gradInput[1] = torch.sum(torch.cmul(gradOutput_reshaped, grad_s1))
  self.gradInput[2] = torch.sum(torch.cmul(gradOutput_reshaped, grad_s2))
  self.gradInput[3] = torch.sum(torch.cmul(gradOutput_reshaped, grad_t1))
  self.gradInput[4] = torch.sum(torch.cmul(gradOutput_reshaped, grad_t2))
  self.gradInput[5] = torch.sum(torch.cmul(gradOutput_reshaped, grad_z))
  self.gradInput[6] = torch.sum(torch.cmul(gradOutput_reshaped, grad_theta))
  self.gradInput[7] = gradOutput[10] -- intensity is unchanged in this module

  return self.gradInput
end
