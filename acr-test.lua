require 'nn'
dofile 'INTM.lua'
dofile 'ACR.lua'

intm = nn.INTM(7,10)
pose = intm:forward({0,0,1,1,0,0*math.pi/4,  1})[{{1,9}}]
pose = pose:reshape(3,3)
print(pose)

acr = nn.ACR(15)
template = torch.ones(3,3)
print(acr:forward({template, pose}))

