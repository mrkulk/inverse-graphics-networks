require 'nn'
require 'ACR'

batch_size = 1
image_width = 10
template_width = 3

acr = nn.ACR(batch_size, image_width)


randomize = false

if randomize then
  iGeoPose = torch.rand(1, 10)
else
  torch.manualSeed(1)
  t1 = -8.51
  t2 = -8.49
  s1 = 1
  s2 = 1
  z  = 0.0
  theta = math.pi/8 * 0

  cos = math.cos
  sin = math.sin

  iGeoPose = torch.Tensor({
             s1*cos(theta), -s2*(sin(theta) - z*cos(theta)), s1*t1*cos(theta) - s2*t2*(sin(theta) - z*cos(theta)),
             s1*sin(theta),  s2*(cos(theta) + z*sin(theta)), s1*t1*sin(theta) + s2*t2*(cos(theta) + z*sin(theta)),
             0,              0,                              1,
        1})

  -- iGeoPose = torch.Tensor({ 1,  0.5, -7,
  --                           0,  1, -3,
  --                           0,  0,  1, 1}):reshape(1, 10)

  iGeoPose = iGeoPose:reshape(1, 10)
end

template = torch.rand(batch_size, template_width ^ 2)

input = {template, iGeoPose}

function loss(acrOutput)
  return acrOutput:sum()
end

function gradLoss(acrOutput)
  return torch.ones(acrOutput:size())
end

output = acr:forward(input):clone()
acrForwardLoss = loss(output)

print(output)
acrGradInput = acr:updateGradInput(input, gradLoss(output))

function finiteDiff()
  epsilon = 1e-3

  gradTemplate = torch.zeros(template:size())
  for i = 1, gradTemplate:size()[2] do
    tempTemplate = template:clone()

    tempTemplate[1][i] = template[1][i] - epsilon
    outputNeg = acr:forward({tempTemplate, iGeoPose})
    lossNeg = loss(outputNeg)

    tempTemplate[1][i] = template[1][i] + epsilon
    outputPos = acr:forward({tempTemplate, iGeoPose})
    lossPos = loss(outputPos)

    finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
    gradTemplate[1][i] = finiteDiffGrad
  end

  gradGeoPose = torch.zeros(iGeoPose:size())
  for i = 1, gradGeoPose:size()[2] do
    tempGeoPose = iGeoPose:clone()

    tempGeoPose[1][i] = iGeoPose[1][i] - epsilon
    outputNeg = acr:forward({template, tempGeoPose})
    lossNeg = loss(outputNeg)

    tempGeoPose[1][i] = iGeoPose[1][i] + epsilon
    outputPos = acr:forward({template, tempGeoPose})
    lossPos = loss(outputPos)

    finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
    gradGeoPose[1][i] = finiteDiffGrad
  end

  return {gradTemplate, gradGeoPose}
end

fd = finiteDiff()

-- print("ACR gradTemplate")
-- print(acrGradInput[1])
-- print("true gradTemplate")
-- print(fd[1])

print("ACR gradGeoPose")
print(acrGradInput[2])
print("true gradGeoPose")
print(fd[2])

print("Pose gradient error:", torch.norm(acrGradInput[2] - fd[2]))











