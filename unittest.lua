require 'nn'
require 'ACR'
require 'vis'

batch_size = 2
image_width = 10
template_width = 3

acr = nn.ACR(batch_size, image_width)


randomize = false

if randomize then
  iGeoPose = torch.rand(batch_size, 10)
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

  iGeoPose = torch.rand(batch_size, 10)

  for i=1,batch_size do
    tmp_iGeoPose = torch.Tensor({
               s1*cos(theta), -s2*(sin(theta) - z*cos(theta)), s1*t1*cos(theta) - s2*t2*(sin(theta) - z*cos(theta)),
               s1*sin(theta),  s2*(cos(theta) + z*sin(theta)), s1*t1*sin(theta) + s2*t2*(cos(theta) + z*sin(theta)),
               0,              0,                              1,
          1})
    iGeoPose[i] = tmp_iGeoPose
  end
  -- iGeoPose = torch.Tensor({ 1,  0.5, -7,
  --                           0,  1, -3,
  --                           0,  0,  1, 1}):reshape(1, 10)

  -- iGeoPose = iGeoPose:reshape(batch_size, 10)
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
  for batchIndex = 1, batch_size do
    for i = 1, gradTemplate:size()[2] do
      tempTemplate = template:clone()

      tempTemplate[batchIndex][i] = template[batchIndex][i] - epsilon
      outputNeg = acr:forward({tempTemplate, iGeoPose})
      lossNeg = loss(outputNeg)

      tempTemplate[batchIndex][i] = template[batchIndex][i] + epsilon
      outputPos = acr:forward({tempTemplate, iGeoPose})
      lossPos = loss(outputPos)

      finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
      gradTemplate[batchIndex][i] = finiteDiffGrad
    end
  end

  gradGeoPose = torch.zeros(iGeoPose:size())
  for batchIndex = 1, batch_size do
    for i = 1, gradGeoPose:size()[2] do
      tempGeoPose = iGeoPose:clone()

      tempGeoPose[batchIndex][i] = iGeoPose[batchIndex][i] - epsilon
      outputNeg = acr:forward({template, tempGeoPose})
      lossNeg = loss(outputNeg)

      tempGeoPose[batchIndex][i] = iGeoPose[batchIndex][i] + epsilon
      outputPos = acr:forward({template, tempGeoPose})
      lossPos = loss(outputPos)

      finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
      gradGeoPose[batchIndex][i] = finiteDiffGrad
    end
  end

  return {gradTemplate, gradGeoPose}
end

fd = finiteDiff()



for i = 1, batch_size do
  print("\n".. colors.HEADER .."BATCH "..i)
  print("ACR gradTemplate ↓")
  print(simplestr(acrGradInput[1][i]))
  print(simplestr(fd[1][i]))
  print("TRUE gradTemplate ↑")

  -- print("DIFF: "..colors.FAIL)
  print(colors.FAIL .. simplestr(fd[1][i] - acrGradInput[1][i]))

  print("\nACR gradPose ↓")
  print(simplestr(acrGradInput[2][i]))
  print(simplestr(fd[2][i]))
  print("TRUE gradPose ↑")

  print(colors.FAIL .. simplestr(fd[2][i] - acrGradInput[2][i]))
end



template_error = torch.norm(acrGradInput[1] - fd[1])
pose_error = torch.norm(acrGradInput[2] - fd[2])
print("\nTemplate gradient error: "..colors.FAIL .. tostring(template_error))
print("Pose gradient error:\t "..colors.FAIL .. tostring(pose_error))











