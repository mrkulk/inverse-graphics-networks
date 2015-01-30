local INTMReg, parent = torch.class('nn.INTMReg', 'nn.Module')

function INTMReg:__init()
  self.learningRate = 0
end

function INTMReg:updateOutput(input)
  self.output = input
  return input
end

function INTMReg:updateGradInput(input, gradOutput)
  local geoPose = input[{{}, {1,9}}]
  local intensity = input[{{}, 10}]

  --[[
  This calculates the gradient of the determinant of the matrix
    ((a b c)
     (d e f)
     (g h i))
  , which is
    ((ei - fh  fg - di  dh - eg)
     (ch - bi  ai - cg  bg - ah)
     (bf - ce  ch - af  ae - bd))
--]]

  local gradDetInput = torch.Tensor(input:size()[1], 10)
  gradDetInput[{{}, 1}] = geoPose[{{}, 5}] * geoPose[{{}, 9}] - geoPose[{{}, 6}] * geoPose[{{}, 8}]
  gradDetInput[{{}, 2}] = geoPose[{{}, 6}] * geoPose[{{}, 7}] - geoPose[{{}, 4}] * geoPose[{{}, 9}]
  gradDetInput[{{}, 3}] = geoPose[{{}, 4}] * geoPose[{{}, 8}] - geoPose[{{}, 5}] * geoPose[{{}, 7}]

  gradDetInput[{{}, 4}] = geoPose[{{}, 3}] * geoPose[{{}, 8}] - geoPose[{{}, 2}] * geoPose[{{}, 9}]
  gradDetInput[{{}, 5}] = geoPose[{{}, 1}] * geoPose[{{}, 9}] - geoPose[{{}, 3}] * geoPose[{{}, 7}]
  gradDetInput[{{}, 6}] = geoPose[{{}, 2}] * geoPose[{{}, 7}] - geoPose[{{}, 1}] * geoPose[{{}, 8}]

  gradDetInput[{{}, 7}] = geoPose[{{}, 2}] * geoPose[{{}, 6}] - geoPose[{{}, 3}] * geoPose[{{}, 5}]
  gradDetInput[{{}, 8}] = geoPose[{{}, 3}] * geoPose[{{}, 8}] - geoPose[{{}, 1}] * geoPose[{{}, 6}]
  gradDetInput[{{}, 9}] = geoPose[{{}, 1}] * geoPose[{{}, 5}] - geoPose[{{}, 2}] * geoPose[{{}, 4}]

  -- don't regularize the intensity
  gradDetInput[{{}, 10}] = 0

  self.gradInput = gradOutput + gradDetInput * self.learningRate
  return self.gradInput
end
