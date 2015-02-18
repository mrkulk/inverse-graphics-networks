local INTMReg, parent = torch.class('nn.INTMReg', 'nn.Module')

function INTMReg:__init()
  self.regStrength = 1e2
end

function INTMReg:updateOutput(input)
  self.output = input
  return input
end

function INTMReg:updateGradInput(input, gradOutput)
  local geoPose = input[{{}, {1,9}}]
  local intensity = input[{{}, 10}]

  local optimalDeterminant = 0.1182
  -- local optimalDeterminant = 1

  local A11 = geoPose[{{}, 1}]
  local A12 = geoPose[{{}, 2}]
  local A13 = geoPose[{{}, 3}]
  local A21 = geoPose[{{}, 4}]
  local A22 = geoPose[{{}, 5}]
  local A23 = geoPose[{{}, 6}]
  local A31 = geoPose[{{}, 7}]
  local A32 = geoPose[{{}, 8}]
  local A33 = geoPose[{{}, 9}]

  local determinant =  torch.cmul( A11, (torch.cmul(A22, A33) - torch.cmul(A23, A32)) )
                     - torch.cmul( A12, (torch.cmul(A21, A33) - torch.cmul(A23, A31)) )
                     + torch.cmul( A13, (torch.cmul(A21, A32) - torch.cmul(A22, A31)) )

  -- local determinant = A11*(A22*A33−A23*A32)
  --                    −A12*(A21*A33−A23*A31)
  --                    +A13*(A21*A32−A22*A31)

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


  local diff = (determinant - optimalDeterminant) * -1
  local indices = {}
  for i = 1, diff:size()[1] do
    if diff[i] > 0 then
      table.insert(indices, i)
    end
  end

  local gradDetInput = torch.zeros(input:size()[1], 10)

  if #indices ~= 0 then
    gradDetInput[{indices, 1}] = torch.cmul(geoPose[{indices, 5}], geoPose[{indices, 9}])
                              - torch.cmul(geoPose[{indices, 6}], geoPose[{indices, 8}])
    gradDetInput[{indices, 2}] = torch.cmul(geoPose[{indices, 6}], geoPose[{indices, 7}])
                              - torch.cmul(geoPose[{indices, 4}], geoPose[{indices, 9}])
    gradDetInput[{indices, 3}] = torch.cmul(geoPose[{indices, 4}], geoPose[{indices, 8}])
                              - torch.cmul(geoPose[{indices, 5}], geoPose[{indices, 7}])

    gradDetInput[{indices, 4}] = torch.cmul(geoPose[{indices, 3}], geoPose[{indices, 8}])
                              - torch.cmul(geoPose[{indices, 2}], geoPose[{indices, 9}])
    gradDetInput[{indices, 5}] = torch.cmul(geoPose[{indices, 1}], geoPose[{indices, 9}])
                              - torch.cmul(geoPose[{indices, 3}], geoPose[{indices, 7}])
    gradDetInput[{indices, 6}] = torch.cmul(geoPose[{indices, 2}], geoPose[{indices, 7}])
                              - torch.cmul(geoPose[{indices, 1}], geoPose[{indices, 8}])

    gradDetInput[{indices, 7}] = torch.cmul(geoPose[{indices, 2}], geoPose[{indices, 6}])
                              - torch.cmul(geoPose[{indices, 3}], geoPose[{indices, 5}])
    gradDetInput[{indices, 8}] = torch.cmul(geoPose[{indices, 3}], geoPose[{indices, 8}])
                              - torch.cmul(geoPose[{indices, 1}], geoPose[{indices, 6}])
    gradDetInput[{indices, 9}] = torch.cmul(geoPose[{indices, 1}], geoPose[{indices, 5}])
                              - torch.cmul(geoPose[{indices, 2}], geoPose[{indices, 4}])
  end

  self.gradInput = gradOutput - gradDetInput * self.regStrength
  print("intm-REG determinant", determinant[1])
  print("intm-REG reg amount", (gradDetInput * self.regStrength):sum())

  return self.gradInput
end







