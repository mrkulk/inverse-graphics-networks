-- Unsupervised Capsule Deep Network

require 'torch'
require 'image'
require 'math'
require 'parallel'
require 'nn'

require 'dataset-mnist'
require 'INTM'
require 'Bias'
require 'ACR'
require 'ParallelParallel'
require 'PrintModule'
require 'INTMReg'
require 'checkgradients'

torch.manualSeed(1)

CHECK_GRADS = false

function getDigitSet(digit)
  trainData = mnist.loadTrainSet(60000, {32,32})
  trainData:normalizeGlobal()
  local digitSet = {_indices = {}, _raw = trainData}
  for i = 1, trainData:size() do
    if trainData[i][2][digit + 1] == 1 then
      table.insert(digitSet._indices, i)
    end
  end
  setmetatable(digitSet, {__index = function (tbl,  key)
      if type(key) == 'table' then
        local nElems = key[1][2] - key[1][1] + 1
        set = torch.Tensor(nElems, tbl._raw[1][1][1]:size()[1], tbl._raw[1][1][1]:size()[2])
        for i = 0, nElems - 1 do
          set[{i + 1}] = tbl._raw[tbl._indices[key[1][1] + i]][1][1]
        end
        return set
      elseif type(key) == 'number' then
        return tbl._raw[tbl._indices[key]][1][1]
      end
    end})
  function digitSet:size() return #digitSet._indices end
  return digitSet
end

trainset = getDigitSet(1)

--number of acr's
num_acrs = 9
image_width = 32
h1size = 500
outsize = 7*num_acrs --affine variables
intm_out_dim = 10
bsize = 30
template_width = 11

architecture = nn.Sequential()
encoder = nn.Sequential()
  encoder:add(nn.Reshape(image_width * image_width))
  encoder:add(nn.Linear(image_width * image_width,h1size))
  -- encoder:add(nn.PrintModule())
  encoder:add(nn.Tanh())
  encoder:add(nn.Linear(h1size, outsize))
architecture:add(encoder)

architecture:add(nn.Reshape(num_acrs,7))

-- Creating intm and acr's
decoder = nn.Parallel(2,2)
-- decoder = nn.ParallelParallel(2,2)
for ii=1,num_acrs do
  local acr_wrapper = nn.Sequential()
  acr_wrapper:add(nn.Replicate(2))

  acr_wrapper:add(nn.SplitTable(1))

  local acr_in = nn.ParallelTable()
  local biasWrapper = nn.Sequential()
    biasWrapper:add(nn.Bias(bsize, template_width*template_width))
    --biasWrapper:add(nn.PrintModule("PostBias"))
    biasWrapper:add(nn.Exp())
    biasWrapper:add(nn.AddConstant(1))
    biasWrapper:add(nn.Log())
  acr_in:add(biasWrapper)

  local INTMWrapper = nn.Sequential()
    local splitter = nn.Parallel(2,2)
      for i = 1,6 do
        splitter:add(nn.Reshape(bsize, 1))
      end
      local intensityWrapper = nn.Sequential()
        intensityWrapper:add(nn.Exp())
        intensityWrapper:add(nn.AddConstant(1))
        intensityWrapper:add(nn.Log())
        intensityWrapper:add(nn.Reshape(bsize, 1))
      splitter:add(intensityWrapper)
    INTMWrapper:add(splitter)
    INTMWrapper:add(nn.INTM(bsize, 7,intm_out_dim))
    INTMWrapper:add(nn.INTMReg())
  acr_in:add(INTMWrapper)
  acr_wrapper:add(acr_in)
  acr_wrapper:add(nn.ACR(bsize, image_width))

  decoder:add(acr_wrapper)
end

architecture:add(decoder)
architecture:add(nn.Reshape(num_acrs, image_width,image_width))


architecture:add(nn.MulConstant(100))
architecture:add(nn.Exp())
architecture:add(nn.Sum(2))
-- architecture:add(nn.PrintModule("before log"))
architecture:add(nn.Log())
-- architecture:add(nn.PrintModule("after log"))

architecture:add(nn.MulConstant(1/100))


criterion = nn.MSECriterion()


function test_network()
  for i = 1, 10 do
    print "Forward ..."
    res=architecture:forward(torch.rand(bsize, image_width * image_width))
    -- print(res)

    print('Backward ...')
    architecture:backward(torch.rand(bsize, image_width * image_width), torch.rand(bsize, image_width , image_width))
  end
end



function saveACRs(step, model)
  -- acrs = model:findModules('nn.ACR')
  acrs = model.modules[3].modules
  padding = 5
  acrs_across = math.ceil(math.sqrt(#acrs))
  acrs_down = math.ceil(#acrs / acrs_across)

  acr_output = torch.zeros(
                    image_width * acrs_down + (acrs_down - 1) * padding,
                    image_width * acrs_across + (acrs_across - 1) * padding)
  for j, acr in ipairs(acrs) do
    x_index = math.floor((j - 1) / acrs_down)
    x_location = (x_index) * image_width + x_index * padding
    y_index = math.floor((j - 1) % acrs_down)
    y_location = (y_index) * image_width + y_index * padding
    acr_output[{{x_location + 1, x_location + image_width},
                {y_location + 1, y_location + image_width}}] = acr.output[1]
  end
  acr_output = acr_output:reshape(1, acr_output:size()[1], acr_output:size()[2])
  image.save("test_images/step_"..step.."_acrs.png", acr_output)
end


-- local learning_rate = 0.000001
local meta_learning_alpha = 0.00001
local gamma = math.exp(3)
local momentum = 0.95

-- local temp = {}
-- for ac=1,num_acrs do
--   temp[ac] = torch.zeros(template_width*template_width)
-- end
local gradAverage = {
  templates = torch.ones(num_acrs),
  encoderHidden = 1,
  encoderOutput = 1,
  encoderHiddenBias = 1,
  encoderOutputBias = 1
}

function train(epc)
  total_recon_error = 0
  for i = 1,1 do--1, trainset:size() do
    batch = trainset[{{i * bsize, (i + 1) * bsize - 1}}]
    recon_error = criterion:forward(architecture:forward(batch), batch)
    total_recon_error = total_recon_error + recon_error

    print("epoch:" .. tostring(epc) .. " batch:"..i.."/" .. tostring(trainset:size()) .. " error: " .. recon_error )
    --print(architecture:forward(batch))
    -- print(architecture.output)


    if i % 1 == 0 then
      local out = torch.reshape(architecture.output[1], 1,image_width,image_width)
      -- saveACRs(i, architecture)
      -- image.save("test_images/step_"..i.."_recon.png", out)
      --image.save("test_images/step_"..i.."_truth.png", batch[1])
    end

    architecture:zeroGradParameters()
    architecture:backward(batch, criterion:backward(architecture.output, batch))

    RMSPROP = true

    if RMSPROP == true then
      --------calculating gradient averages---------------
      local encoder_hidden = architecture.modules[1].modules[2]
      gradAverage.encoderHidden = 1/gamma * torch.pow(encoder_hidden.gradWeight:norm(),2)
                                  + (1-(1/gamma))*gradAverage.encoderHidden
      gradAverage.encoderHiddenBias = 1/gamma * torch.pow(encoder_hidden.gradBias:norm(),2)
                                  + (1-(1/gamma))*gradAverage.encoderHiddenBias

      local encoder_output = architecture.modules[1].modules[4]
      gradAverage.encoderOutput = 1/gamma * torch.pow(encoder_output.gradWeight:norm(),2)
                                  + (1-(1/gamma))*gradAverage.encoderOutput
      gradAverage.encoderOutputBias = 1/gamma * torch.pow(encoder_output.gradBias:norm(),2)
                                  + (1-(1/gamma))*gradAverage.encoderOutputBias

      for ac = 1,num_acrs do
        local ac_bias = architecture.modules[3].modules[ac].modules[3].modules[1].modules[1]
        gradAverage.templates[ac] = 1/gamma * torch.pow(ac_bias.gradBias:norm(),2)
                                  + (1-(1/gamma))*gradAverage.templates[ac]
      end

      --------update parameters --------------------------
      encoder_hidden.weight = encoder_hidden.weight
              - (encoder_hidden.gradWeight * meta_learning_alpha / gradAverage.encoderHidden)
      --print('encoder_hidden weight', torch.sum(encoder_hidden.gradWeight * meta_learning_alpha / gradAverage.encoderHidden))


      encoder_hidden.bias = encoder_hidden.bias
              - (encoder_hidden.gradBias * meta_learning_alpha / gradAverage.encoderHiddenBias)
      --print('encoder_hidden bias', torch.sum(encoder_hidden.gradBias * meta_learning_alpha / gradAverage.encoderHiddenBias))

      encoder_output.weight = encoder_output.weight
              - (encoder_output.gradWeight * meta_learning_alpha / gradAverage.encoderOutput)
      --print('encoder_output weight', torch.sum(encoder_output.gradWeight * meta_learning_alpha / gradAverage.encoderOutput))

      encoder_output.bias = encoder_output.bias
              - (encoder_output.gradBias * meta_learning_alpha / gradAverage.encoderOutputBias)
      --print('encoder_output bias', torch.sum(encoder_output.gradBias * meta_learning_alpha / gradAverage.encoderOutputBias))

      for ac=1,num_acrs do
        local ac_bias = architecture.modules[3].modules[ac].modules[3].modules[1].modules[1]
        ac_bias.bias = ac_bias.bias
              - (ac_bias.gradBias * meta_learning_alpha / gradAverage.templates[ac])
        --print('template bias:', torch.sum(ac_bias.gradBias * meta_learning_alpha / gradAverage.templates[ac]), torch.sum(ac_bias.gradBias), gradAverage.templates[ac])
      end

      --print('encoderHidden grad sum:', torch.sum(encoder_hidden.gradWeight), torch.sum(encoder_hidden.gradBias))
      --print('encoderOut grad sum:', torch.sum(encoder_output.gradWeight), torch.sum(encoder_output.gradBias))

      -- testOut = architecture:forward(trainset[{{1, 30}}])
      -- image.save("test_images/step_"..i.."_fixed.png", torch.reshape(testOut[1], 1, image_width, image_width))
    else
      print('Updating .. ')
      architecture:updateParameters(0.0001)
      for ac = 1,num_acrs do
        local ac_bias = architecture.modules[3].modules[ac].modules[3].modules[1].modules[1]
        print(torch.sum(ac_bias.gradBias))
      end
    end
  end
end

if CHECK_GRADS then
  i = 1
  batch = trainset[{{i * bsize, (i + 1) * bsize - 1}}]
  -- print(batch)
  if CHECK_GRADS then
    checkINTMGrads(criterion, architecture, batch, num_acrs)
    checkTemplateGrads(criterion, architecture, batch, num_acrs)
  end
else
  for epc = 1,100 do
    train(epc)
  end
end

parallel.close()
















