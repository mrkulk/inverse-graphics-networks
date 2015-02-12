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
require 'xlua'
require 'rmsprop'

torch.manualSeed(10)

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Build and test an inverse graphics deep network.')
cmd:text()
cmd:text('Options')
cmd:option('-imagefolder','default','where to store image outputs from the network')
cmd:text()

params = cmd:parse(arg)
os.execute("mkdir test_images/"..params.imagefolder)
saved_image_path = "test_images/"..params.imagefolder.."/"

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
  -- function digitSet:size() return 1 end 
  function digitSet:size() return #self._indices end

  function digitSet:getBatch(batch_size, index)
    if batch_size * index > self:size() then
      error("Batch index too high! Not that many samples.")
    end
    return self[{{(index - 1) * batch_size + 1, index * batch_size}}]
  end
  function digitSet:nBatches(batch_size)
    return math.floor(self:size()/bsize)
  end
  return digitSet
end

trainset = getDigitSet(1)

--number of acr's
num_acrs = 1
image_width = 32
h1size = 500
outsize = 7*num_acrs --affine variables
intm_out_dim = 10
bsize = 50
template_width = 11

architecture = nn.Sequential()
encoder = nn.Sequential()
  encoder:add(nn.Reshape(image_width * image_width))
  encoder:add(nn.Linear(image_width * image_width,h1size))
  encoder:add(nn.ReLU())
  encoder:add(nn.Linear(h1size, outsize))

architecture:add(encoder)

architecture:add(nn.Reshape(num_acrs,7))

-- Creating intm and acr's
-- decoder = nn.Parallel(2,2)
decoder = nn.ParallelParallel(2,2)
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
        -- intensityWrapper:add(nn.PrintModule('INTENSITY MOD'))
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

BCE = true
if BCE then
  architecture:add(nn.Reshape(num_acrs, image_width,image_width))
  architecture:add(nn.Sum(2))
  architecture:add(nn.Reshape(image_width*image_width))
  architecture:add(nn.Sigmoid())
  criterion = nn.BCECriterion()
  -- criterion.sizeAverage = false

else
  architecture:add(nn.Reshape(num_acrs, image_width,image_width))

  architecture:add(nn.MulConstant(100))
  architecture:add(nn.Exp())
  architecture:add(nn.Sum(2))
  -- architecture:add(nn.PrintModule("before log"))
  architecture:add(nn.Log())
  -- architecture:add(nn.PrintModule("after log"))

  architecture:add(nn.MulConstant(1/100))
  criterion = nn.MSECriterion()
  -- criterion.sizeAverage = false
end


rmsGradAverages = {
  templates = torch.ones(num_acrs),
  encoderHiddenWt = 1,
  encoderOutputWt = torch.ones(7),
  encoderHiddenBias = 1,
  encoderOutputBias = torch.ones(7)
}

config = {
    learningRate = -0.001,
    momentumDecay = 0.1,
    updateDecay = 0.01
}

--encoder meta learning rate
ENC_LR1 = 1

affine = {
  learningRates = {
    0.1, --t1    [1]
    0.1, --t2    [2]
    0.01, --s1   [3]
    0.01, --s2   [4]
    0.1, --z     [5]
    0.1, --theta [6]
    1 --intensity[7]
  }
}
ACR_MLR = 10000

state = {}


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
    x_index = math.floor((j - 1) / acrs_across)
    x_location = (x_index) * image_width + x_index * padding
    y_index = math.floor((j - 1) % acrs_down)
    y_location = (y_index) * image_width + y_index * padding
    acr_output[{{x_location + 1, x_location + image_width},
                {y_location + 1, y_location + image_width}}] = acr.output[1]
  end
  acr_output = acr_output:reshape(1, acr_output:size()[1], acr_output:size()[2])
  image.save(saved_image_path.."epoch_"..epc.."_step_"..step.."_acrs.png", acr_output)
end

function saveTemplates(epc, step, model)
  -- acrs = model:findModules('nn.ACR')
  local biases = {}
  for i = 1, num_acrs do
    table.insert(biases, model.modules[3].modules[i].modules[3].modules[1].modules[1])
  end
  padding = 5
  across = math.ceil(math.sqrt(#biases))
  down = math.ceil(#biases / across)

  bias_output = torch.zeros(
                    template_width * down + (down - 1) * padding,
                    template_width * across + (across - 1) * padding)
  for j, bias in ipairs(biases) do
    x_index = math.floor((j - 1) / across)
    x_location = (x_index) * template_width + x_index * padding
    y_index = math.floor((j - 1) % down)
    y_location = (y_index) * template_width + y_index * padding
    bias_output[{{x_location + 1, x_location + template_width},
                {y_location + 1, y_location + template_width}}] = bias.output[1]
  end
  bias_output = bias_output:reshape(1, bias_output:size()[1], bias_output:size()[2])
  image.save(saved_image_path.."epoch_"..epc.."_step_"..step.."_templates.png", bias_output)
end


function train(epc)
  total_recon_error = 0
  for i = 1, trainset:nBatches(bsize) do
    batch = trainset:getBatch(bsize, i)--trainset[{{(i - 1) * bsize + 1, i * bsize}}]
    if BCE then
      obatch = batch:reshape(bsize, image_width*image_width)
    else
      obatch = batch
    end
    recon_error = criterion:forward(architecture:forward(batch), obatch)
    total_recon_error = total_recon_error + recon_error

    --print("epoch:" .. tostring(epc) .. " batch:"..i.."/" .. tostring(trainset:size()) .. " error: " .. recon_error )

    --print(architecture:forward(batch))
    -- print(architecture.output)

    architecture:zeroGradParameters()
    architecture:backward(batch, criterion:backward(architecture.output, obatch))

    RMSPROP = true

    if RMSPROP == true then
      

      -- Stochastic RMSProp on separate layers
      local encoder_hidden = architecture.modules[1].modules[2]
      prev = encoder_hidden.weight:clone() 
      -- print(torch.sum(encoder_hidden.gradWeight))
      encoder_hidden.weight = rmsprop(encoder_hidden.weight, encoder_hidden.gradWeight, rmsGradAverages.encoderHiddenWt,ENC_LR1)
      print('enc_hidden weight diff:', torch.norm(prev - encoder_hidden.weight), torch.sum(torch.abs(encoder_hidden.gradWeight)))

      -- prev = encoder_hidden.bias:clone() 
      -- print(torch.sum(encoder_hidden.gradBias))
      encoder_hidden.bias = rmsprop(encoder_hidden.bias, encoder_hidden.gradBias, rmsGradAverages.encoderHiddenBias,ENC_LR1)
      -- print('enc_hidden bias diff:', torch.norm(prev - encoder_hidden.bias), torch.sum(torch.abs(encoder_hidden.gradBias)))


      local encoder_output = architecture.modules[1].modules[4]
      for ii = 1,num_acrs*7 do    
        prev = encoder_output.weight[{ii,{}}]:clone()
        -- print(torch.sum(encoder_output.gradWeight))
        encoder_output.weight[{ii,{}}] = rmsprop(encoder_output.weight[{ii,{}}], encoder_output.gradWeight[{ii,{}}], rmsGradAverages.encoderOutputWt[math.fmod(ii,7)+1], affine.learningRates[math.fmod(ii,7)+1])
        --print('enc_output weight diff:', torch.norm(prev - encoder_output.weight), torch.sum(torch.abs(encoder_output.gradWeight)))
        -- prev = encoder_output.bias:clone()
        -- print(torch.sum(encoder_output.gradBias))
        encoder_output.bias[ii] = rmsprop(encoder_output.bias[ii], encoder_output.gradBias[ii], rmsGradAverages.encoderOutputBias[math.fmod(ii,7)+1],affine.learningRates[math.fmod(ii,7)+1])
        --print('enc_output bias diff:', torch.norm(prev - encoder_output.bias), torch.sum(torch.abs(encoder_output.gradBias)))
        print('ID', math.fmod(ii,7)+1, 'val:',encoder_output.output[1][ii], ' enc_output bias diff:', torch.norm(prev - encoder_output.weight[{ii,{}}]), torch.sum(torch.abs(encoder_output.gradWeight[{ii,{}}] )))
      end


      for ac = 1,num_acrs do
        local ac_bias = architecture.modules[3].modules[ac].modules[3].modules[1].modules[1]
        prev_bias = ac_bias.bias:clone()
        ac_bias.bias = rmsprop(ac_bias.bias, ac_bias.gradBias, rmsGradAverages.templates[ac], ACR_MLR) --1 - MSE 5-BCE
        print('ac#', ac,  ' diff:', torch.norm(prev_bias - ac_bias.bias), torch.sum(torch.abs(ac_bias.gradBias)))
      end



      -- disp progress
      xlua.progress(i, trainset:nBatches(bsize))
    else
      print('Updating .. ')
      architecture:updateParameters(0.0001)
      -- disp progress
      xlua.progress(i, trainset:nBatches(bsize))

    end

    saveImages = true
    if saveImages and i % 1 == 0 then
      test_batch = trainset:getBatch(bsize, 1)
      architecture:forward(test_batch)
      local out = torch.reshape(architecture.output[1], 1,image_width,image_width)
      
      saveTemplates(epc, i, architecture)
      image.save(saved_image_path.."epoch_"..epc.."_step_"..i.."_recon.png", out)
      image.save(saved_image_path.."epoch_"..epc.."_step_"..i.."_truth.png", test_batch[1])-- batch[1])
    end

    -- display progress
    xlua.progress(i, trainset:nBatches(bsize))
  end
end

if CHECK_GRADS then
  i = 1
  batch = trainset:getBatch(bsize, i)
  -- print(batch)
  if CHECK_GRADS then
    --checkEncoderGrads(criterion, architecture, batch)
    checkTemplateGrads(criterion, architecture, batch, num_acrs)
  end
else
  -- num_train_batches = math.floor(trainset:size()/bsize)
  for epc = 1,5000 do
    train(epc)
  end
end


parallel.close()
















