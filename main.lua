-- Unsupervised Capsule Deep Network

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run this bad boy.')
cmd:text()
cmd:text('Options')
cmd:option('--gpu',false,'use the GPU for the encoder')
cmd:option('--threads', 4, 'how many threads to use')
cmd:text()
params = cmd:parse(arg)
-- params = {}

--torch.setdefaulttensortype('torch.CudaTensor')
torch.setnumthreads(params.threads)

require 'image'
require 'math'

if params.gpu then
  require 'cutorch'
  require 'cunn'
else
  require 'nn'
end

require 'dataset-mnist'
require 'INTM'
require 'Bias'
require 'ACR'

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
bsize = 2

architecture = nn.Sequential()
encoder = nn.Sequential()
encoder:add(nn.Reshape(image_width * image_width))
encoder:add(nn.Linear(image_width * image_width,h1size))
encoder:add(nn.Tanh())
encoder:add(nn.Linear(h1size, outsize))

if params.gpu then
  architecture:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
  encoder:cuda()
  architecture:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
else
  architecture:add(encoder)
end

architecture:add(nn.Reshape(num_acrs,7))
--architecture:add(nn.SplitTable(1))


-- Creating intm and acr's
decoder = nn.Parallel(2,2)
for ii=1,num_acrs do
  local acr_wrapper = nn.Sequential()
  acr_wrapper:add(nn.Replicate(2))

  acr_wrapper:add(nn.SplitTable(1))

  local acr_in = nn.ParallelTable()
  acr_in:add(nn.Bias(bsize, 11*11))
  acr_in:add(nn.INTM(bsize, 7,intm_out_dim))
  acr_wrapper:add(acr_in)
  acr_wrapper:add(nn.ACR(bsize, image_width))
  
  decoder:add(acr_wrapper)
end

architecture:add(decoder)
architecture:add(nn.Reshape(num_acrs, image_width,image_width))


architecture:add(nn.MulConstant(100))
architecture:add(nn.Exp())
architecture:add(nn.Sum(2))
architecture:add(nn.Log())
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
  acrs = model:findModules('nn.ACR')
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


for i = 1, 5 do
  batch = trainset[{{i * bsize, (i + 1) * bsize - 1}}]
  print("error "..i..": " .. criterion:forward(architecture:forward(batch), batch) )
  --print(architecture:forward(batch))
  -- print(architecture.output)


  if i % 1 == 0 then
    local out = torch.clamp(torch.reshape(architecture.output[1], 1,image_width,image_width), 0,1)
    saveACRs(i, architecture)
    image.save("test_images/step_"..i.."_recon.png", out)
    image.save("test_images/step_"..i.."_truth.png", batch[1])
  end

  architecture:zeroGradParameters()
  architecture:backward(batch, criterion:backward(architecture.output, batch))
  architecture:updateParameters(0.000001)

end

















