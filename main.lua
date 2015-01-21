-- Unsupervised Capsule Deep Network

require 'cutorch'
require 'cunn'
require 'nngraph'
require 'image'
require 'math'
require 'torch'

require 'dataset-mnist'
require 'INTM'
require 'Bias'
require 'ACR'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run this bad boy.')
cmd:text()
cmd:text('Options')
cmd:option('--gpu',false,'use the GPU for the encoder')
cmd:option('--threads', 4, 'how many threads to use')
cmd:text()
params = cmd:parse(arg)

--torch.setdefaulttensortype('torch.CudaTensor')
torch.setnumthreads(params.threads)

function getDigitSet(digit)
  local trainData = mnist.loadTrainSet(60000, {32,32})
  local digitSet = {_indices = {}, _raw = trainData}
  for i = 1, trainData:size() do
    if trainData[i][2][digit + 1] == 1 then
      table.insert(digitSet._indices, i)
    end
  end
  setmetatable(digitSet, {__index = function (tbl,  key)
      return {tbl._raw[tbl._indices[key]][1][1],
              tbl._raw[tbl._indices[key]][2]}
    end})
  function digitSet:size() return #digitSet._indices end
  return digitSet
end

trainset = getDigitSet(1)

--number of acr's
num_acrs = 9
image_width = 32
h1size = 2000
outsize = 7*num_acrs --affine variables
intm_out_dim = 10

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
decoder = nn.Parallel(1,2)
for ii=1,num_acrs do
  local acr_wrapper = nn.Sequential()
  acr_wrapper:add(nn.Replicate(2))

  acr_wrapper:add(nn.SplitTable(1))

  local acr_in = nn.ParallelTable()
  acr_in:add(nn.Bias(11*11))
  acr_in:add(nn.INTM(7,intm_out_dim))
  acr_wrapper:add(acr_in)
  acr_wrapper:add(nn.ACR(image_width))
  decoder:add(acr_wrapper)
end

architecture:add(decoder)
architecture:add(nn.Reshape(num_acrs, image_width,image_width))


architecture:add(nn.Mul(100))
architecture:add(nn.Exp())
architecture:add(nn.Sum(1))
architecture:add(nn.Log())
architecture:add(nn.Mul(1/100))



criterion = nn.MSECriterion()
--
for i = 1, 300 do
  print("error "..i..": " .. criterion:forward(architecture:forward(trainset[i][1]), trainset[i][1]) )
  
  -- reshape to add an extra dimension for image and clamp so image will interpret correctly
  local out = torch.clamp(torch.reshape(architecture.output, 1,32,32), 0,1)
  if i % 10 == 0 then
    image.save("test_images/recon_"..i..".png", out)
    image.save("test_images/truth_"..i..".png", trainset[i][1])
  end
  
  architecture:zeroGradParameters()
  architecture:backward(trainset[i][1], criterion:backward(architecture.output, trainset[i][1]))
  architecture:updateParameters(0.00002)

end
--


















