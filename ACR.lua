-- ACR module

-- require 'xlua'
local ACR_helper = require 'ACR_helper'
local ACR, Parent = torch.class('nn.ACR', 'nn.Module')
require("gradACRWrapper") -- GPU CUDA kernel for fast gradient (ACR)

-- parallel = require 'parallel'
local Threads = require 'threads'
local sdl = require 'sdl2'
-- local profile = xlua.Profiler()

function ACR:__init(bsize, output_width)
  Parent.__init(self)
  self.bsize = bsize
  self.output = torch.zeros(bsize, output_width, output_width)

  --
end

--[[
function ACR:makeThreads()

  -- define code for workers:
  function worker()
    -- a worker starts with a blank stack, we need to reload
    -- our libraries
    require 'sys'
    require 'torch'
    local ACR_helper = require 'ACR_helper'

    parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)
    parallel.yield() -- parent will wait for this thread to start before continuing
    while true do
      local params = parallel.parent:receive()
      local gradTemplate_thread; local gradPose_thread
      gradTemplate_thread, gradPose_thread = ACR_helper:gradHelper("multicore",params.start_x, params.start_y, params.endhere_x, params.endhere_y, params.output, params.pose, params.bsize, params.template,
                                                                    params.gradOutput, params.gradTemplate, params.gradPose)
      parallel.parent:send({gradTemplate_thread, gradPose_thread})
    end
  end

  parallel.print('making threads')
   local njob = 4*4 --divide image into 4x4 block
   -- fork N processes
   parallel.nfork(njob)
   -- exec worker code in each process
   parallel.children:exec(worker)

  for i=1, njob do
    parallel.children[i]:join()
  end
end
--]]

function ACR:updateOutput(input)
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]

  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)

  local GPU = 0;
  if GPU == 1 then
    a=1;
  else
    for output_x = 1, self.output:size()[2] do
      for output_y = 1, self.output:size()[3] do
        output_coords = torch.Tensor({output_x, output_y, 1})

        --template_coords = pose * output_coords
        --self.output[output_x][output_y] = ACR_helper:getInterpolatedTemplateValue(
        --                                          template,
        --                                          template_coords[1],
        --                                          template_coords[2])

        template_coords = torch.zeros(bsize, 3)
        for i=1, bsize do
          template_coords[{i, {}}] = pose[{i,{},{}}]*output_coords
        end

        self.output[{{}, output_x, output_y}] = torch.cmul(intensity, ACR_helper:getInterpolatedTemplateValue(
                                                    bsize,
                                                    template,
                                                    template_coords[{{},1}], -- x
                                                    template_coords[{{},2}])) -- y
      end
    end
  end
  -- print('INTENSITY', intensity)
  -- print(template)
  return self.output
end



--[[
-- define code for parent:
function ACR:parent(grid_size, output, gradOutput, pose, bsize, template, gradTemplate, gradPose)
  local njob = parallel.nchildren
  grid_side = math.sqrt(njob)
  for jid = 1,njob do
    local ii = (jid-1)% grid_side
    local jj = math.floor((jid-1)/grid_side)
    local factor = math.floor(output:size()[2]/grid_side)

    local start_x=ii*factor + 1;
    local start_y=jj*factor + 1;
    local endhere_x=ii*factor+factor;
    local endhere_y=jj*factor + factor

    --print(string.format('jid: %d ii:%d jj:%d ||| sx: %d, sy:%d, ex:%d, ey:%d\n',jid,ii,jj, start_x, start_y, endhere_x, endhere_y))
    worker_struct = {start_x=start_x, start_y=start_y, endhere_x=endhere_x, endhere_y=endhere_y, output=output,pose= pose, bsize=bsize, template=template,
                                                                  gradOutput=gradOutput, gradTemplate=gradTemplate, gradPose=gradPose}
    parallel.children[jid]:send(worker_struct)
    --recv = parallel.children[jid]:receive()
    --gradTemplate_thread, gradPose_thread = ACR_helper:gradHelper("multicore",start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template,
     --                                                             gradOutput, gradTemplate, gradPose)
  end
  replies = parallel.children:receive()
  for i = 1, #replies do
    self.gradTemplate = self.gradTemplate + replies[i][1]
    self.gradPose = self.gradPose + replies[i][2]
  end
end
--]]

function ACR:updateGradInput(input, gradOutput)
  print('ACR gradOutput', gradOutput:sum())
  --print('ACR grad')
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)


  -- print("IGP: ", iGeoPose)
  -- print("intensity:", intensity)

  self.gradTemplate = self.gradTemplate or torch.Tensor(template:size())
  self.gradTemplate:fill(0)
  self.gradPose = torch.Tensor(pose:size())
  self.gradPose:fill(0)

  local GPU = 1
  local runMulticore = 0

  if GPU == 1 then
    --------------- GPU --------------
    local imwidth = self.output:size()[2];
    local tdim = template:size()[2];
    --pack both grad outputs in one structure for lua-C interface and then unpack
    local gradAll = torch.zeros(bsize*tdim*tdim + bsize*3*3)
    gradAll[{{1, bsize*tdim*tdim}}]=self.gradTemplate:reshape(bsize*tdim*tdim)
    gradAll[{{bsize*tdim*tdim+1, bsize*tdim*tdim+ bsize*3*3}}]=self.gradPose:reshape(bsize*3*3)

    res = gradACRWrapper(imwidth, tdim, bsize,
      self.output:reshape(bsize*imwidth*imwidth), pose:reshape(bsize*3*3),
      template:reshape(bsize*tdim*tdim), gradOutput:reshape(bsize*imwidth*imwidth), intensity,
      gradAll)

    --unpack from lua-C interface
    self.gradTemplate = res[{{1, bsize*tdim*tdim}}]:reshape(bsize, tdim, tdim)
    self.gradPose = res[{{bsize*tdim*tdim+1, bsize*tdim*tdim + bsize*3*3}}]:reshape(bsize,3,3)

  elseif runMulticore == 1 then
    self:parent(grid_size, self.output, gradOutput, pose, bsize, template, self.gradTemplate, self.gradPose)
  else
    start_x = 1; start_y=1;  endhere_x = self.output:size()[2]; endhere_y = self.output:size()[3]
    self.gradTemplate, self.gradPose = ACR_helper:gradHelper("singlecore", start_x, start_y, endhere_x, endhere_y ,self.output, pose, bsize, template,
                                                              gradOutput, self.gradTemplate, self.gradPose, intensity)
  end

  --scaling gradient with intensity
  for i=1,bsize do
    self.gradTemplate[{i,{},{}}] = self.gradTemplate[{i,{},{}}] * intensity[i]
  end

  --self.gradTemplate = self.gradTemplate:reshape(input[1]:size())

  -- print("gradPose before final", self.gradPose:sum())
  self.gradPose = self.gradPose:reshape(bsize,9)
  self.finalgradPose = torch.zeros(bsize, 10)
  self.finalgradPose[{{},{1,9}}] = self.gradPose

  for i=1,bsize do
    -- dividing by intensity as it is scaled already in self.output
    self.finalgradPose[{i,10}] = torch.sum(torch.cmul(self.output[{{i,{},{}}}] / intensity[i], gradOutput[{{i,{},{}}}])) --gradOutput[{i,{},{}}]:sum()
  end

  self.gradInput = {self.gradTemplate, self.finalgradPose}
  -- print('ACR GRAD POSE', torch.sum(intensity))
  print('ACR gradTemplate', self.gradTemplate:sum(), torch.max(self.gradTemplate), torch.min(self.gradTemplate))
  print('ACR gradPose', self.gradPose:sum())
  return self.gradInput
end









