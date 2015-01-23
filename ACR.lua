-- ACR module

require 'xlua'
local ACR_helper = require 'ACR_helper'
local ACR, Parent = torch.class('nn.ACR', 'nn.Module')


local profile = xlua.Profiler()


function ACR:__init(bsize, output_width)
  Parent.__init(self)
  self.bsize = bsize
  self.output = torch.zeros(bsize, output_width, output_width)
end


function ACR:updateOutput(input)
  --profile:start('ACR_updateOutput')
  
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)

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
        template_coords[{i, {}}] = pose[{bsize,{},{}}]*output_coords
      end

      self.output[{{}, output_x, output_y}] = torch.cmul(intensity, ACR_helper:getInterpolatedTemplateValue(
                                                  bsize,
                                                  template,
                                                  template_coords[{{},1}], -- x
                                                  template_coords[{{},2}])) -- y

    end
  end

  --self.output = self.output * intensity[1]
  --profile:lap('ACR_updateOutput')
  
  --profile:printAll()
  
  return self.output
end


function ACR:updateGradInput(input, gradOutput)
  print('ACR grad')
  
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)

  self.gradTemplate = self.gradTemplate or torch.Tensor(template:size())
  self.gradTemplate:fill(0)
  self.gradPose = torch.Tensor(pose:size())
  self.gradPose:fill(0) 

  local runMulticore = 1
  
  if runMulticore == 1 then
    print('Running Multicore ... ')
    -- GPU CODE ENDS HERE
    local Threads = require 'threads'
    local sdl = require 'sdl2'
    local nthread = 8
    local njob = 4*4-1 --divide image into 4x4 block
    local _output = torch.Tensor(self.output:size()):copy(self.output)
    local _gradTemplate = torch.Tensor(self.gradTemplate:size())
    local _gradPose = torch.Tensor(self.gradPose:size())
    
    sdl.init(0)

    local threads = Threads(nthread,
       function()
          gsdl = require 'sdl2'
          ACR_helper = require 'ACR_helper'
          math = require 'math'

       end,
       function(idx)
          --print('starting a new thread/state number:', idx)
          -- we copy here an upvalue of the main thread
          grid_side = math.sqrt(njob+1)
          output = _output
          gradOutput = gradOutput
          pose = pose
          bsize = bsize
          template=template
          gradTemplate = _gradTemplate
          gradPose = _gradPose
       end
    )
    -- now add jobs
    local jobdone = 0
    for jid=1,njob do
       threads:addjob(
          -- the job callback
          function(jobdone)
             local id = tonumber(gsdl.threadID())
             -- note that gmsg was intialized in last worker callback
             --print(string.format('%s -- thread ID is %x counter:%d', gmsg, id, counter))
             -- return a value to the end callback
             --print(counter)
             --return counter
              local ii = (jid-1)% grid_side 
              local jj = math.floor((jid-1)/grid_side)
              local factor = math.floor(output:size()[2]/grid_side)

              local start_x=ii*factor + 1; 
              local start_y=jj*factor + 1;  
              local endhere_x=ii*factor+factor; 
              local endhere_y=jj*factor + factor

              --print(string.format('jid: %d ii:%d jj:%d ||| sx: %d, sy:%d, ex:%d, ey:%d\n',jid,ii,jj, start_x, start_y, endhere_x, endhere_y))
              
              local gradTemplate_thread; local gradPose_thread
              gradTemplate_thread, gradPose_thread = ACR_helper:gradHelper("multicore",start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, 
                                                                            gradOutput, gradTemplate, gradPose)
              return gradTemplate_thread, gradPose_thread
          end,
          -- takes output of the previous function as argument
          function(gradTemplate_thread, gradPose_thread)
             -- note that we can manipulate upvalues of the main thread
             -- as this callback is ran in the main thread!
             --tt[counter] = counter
             self.gradTemplate = self.gradTemplate + gradTemplate_thread
             self.gradPose = self.gradPose + gradPose_thread
             jobdone = jobdone + 1 
          end,
          jid -- argument
       )
    end
    threads:synchronize()-- wait for all jobs to finish
    --print(string.format('%d jobs done', jobdone))
    threads:terminate()

  else    

    start_x = 1; start_y=1;  endhere_x = self.output:size()[2]; endhere_y = self.output:size()[3]
    self.gradTemplate, self.gradPose = ACR_helper:gradHelper("singlecore", start_x, start_y, endhere_x, endhere_y ,self.output, pose, bsize, template, 
                                                              gradOutput, self.gradTemplate, self.gradPose)
  end

  
  for i=1,bsize do
    self.gradTemplate[{i,{},{}}] = self.gradTemplate[{i,{},{}}] * intensity[i]
    self.gradPose[{i,{},{}}] = self.gradPose[{i,{},{}}] * intensity[i]
  end

  self.gradPose = self.gradPose:reshape(bsize,9)

  self.finalgradPose = torch.zeros(bsize, 10)
  self.finalgradPose[{{},{1,9}}] = self.gradPose

  for i=1,bsize do
    self.finalgradPose[{i,10}] = gradOutput[{i,{},{}}]:sum()
  end

  self.gradInput = {self.gradTemplate, self.finalgradPose}

  -- if self.gradInput:nElement() == 0 then
  --   self.gradInput = torch.zeros(input:size())
  -- end

  
  return self.gradInput
end

-- function getTemplateGradient(template, pose, output_x, output_y)

-- end








