-- ACR module

-- require 'xlua'
local ACR_helper = require 'ACR_helper'
local ACR, Parent = torch.class('nn.ACR', 'nn.Module')
--require("gradACRWrapper")

parallel = require 'parallel'
local Threads = require 'threads'
local sdl = require 'sdl2'
-- local profile = xlua.Profiler()

function ACR:__init(bsize, output_width)
  Parent.__init(self)
  self.bsize = bsize
  self.output = torch.zeros(bsize, output_width, output_width)

  --
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


-- define code for workers:
function worker()
  -- a worker starts with a blank stack, we need to reload
  -- our libraries
  require 'sys'
  require 'torch'

  -- print from worker:
  parallel.print('Im a worker, my ID is: ' .. parallel.id .. ' and my IP: ' .. parallel.ip)

  -- define a storage to receive data from top process
  --[[while true do
    -- yield = allow parent to terminate me
    m = parallel.yield()
    if m == 'break' then break end
    -- receive data
    local t = parallel.parent:receive()
    parallel.print('received: ', t.gradTemplate)
    -- send some data back
    parallel.parent:send('dfdf')
  end --]]
  
  local t = parallel.parent:receive()
  parallel.parent:send('dfdf')
end

-- define code for parent:
function parent(grid_size, output, gradOutput, pose, bsize, template, gradTemplate, gradPose)
   parallel.print('Im the parent, my ID is: ' .. parallel.parent.id)

   local njob = 2*2 --divide image into 4x4 block
   -- fork N processes
   parallel.nfork(njob)
   -- exec worker code in each process
   parallel.children:exec(worker)
   -- create a complex object to send to workers
   --[[t = {name='my variable', data=torch.randn(100,100)}
   -- transmit object to each worker
   for i = 1,4 do
      parallel.children:join()
      parallel.children:send(t)
      replies = parallel.children:receive()
   end --]]
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
      --parallel.children[jid]:join()
      parallel.children[jid]:send(worker_struct)
      --recv = parallel.children[jid]:receive()
      --gradTemplate_thread, gradPose_thread = ACR_helper:gradHelper("multicore",start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template,
       --                                                             gradOutput, gradTemplate, gradPose)      
   end
   replies = parallel.children:receive()   
   -- sync/terminate when all workers are done
   parallel.children:join('break')
   parallel.close()
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

  local runMulticore = 0

  if runMulticore == 1 then
    print('Running Multicore ... ')

    if true then
      -- protected execution:
      -- parent(grid_size, self.output, gradOutput, pose, bsize, template, self.gradTemplate, self.gradPose)
      gradACRWrapper(0)
    else
      --using threads-ffi package
      local njob = 2*2 --divide image into 4x4 block
      local output = torch.Tensor(self.output:size()):copy(self.output)
      local gradTemplate = torch.Tensor(self.gradTemplate:size())
      local gradPose = torch.Tensor(self.gradPose:size())

      local nthread = 2

      sdl.init(1)

      local acr_threads = Threads(nthread,
         function()
            gsdl = require 'sdl2'
            ACR_helper = require 'ACR_helper'
         end,
         function(idx)
            --print('starting a new thread/state number:', idx)
            -- we copy here an upvalue of the main thread
            -- grid_side = math.sqrt(njob+1)
            -- output = _output
            -- gradOutput = gradOutput
            -- pose = pose
            -- bsize = bsize
            -- template=template
            -- gradTemplate = _gradTemplate
            -- gradPose = _gradPose
         end
      )


      -- now add jobs
      local jobdone = 0
      for jid=1,njob do
         acr_threads:addjob(
            -- the job callback
            function(jobdone)
               local grid_side = math.sqrt(njob)
               local id = tonumber(gsdl.threadID())

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
                
                --gradTemplate_thread = torch.zeros(20, 11,11)
                --gradPose_thread = torch.zeros(20,7)
                --print('[thread finished] jobid:',jid)
                return gradTemplate_thread, gradPose_thread
            end,
            -- takes output of the previous function as argument
            function(gradTemplate_thread, gradPose_thread)
               -- note that we can manipulate upvalues of the main thread
               -- as this callback is ran in the main thread!
               self.gradTemplate = self.gradTemplate + gradTemplate_thread
               self.gradPose = self.gradPose + gradPose_thread
               jobdone = jobdone + 1
            end,
            jid -- argument
         )
      end
      acr_threads:synchronize()-- wait for all jobs to finish
      --print(string.format('%d jobs done', jobdone))
      acr_threads:terminate()
      --]]
    end
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








