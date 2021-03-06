require 'nn'
require 'nngraph'
require 'INTM'
require 'torch'

--require('mobdebug').start()

function intm_test()
  imsize=10; h1size=20; igeo_outsize=7;
  input = nn.Identity()()
  h1 = nn.Tanh()(nn.Linear(imsize,h1size)(input))
  
  igeopose = nn.INTM(h1size, igeo_outsize)(h1)
  cost = nn.Mean()(igeopose)
  graph=nn.gModule({input},{igeopose})
  
  print('Forward pass ... ')
  indata = torch.rand(imsize)
  print(graph:forward(indata))
  
  print('Backward pass ...')
  dummy_target = torch.rand(igeo_outsize)
  print(graph:backward(indata, dummy_target))
end

function test_threads()
  local Threads = require 'threads'
  local sdl = require 'sdl2'

  local nthread = 4
  local njob = 4
  local msg = "hello from a satellite thread"
  local tt = torch.rand(4)
  -- init SDL (follow your needs)
  sdl.init(0)

  -- init the thread system
  -- one lua state is created for each thread

  -- the function takes several callbacks as input, which will be executed
  -- sequentially on each newly created lua state
  local threads = Threads(nthread,
     -- typically the first callback requires modules
     -- necessary to serialize other callbacks
     function()
        gsdl = require 'sdl2'
     end,

     -- other callbacks (one is enough in general!) prepare stuff
     -- you need to run your program
     function(idx)
        print('starting a new thread/state number:', idx)
        gmsg = msg -- we copy here an upvalue of the main thread
        counter = idx
     end
  )

  -- now add jobs
  local jobdone = 0
  for i=1,njob do
     threads:addjob(
        -- the job callback
        function(i)
           local id = tonumber(gsdl.threadID())
           -- note that gmsg was intialized in last worker callback
           print(string.format('%s -- thread ID is %x counter:%d', gmsg, id, counter))
           -- return a value to the end callback

           return counter
        end,

        -- the end callback runs in the main thread.
        -- takes output of the previous function as argument
        function(counter)
           -- note that we can manipulate upvalues of the main thread
           -- as this callback is ran in the main thread!
           tt[counter] = counter
           jobdone = jobdone + 1 
        end
     )
  end

  -- wait for all jobs to finish
  threads:synchronize()

  print(string.format('%d jobs done', jobdone))

  -- of course, one can run more jobs if necessary!

  -- terminate threads
  threads:terminate()
  print("\nAFTER:")
  print(tt)
end


function basic_torch()
  x1=nn.Linear(20,20)()
  x2=nn.Linear(10,10)()
  m0=nn.Linear(20,1)(nn.Tanh()(x1))
  m1=nn.Linear(10,1)(nn.Tanh()(x2))
  madd=nn.CAddTable()({m0,m1})
  m2=nn.Sigmoid()(madd)
  m3=nn.Tanh()(madd)
  gmod = nn.gModule({x1,x2},{m2,m3})

  x = torch.rand(20)
  y = torch.rand(10)

  gmod:updateOutput({x,y})
  gmod:updateGradInput({x,y},{torch.rand(1),torch.rand(1)})

  out=gmod:forward({x,y})
  grad=gmod:backward({x,y},{torch.rand(1),torch.rand(1)})
  print(grad[2])
end

function istensorEqtest(t1,t2)
  local errors = torch.sum(t1-t2)
  if errors == 0 then
    return "OK"
  else
    return "FAILED"
  end
end

function custom_cuda_kernels()
  require("gradACRWrapper")
  local ACR_helper = require("ACR_helper")

  torch.manualSeed(1)

  --------------- GPU --------------
  imwidth = 32; 
  tdim = 11; 
  bsize = 1; 
  output = torch.rand(bsize,imwidth,imwidth); prev_output=output:clone()
  pose = torch.rand(bsize,3,3); prev_pose = pose:clone()
  template = torch.rand(bsize,tdim,tdim); prev_template = template:clone();
  gradOutput = torch.rand(bsize, imwidth, imwidth); prev_gradOutput = gradOutput:clone();

  intensity = torch.ones(bsize)

  gradTemplate = torch.zeros(bsize, tdim, tdim)
  gradPose = torch.zeros(bsize, 3 , 3)

  gradAll = torch.zeros(bsize*tdim*tdim + bsize*3*3)
  gradAll[{{1, bsize*tdim*tdim}}]=gradTemplate:reshape(bsize*tdim*tdim)
  gradAll[{{bsize*tdim*tdim+1, bsize*tdim*tdim+ bsize*3*3}}]=gradPose:reshape(bsize*3*3)
  

  res = gradACRWrapper(imwidth, tdim, bsize, 
    output:reshape(bsize*imwidth*imwidth), pose:reshape(bsize*3*3), 
    template:reshape(bsize*tdim*tdim), gradOutput:reshape(bsize*imwidth*imwidth), intensity, gradAll)

  gpu_gradTemplate = res[{{1, bsize*tdim*tdim}}]:reshape(bsize, tdim, tdim):clone()
  gpu_gradPose = res[{{bsize*tdim*tdim+1, bsize*tdim*tdim + bsize*3*3}}]:reshape(bsize,3,3):clone()
  --print(gradTemplate:size())
  --print(gradPose:size())

  ------------- CPU -----------------
  endhere_x = output:size()[2]; endhere_y = output:size()[3];
  cgradTemplate = torch.zeros(bsize, tdim, tdim)
  cgradPose = torch.zeros(bsize, 3 , 3)
  cpu_gradTemplate, cpu_gradPose = ACR_helper:gradHelper("singlecore", 1, 1, endhere_x, endhere_y, 
              output, pose, bsize, template, gradOutput, cgradTemplate, cgradPose, intensity)

  print('------------------------------------------------------------')
  print('gradTemplate Errors: ', torch.sum(cpu_gradTemplate - gpu_gradTemplate))
  print('gradPose Errors: ', torch.sum(cpu_gradPose - gpu_gradPose))
  print('--------Test to make sure others variables unchanged -------')
  print('output: ', istensorEqtest(output, prev_output))
  print('pose: ', istensorEqtest(pose, prev_pose))
  print('template: ', istensorEqtest(template, prev_template))
  print('gradOutput: ', istensorEqtest(gradOutput, prev_gradOutput))
  
  
  print('cpu:')
  print(cpu_gradPose)
  print('gpu:')
  print(gpu_gradPose)
end

--test_threads()
custom_cuda_kernels()
--intm_test()