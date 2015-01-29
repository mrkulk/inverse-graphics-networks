
local ACR_helper = {}

require("sys")


function ACR_helper:gradHelper(mode, start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, gradOutput, _gradTemplate, _gradPose)
  if mode == "singlecore" then
    start_x = 1; start_y=1;
    endhere_x = output:size()[2]; endhere_y = output:size()[3]
    gradTemplate = _gradTemplate; gradPose = _gradPose
  else
    --need to add them outside the thread as threads may override these variables causing funny errors
    gradTemplate = torch.zeros(_gradTemplate:size()); gradPose = torch.zeros(_gradPose:size())
  end

  for output_x = start_x, endhere_x do
    for output_y = start_y, endhere_y do
      --sys.tic()
      -- calculate the correspondence between template and output
      output_coords = torch.Tensor({output_x, output_y, 1})
      --template_coords = pose * output_coords
      template_coords = torch.zeros(bsize, 3)
      for i=1, bsize do
        template_coords[{i, {}}] = pose[{bsize,{},{}}]*output_coords
      end

      template_x = template_coords[{{},1}]--template_coords[1]
      template_y = template_coords[{{},2}] --template_coords[2]

      template_x = template_x - 1/2
      template_y = template_y - 1/2

      local x_high_coeff = torch.Tensor(bsize)
      local y_high_coeff = torch.Tensor(bsize)

      --x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
      --y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1
      
      for i=1,bsize do
        x_high_coeff[i] = math.fmod(template_x[i],1)
        y_high_coeff[i] = math.fmod(template_y[i],1)
      end

      x_low_coeff  =  -x_high_coeff + 1
      y_low_coeff  =  -y_high_coeff + 1

      x_low  = torch.floor(template_x)
      x_high = x_low + 1
      y_low  = torch.floor(template_y)
      y_high = y_low + 1


      for ii=1,bsize do
        -----------------------------------------------------------------------------
        --------------------------- Template gradient -------------------------------
        -----------------------------------------------------------------------------
        -- calculate the derivatives for the template
        x_vec = torch.Tensor({x_low_coeff[ii], x_high_coeff[ii]})
        y_vec = torch.Tensor({y_low_coeff[ii], y_high_coeff[ii]})

        -- outer product
        dOutdPose = torch.ger(x_vec, y_vec)
        for i, x in ipairs({x_low[ii], x_high[ii]}) do
          for j, y in ipairs({y_low[ii], y_high[ii]}) do
              if x >= 1 and x <= template:size()[2]
                and y >= 1 and y <= template:size()[3] then
                gradTemplate[ii][x][y] = gradTemplate[ii][x][y] + dOutdPose[i][j]
              end
          end
        end
      end
      --print('template:', sys.toc())

      --sys.tic()

      template_val_xhigh_yhigh = ACR_helper:getTemplateValue(bsize, template, x_high, y_high)
      template_val_xhigh_ylow = ACR_helper:getTemplateValue(bsize, template, x_high, y_low)
      template_val_xlow_ylow = ACR_helper:getTemplateValue(bsize, template, x_low, y_low)
      template_val_xlow_yhigh = ACR_helper:getTemplateValue(bsize, template, x_low, y_high)
      
      pose_1_1 = pose[{{},1,1}]
      pose_1_2 = pose[{{},1,2}]
      pose_1_3 = pose[{{},1,3}]
      pose_2_1 = pose[{{},2,1}]
      pose_2_2 = pose[{{},2,2}]
      pose_2_3 = pose[{{},2,3}]


      cache1 = (pose_2_3 - y_low + pose_2_1*output_x + pose_2_2*output_y)
      cache2 = (pose_2_3 - y_high + pose_2_1*output_x + pose_2_2*output_y)
      cache3 = (pose_1_3 - x_low + pose_1_1*output_x + pose_1_2*output_y)
      cache4 = (pose_1_3 - x_high + pose_1_1*output_x + pose_1_2*output_y)

      cache5 = torch.cmul(template_val_xhigh_yhigh, cache3)
      cache6 = torch.cmul(template_val_xlow_yhigh, cache4)
      cache7 = torch.cmul(template_val_xhigh_ylow, cache3)
      cache8 = torch.cmul(template_val_xlow_ylow, cache4)

      cache9 = torch.cmul(gradOutput[{{},output_x,output_y}], cache5-cache6 )
      cache10 = (cache7-cache8)

      cache11 = torch.cmul(template_val_xhigh_ylow - template_val_xlow_ylow, cache2)

      cache12 = torch.cmul(
              gradOutput[{{},output_x,output_y}], 
              torch.cmul(template_val_xhigh_yhigh - template_val_xlow_yhigh, cache1)
            )

      cache13 =  cache12 - cache11

      cache14 = (cache9 - cache10)


      -- add dCost/dOut(x,y) * dOut(x,y)/dPose for this (x,y)
      gradPose[{{},1,1}] = gradPose[{{},1,1}] + cache13*output_x

      gradPose[{{},1,2}] = gradPose[{{},1,2}] + cache13*output_y

      gradPose[{{},1,3}] = gradPose[{{},1,3}] + cache12 - cache11

      gradPose[{{},2,1}] = gradPose[{{},2,1}] +  cache14*output_x

      gradPose[{{},2,2}] = gradPose[{{},2,2}] + cache14*output_y

      gradPose[{{},2,3}] = gradPose[{{},2,3}] +  
              torch.cmul(gradOutput[{{},output_x,output_y}], cache5) - cache6 - cache7 + cache8

      --print(output_x, output_y,gradPose[{{},2,3}][1] )
      --if output_x == 3 and output_y == 4 then
      --  print('CPU:',  template_val_xlow_ylow[1], template_val_xlow_yhigh[1], template_val_xhigh_ylow[1], template_val_xhigh_yhigh[1])
        --for i = 1, 3 do
        --  for j = 1, 3 do
        --    print(pose[1][i][j])
        --  end
        --end
      --end

      --print('posegrad:' , sys.toc())
    end
  end
    return gradTemplate, gradPose
    --]]
end


function ACR_helper:getTemplateValue(bsize, template, template_x, template_y)
  local template_x_size = template:size()[2] + 1
  local template_y_size = template:size()[3] + 1
  local output_x = torch.floor(template_x + template_x_size / 2)
  local output_y = torch.floor(template_y + template_y_size / 2)

  local res = torch.zeros(bsize)
  for i = 1,bsize do
    if output_x[i] < 1 or output_x[i] > template:size()[2] or output_y[i] < 1 or output_y[i] > template:size()[3] then
      res[i] = 0
    else
      --[[if output_x[i] > template:size()[2] or output_x[i] < 1 then
        print('WTF-x')
      end
      if output_y[i] > template:size()[3] or output_y[i] < 1 then
        print('WTF-y')
      end--]]
      res[i] = template[i][output_x[i]][output_y[i]]
    end
  end
  return res
end

function ACR_helper:getInterpolatedTemplateValue(bsize, template, template_x, template_y)
  template_x = template_x - 1/2
  template_y = template_y - 1/2

  local x_high_coeff = torch.Tensor(bsize)
  local y_high_coeff = torch.Tensor(bsize)
  
  --x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
  --y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1

  for i=1,bsize do
    x_high_coeff[i] = math.fmod(template_x[i],1)
    y_high_coeff[i] = math.fmod(template_y[i],1)
  end

  x_low_coeff  =  -x_high_coeff + 1
  y_low_coeff  =  -y_high_coeff + 1

  x_low  = torch.floor(template_x)
  x_high = x_low + 1
  y_low  = torch.floor(template_y)
  y_high = y_low + 1

  --return ACR_helper:getTemplateValue(bsize, template, x_low,  y_low)  * x_low_coeff  * y_low_coeff  +
  --       ACR_helper:getTemplateValue(bsize, template, x_high, y_low)  * x_high_coeff * y_low_coeff  +
  --       ACR_helper:getTemplateValue(bsize, template, x_low,  y_high) * x_low_coeff  * y_high_coeff +
  --       ACR_helper:getTemplateValue(bsize, template, x_high, y_high) * x_high_coeff * y_high_coeff

  return torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low,  y_low) , torch.cmul(x_low_coeff  , y_low_coeff))  +
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_low) , torch.cmul(x_high_coeff , y_low_coeff )) +
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low,  y_high), torch.cmul(x_low_coeff  , y_high_coeff ))+
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_high), torch.cmul(x_high_coeff , y_high_coeff ))
end




return ACR_helper