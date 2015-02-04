
local ACR_helper = {}

require("sys")


function ACR_helper:gradHelper(mode, start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, gradOutput, _gradTemplate, _gradPose)
  -- print(gradOutput)
  if mode == "singlecore" then
    start_x = 1; start_y=1;
    endhere_x = output:size()[2]; endhere_y = output:size()[3]
    gradTemplate = _gradTemplate; gradPose = _gradPose
  else
    --need to add them outside the thread as threads may override these variables causing funny errors
    gradTemplate = torch.zeros(_gradTemplate:size()); gradPose = torch.zeros(_gradPose:size())
  end

  -- print(gradOutput)

  for output_x = start_x, endhere_x do
    for output_y = start_y, endhere_y do
      --sys.tic()
      -- calculate the correspondence between template and output
      output_coords = torch.Tensor({output_x, output_y, 1})
      --template_coords = pose * output_coords
      template_coords = torch.zeros(bsize, 3)
      for i=1, bsize do
        template_coords[{i, {}}] = pose[{i,{},{}}]*output_coords
      end

      template_x = template_coords[{{},1}] --template_coords[1]
      template_y = template_coords[{{},2}] --template_coords[2]

      -- template_x = template_x - 1/2
      -- template_y = template_y - 1/2

      local template_x_size = template:size()[2]
      local template_y_size = template:size()[3]
      local template_x_after_offset = template_x + template_x_size / 2
      local template_y_after_offset = template_y + template_y_size / 2

      -- print(template_x_after_offset[1], template_y_after_offset[1])

      local x_high_coeff = torch.Tensor(bsize)
      local y_high_coeff = torch.Tensor(bsize)

      --x_high_coeff:map(template_x_after_offset, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x_after_offset % 1
      --y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1

      for i=1,bsize do
        x_high_coeff[i] = math.fmod(template_x_after_offset[i],1)
        y_high_coeff[i] = math.fmod(template_y_after_offset[i],1)
      end

      x_low_coeff  =  -x_high_coeff + 1
      y_low_coeff  =  -y_high_coeff + 1

      x_low  = torch.floor(template_x_after_offset)
      x_high = x_low + 1
      y_low  = torch.floor(template_y_after_offset)
      y_high = y_low + 1

      --[[
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
                gradTemplate[ii][x][y] = gradTemplate[ii][x][y] + dOutdPose[i][j] * gradOutput[ii][output_x][output_y]
              end
          end
        end
      end
      --]]



      -- xt = a*x + b*y + c;
      -- yt = d*x + e*y + f;
      -- Ixy = (1./((x2-x1)*(y2-y1))) * ( (t11*(x2-xt)*(y2-yt)) + (t21*(xt-x1)*(y2-yt)) + (t12*(x2-xt)*(yt-y1)) + (t22*(xt-x1)*(yt-y1)) )

      xxx = nil; yyy=nil;
      for ii=1,bsize do
        local x_low_ii = x_low[ii]; local y_low_ii = y_low[ii];
        local x_high_ii = x_high[ii]; local y_high_ii = y_high[ii];
        local ratio_xy = (x_low_ii-x_high_ii)*(y_low_ii-y_high_ii)

        --------------------------- Template gradient -------------------------------
        if x_low_ii >= 1 and x_low_ii <= template:size()[2] and y_low_ii >= 1 and y_low_ii <= template:size()[2] then
          gradTemplate[{ii, x_low_ii, y_low_ii}] = gradTemplate[{ii, x_low_ii, y_low_ii}]
                                                   + ( (((template_x_after_offset-x_high_ii)*(template_y_after_offset-y_high_ii))/ ratio_xy )
                                                          * gradOutput[ii][output_x][output_y] )

          xxx = x_low_ii; yyy= y_low_ii

          -- if xxx == 1 and yyy == 1  then
          --   print("updateIndex: ", 1)
          --   print("output coordinates: ", output_x, output_y)
          --   print('template_coords' , template_coords)
          --   print('template x, y after offset', template_x_after_offset[1], template_y_after_offset[1])
          --   print(x_low_ii, x_high_ii, y_low_ii, y_high_ii)
          --   print(gradTemplate[{ii,xxx,yyy}], gradOutput[{ii,output_x,output_y}])
          --   print('--------------\n\n')
          -- end
        end

        if x_low_ii >= 1 and x_low_ii <= template:size()[2] and y_high_ii >= 1 and y_high_ii <= template:size()[2] then
          gradTemplate[{ii, x_low_ii,y_high_ii}] = gradTemplate[{ii, x_low_ii,y_high_ii}] + ( -(((template_x_after_offset-x_high_ii)*(template_y_after_offset-y_low_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print(x_low_ii, y_high_ii, template_x_after_offset[1], template_y_after_offset[1], gradOutput[ii][output_x][output_y])
          xxx=x_low_ii; yyy = y_high_ii

          -- if xxx == 1 and yyy == 1  then
          --   print("updateIndex: ", 2)
          --   print("output coordinates: ", output_x, output_y)
          --   print('template_coords' , template_coords)
          --   print('template x, y after offset', template_x_after_offset[1], template_y_after_offset[1])
          --   print(x_low_ii, x_high_ii, y_low_ii, y_high_ii)
          --   print(gradTemplate[{ii,xxx,yyy}], gradOutput[{ii,output_x,output_y}])
          --   print('--------------\n\n')
          -- end
        end

        if x_high_ii >= 1 and x_high_ii <= template:size()[2] and y_low_ii >= 1 and y_low_ii <= template:size()[2] then
          gradTemplate[{ii,x_high_ii,y_low_ii}] = gradTemplate[{ii,x_high_ii,y_low_ii}] + ( -(((template_x_after_offset-x_low_ii)*(template_y_after_offset-y_high_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print(x_high_ii, y_low_ii, template_x_after_offset[1], template_y_after_offset[1], gradOutput[ii][output_x][output_y])
          xxx=x_high_ii; yyy = y_low_ii

          -- if xxx == 1 and yyy == 1  then
          --   print("updateIndex: ", 3)
          --   print("output coordinates: ", output_x, output_y)
          --   print('template_coords' , template_coords)
          --   print('template x, y after offset', template_x_after_offset[1], template_y_after_offset[1])
          --   print(x_low_ii, x_high_ii, y_low_ii, y_high_ii)
          --   print(gradTemplate[{ii,xxx,yyy}], gradOutput[{ii,output_x,output_y}])
          --   print('--------------\n\n')
          -- end

        end

        if x_high_ii >= 1 and x_high_ii <= template:size()[2] and y_high_ii >= 1 and y_high_ii <= template:size()[2] then
          gradTemplate[{ii,x_high_ii,y_high_ii}] = gradTemplate[{ii,x_high_ii,y_high_ii}] + ( (((template_x_after_offset-x_low_ii)*(template_y_after_offset-y_low_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print(x_high_ii, y_high_ii, template_x_after_offset[1], template_y_after_offset[1], gradOutput[ii][output_x][output_y])
          xxx = x_high_ii; yyy = y_high_ii

          -- if xxx == 1 and yyy == 1  then
          --   print("updateIndex: ", 4)
          --   print("output coordinates: ", output_x, output_y)
          --   print('template_coords' , template_coords)
          --   print('template x, y after offset', template_x_after_offset[1], template_y_after_offset[1])
          --   print(x_low_ii, x_high_ii, y_low_ii, y_high_ii)
          --   print(gradTemplate[{ii,xxx,yyy}], gradOutput[{ii,output_x,output_y}])
          --   print('--------------\n\n')
          -- end
        end


      end
      --]]


      ------------------- Pose Gradient ---------------------------
      -- syms a b c d e f x y xt yt x1 x2 y1 y2 t11 t12 t21 t22
      -- >> syms pose_1_1 pose_1_2 pose_1_3 pose_2_1 pose_2_2 pose_2_3
      -- >> xt = pose_1_1*x + pose_1_2*y + pose_1_3;
      -- >> yt = pose_2_1*x + pose_2_2*y + pose_2_3;
      --Ixy = (1./((x2-x1)*(y2-y1))) * ( (t11*(x2-xt)*(y2-yt)) + (t21*(xt-x1)*(y2-yt)) + (t12*(x2-xt)*(yt-y1)) + (t22*(xt-x1)*(yt-y1)) )
      
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
      --cache3 = (pose_1_3 - x_low + pose_1_1*output_x + pose_1_2*output_y)
      --cache4 = (pose_1_3 - x_high + pose_1_1*output_x + pose_1_2*output_y)

      cache5 = torch.cmul(template_val_xlow_ylow, cache2)
      cache6 = torch.cmul(template_val_xlow_yhigh, cache1)
      cache7 = torch.cmul(template_val_xhigh_ylow, cache2)
      cache8 = torch.cmul(template_val_xhigh_yhigh, cache1)

      --print('df', ((cache5 - cache6 - cache7 + cache8) * output_x)[1], gradOutput[{{},output_x,output_y}][1])
      gradPose[{{},1,1}] = gradPose[{{},1,1}] + torch.cmul( (cache5 - cache6 - cache7 + cache8) * output_x, gradOutput[{{},output_x,output_y}] )

      --[[
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
      --]]
    end
  end

  if tostring(torch.sum(gradPose)) == tostring(0/0) then
    print("ERROR!")
--    print('gradOutput:', torch.sum(gradOutput))
  end

  return gradTemplate, gradPose
    --]]
end


function ACR_helper:getTemplateValue(bsize, template, template_x, template_y)
  local res = torch.zeros(bsize)
  for i = 1,bsize do
    if template_x[i] < 1 or template_x[i] > template:size()[2] or template_y[i] < 1 or template_y[i] > template:size()[3] then
      res[i] = 0
    else
      res[i] = template[i][template_x[i]][template_y[i]]
    end
  end
  return res
end

function ACR_helper:getInterpolatedTemplateValue(bsize, template, template_x, template_y)
  -- template_x = template_x - 1/2
  -- template_y = template_y - 1/2
  local template_x_size = template:size()[2]
  local template_y_size = template:size()[3]
  local template_x_after_offset = template_x + template_x_size / 2
  local template_y_after_offset = template_y + template_y_size / 2

  local x_high_coeff = torch.Tensor(bsize)
  local y_high_coeff = torch.Tensor(bsize)

  --x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
  --y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1

  for i=1,bsize do
    x_high_coeff[i] = math.fmod(template_x_after_offset[i],1)
    y_high_coeff[i] = math.fmod(template_y_after_offset[i],1)
  end

  x_low_coeff  =  -x_high_coeff + 1
  y_low_coeff  =  -y_high_coeff + 1

  x_low  = torch.floor(template_x_after_offset)
  x_high = x_low + 1
  y_low  = torch.floor(template_y_after_offset)
  y_high = y_low + 1


  return torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low,  y_low) , torch.cmul(x_low_coeff  , y_low_coeff))  +
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_low) , torch.cmul(x_high_coeff , y_low_coeff )) +
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low,  y_high), torch.cmul(x_low_coeff  , y_high_coeff ))+
         torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_high), torch.cmul(x_high_coeff , y_high_coeff ))

  --[[
  local x2_x1 = x_high - x_low
  local y2_y1 = y_high - y_low
  local x2_x = x_high - template_x
  local y2_y = y_high - template_y
  local x_x1 = template_x - x_low
  local y_y1 = template_y - y_low


  local t11 = ACR_helper:getTemplateValue(bsize, template, x_low, y_low)
  local t12 = ACR_helper:getTemplateValue(bsize, template, x_low, y_high)
  local t21 = ACR_helper:getTemplateValue(bsize, template, x_high, y_low)
  local t22 = ACR_helper:getTemplateValue(bsize, template, x_high, y_high)

  local ratio = torch.pow(torch.cmul(x2_x1, y2_y1),-1)

  local term_t11 = torch.cmul(t11, torch.cmul(x2_x, y2_y))
  local term_t21 = torch.cmul(t21, torch.cmul(x_x1, y2_y))
  local term_t12 = torch.cmul(t12, torch.cmul(x2_x, y_y1))
  local term_t22 = torch.cmul(t22, torch.cmul(x_x1, y_y1))

  return torch.cmul(ratio, (term_t11 + term_t12 + term_t21 + term_t22))
  --]]
end




return ACR_helper