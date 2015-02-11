
local ACR_helper = {}

require("sys")


function ACR_helper:updateGradPoseAtLocation(batchIndex, output_x, output_y, template_x_after_offset, template_y_after_offset, x_low, x_high, y_low, y_high, template, pose, gradPose, gradOutput, intensity, scale)
  scale = scale or 1
  local template_val_xhigh_yhigh = ACR_helper:getSingleTemplateValue(template, x_high, y_high) * intensity
  local template_val_xhigh_ylow = ACR_helper:getSingleTemplateValue(template, x_high, y_low) * intensity
  local template_val_xlow_ylow = ACR_helper:getSingleTemplateValue(template, x_low, y_low) * intensity
  local template_val_xlow_yhigh = ACR_helper:getSingleTemplateValue(template, x_low, y_high) * intensity


  pose_1_1 = pose[{1,1}]
  pose_1_2 = pose[{1,2}]
  pose_1_3 = pose[{1,3}]
  pose_2_1 = pose[{2,1}]
  pose_2_2 = pose[{2,2}]
  pose_2_3 = pose[{2,3}]

  local cache1 = (template_y_after_offset - y_low)  -- second line cache
  local cache2 = (template_y_after_offset - y_high) -- first line cache
  local cache3 = (template_x_after_offset - x_low)
  local cache4 = (template_x_after_offset - x_high)

  local cache5 = template_val_xlow_ylow * cache2
  local cache6 = template_val_xlow_yhigh * cache1
  local cache7 = template_val_xhigh_ylow * cache2
  local cache8 = template_val_xhigh_yhigh * cache1

  local cache9 = (cache5 - cache6 - cache7 + cache8)

  gradPose[{1,1}] = gradPose[{1,1}] + scale * cache9 * output_x * gradOutput[{output_x,output_y}]

  gradPose[{1,2}] = gradPose[{1,2}] + scale * cache9 * output_y * gradOutput[{output_x,output_y}]

  gradPose[{1,3}] = gradPose[{1,3}] + scale * cache9 * gradOutput[{output_x, output_y}]


  local cache10 = template_val_xlow_ylow * cache4
  local cache11 = template_val_xlow_yhigh * cache4
  local cache12 = template_val_xhigh_ylow * cache3
  local cache13 = template_val_xhigh_yhigh * cache3

  local cache14 = (cache10 - cache11 - cache12 + cache13)

  gradPose[{2,1}] = gradPose[{2,1}] + scale * cache14 * output_x * gradOutput[{output_x,output_y}]

  gradPose[{2,2}] = gradPose[{2,2}] + scale * cache14 * output_y * gradOutput[{output_x,output_y}]

  gradPose[{2,3}] = gradPose[{2,3}] + scale * cache14 * gradOutput[{output_x,output_y}]

  if (cache9 * output_x * gradOutput[{output_x,output_y}]) ~= 0 then
    print('bid:',batchIndex, output_x, output_y, cache9 * output_x * gradOutput[{output_x,output_y}])
  end
  -- if torch.cmul( cache14 * output_x, gradOutput[{{},output_x,output_y}])[1] ~= 0 then

  --   -- print("output coords with gradPose(1,1) grads:", output_x, output_y)
  --   -- print("with offset", pose_1_3[1], pose_2_3[1], "cache14 is", cache14[1])
  --   -- print("with offset", pose_1_3[1], pose_2_3[1], "cache4 is", cache4[1])
  --   -- print("with offset", pose_1_3[1], pose_2_3[1], "cache10 is", cache10[1])
  --   print("x, y after offset", template_x_after_offset[1], template_y_after_offset[1], "cache3 is", cache3[1], 'multiplying val at x:', x_high[1])
  --   print("x, y after offset", template_x_after_offset[1], template_y_after_offset[1], "cache4 is", cache4[1], 'multiplying val at x:', x_low[1])
  -- end
end


function ACR_helper:gradHelper(mode, start_x, start_y, endhere_x, endhere_y, output, pose, bsize, template, gradOutput, _gradTemplate, _gradPose, intensity)
  -- print(gradOutput)
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
      -- calculate the correspondence between template and output
      output_coords = torch.Tensor({output_x, output_y, 1})
      template_coords = torch.zeros(bsize, 3)
      for i=1, bsize do
        template_coords[{i, {}}] = pose[{i,{},{}}]*output_coords
      end

      template_x = template_coords[{{},1}] --template_coords[1]
      template_y = template_coords[{{},2}] --template_coords[2]

      local template_x_size = template:size()[2]
      local template_y_size = template:size()[3]
      local template_x_after_offset = template_x + template_x_size / 2
      local template_y_after_offset = template_y + template_y_size / 2



      local x_high_coeff = torch.Tensor(bsize)
      local y_high_coeff = torch.Tensor(bsize)

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

      for ii=1, bsize do
        local x_low_ii = x_low[ii]; local y_low_ii = y_low[ii];
        local x_high_ii = x_high[ii]; local y_high_ii = y_high[ii];
        local ratio_xy = (x_low_ii-x_high_ii)*(y_low_ii-y_high_ii)
        --------------------------- Template gradient -------------------------------
        if x_low_ii >= 1 and x_low_ii <= template:size()[2] and y_low_ii >= 1 and y_low_ii <= template:size()[2] then
          gradTemplate[ii][x_low_ii][y_low_ii] = gradTemplate[ii][x_low_ii][y_low_ii]
                                                   + ( (((template_x_after_offset[ii]-x_high_ii)*(template_y_after_offset[ii]-y_high_ii))/ ratio_xy )
                                                          * gradOutput[ii][output_x][output_y] )
          --print('ox:',output_x, 'oy', output_y, 'gradT', gradTemplate[ii][x_low_ii][y_low_ii])
        end

        if x_low_ii >= 1 and x_low_ii <= template:size()[2] and y_high_ii >= 1 and y_high_ii <= template:size()[2] then
          --print(gradTemplate:size())
          gradTemplate[ii][x_low_ii][y_high_ii] = gradTemplate[ii][x_low_ii][y_high_ii] + ( -(((template_x_after_offset[ii]-x_high_ii)*(template_y_after_offset[ii]-y_low_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print('ox:',output_x, 'oy', output_y, 'gradT', gradTemplate[ii][x_low_ii][y_high_ii])
        end

        if x_high_ii >= 1 and x_high_ii <= template:size()[2] and y_low_ii >= 1 and y_low_ii <= template:size()[2] then
          gradTemplate[ii][x_high_ii][y_low_ii] = gradTemplate[ii][x_high_ii][y_low_ii] + ( -(((template_x_after_offset[ii]-x_low_ii)*(template_y_after_offset[ii]-y_high_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print('ox:',output_x, 'oy', output_y, 'gradT', gradTemplate[ii][x_high_ii][y_low_ii])
        end

        if x_high_ii >= 1 and x_high_ii <= template:size()[2] and y_high_ii >= 1 and y_high_ii <= template:size()[2] then
          gradTemplate[ii][x_high_ii][y_high_ii] = gradTemplate[ii][x_high_ii][y_high_ii] + ( (((template_x_after_offset[ii]-x_low_ii)*(template_y_after_offset[ii]-y_low_ii))/ ratio_xy ) * gradOutput[ii][output_x][output_y] )
          --print('ox:',output_x, 'oy', output_y, 'gradT', gradTemplate[ii][x_high_ii][y_high_ii])
        end


      end

      --]]
      ------------------- Pose Gradient ---------------------------
      -- syms a b c d e f x y xt yt x1 x2 y1 y2 t11 t12 t21 t22
      -- syms pose_1_1 pose_1_2 pose_1_3 pose_2_1 pose_2_2 pose_2_3
      -- xt = pose_1_1*x + pose_1_2*y + pose_1_3;
      -- yt = pose_2_1*x + pose_2_2*y + pose_2_3;
      -- Ixy = (1./((x2-x1)*(y2-y1))) * ( (t11*(x2-xt)*(y2-yt)) + (t21*(xt-x1)*(y2-yt)) + (t12*(x2-xt)*(yt-y1)) + (t22*(xt-x1)*(yt-y1)) )

      for batchIndex = 1, bsize do
        -- if x_low[batchIndex] == template_x_after_offset[batchIndex] then
        --   x_low[batchIndex]  = x_low[batchIndex] + 1e-5
        --   -- x_high[batchIndex] = x_high[batchIndex] + 1e-1
        -- end

        -- if y_low[batchIndex] == template_y_after_offset[batchIndex] then
        --   y_low[batchIndex]  = y_low[batchIndex] + 1e-5
        --   -- y_high[batchIndex] = y_high[batchIndex] + 1e-1
        -- end

        ACR_helper:updateGradPoseAtLocation(batchIndex, output_x,
                                            output_y,
                                            template_x_after_offset[batchIndex],
                                            template_y_after_offset[batchIndex],
                                            x_low[batchIndex],
                                            x_high[batchIndex],
                                            y_low[batchIndex],
                                            y_high[batchIndex],
                                            template[batchIndex],
                                            pose[batchIndex],
                                            gradPose[batchIndex],
                                            gradOutput[batchIndex],
                                            intensity[batchIndex])


        -- scale = 1
        -- if x_low[batchIndex] == template_x_after_offset[batchIndex] and y_low[batchIndex] == template_y_after_offset[batchIndex] then
        --   print("scaling X and Y")
        --   scale = 1/3
        -- elseif x_low[batchIndex] == template_x_after_offset[batchIndex] or y_low[batchIndex] == template_y_after_offset[batchIndex] then
        --   print("scaling X or Y")
        --   scale = 1/2
        -- end

        -- ACR_helper:updateGradPoseAtLocation(output_x,
        --                                     output_y,
        --                                     template_x_after_offset[batchIndex],
        --                                     template_y_after_offset[batchIndex],
        --                                     x_low[batchIndex],
        --                                     x_high[batchIndex],
        --                                     y_low[batchIndex],
        --                                     y_high[batchIndex],
        --                                     template[batchIndex],
        --                                     pose[batchIndex],
        --                                     gradPose[batchIndex],
        --                                     gradOutput[batchIndex],
        --                                     intensity[batchIndex],
        --                                     scale)

        -- if x_low[batchIndex] == template_x_after_offset[batchIndex] then
        --   ACR_helper:updateGradPoseAtLocation(output_x,
        --                                       output_y,
        --                                       template_x_after_offset[batchIndex],
        --                                       template_y_after_offset[batchIndex],
        --                                       x_low[batchIndex] - 1,
        --                                       x_high[batchIndex] - 1,
        --                                       y_low[batchIndex],
        --                                       y_high[batchIndex],
        --                                       template[batchIndex],
        --                                       pose[batchIndex],
        --                                       gradPose[batchIndex],
        --                                       gradOutput[batchIndex],
        --                                       intensity[batchIndex],
        --                                       scale)
        -- end

        -- if y_low[batchIndex] == template_y_after_offset[batchIndex] then
        --   ACR_helper:updateGradPoseAtLocation(output_x,
        --                                       output_y,
        --                                       template_x_after_offset[batchIndex],
        --                                       template_y_after_offset[batchIndex],
        --                                       x_low[batchIndex],
        --                                       x_high[batchIndex],
        --                                       y_low[batchIndex] - 1,
        --                                       y_high[batchIndex] - 1,
        --                                       template[batchIndex],
        --                                       pose[batchIndex],
        --                                       gradPose[batchIndex],
        --                                       gradOutput[batchIndex],
        --                                       intensity[batchIndex],
        --                                       scale)
        -- end
      end
    end
  end

  if tostring(torch.sum(gradPose)) == tostring(0/0) then
    print("ERROR!")
  end

  return gradTemplate, gradPose
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

function ACR_helper:getSingleTemplateValue(template, template_x, template_y)
  if template_x < 1 or template_x > template:size()[1] or template_y < 1 or template_y > template:size()[2] then
    return 0
  else
    return template[template_x][template_y]
  end
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