
local ACR_helper = {}


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
      x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
      x_low_coeff  =  -x_high_coeff + 1

      local y_high_coeff = torch.Tensor(bsize)
      y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1
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

      -- add dCost/dOut(x,y) * dOut(x,y)/dPose for this (x,y)
      gradPose[{{},1,1}] = gradPose[{{},1,1}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( (ACR_helper:getTemplateValue(bsize, template, x_high, y_high)*output_x - ACR_helper:getTemplateValue(bsize, template, x_low, y_high)*output_x), (pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((ACR_helper:getTemplateValue(bsize, template, x_high, y_low)*output_x - ACR_helper:getTemplateValue(bsize, template, x_low, y_low)*output_x), (pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))

      gradPose[{{},1,2}] = gradPose[{{},1,2}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul((ACR_helper:getTemplateValue(bsize, template, x_high, y_high)*output_y - ACR_helper:getTemplateValue(bsize, template, x_low, y_high)*output_y), (pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((ACR_helper:getTemplateValue(bsize, template, x_high, y_low)*output_y - ACR_helper:getTemplateValue(bsize, template, x_low, y_low)*output_y),(pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))

      gradPose[{{},1,3}] = gradPose[{{},1,3}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( (ACR_helper:getTemplateValue(bsize, template, x_high, y_high) - ACR_helper:getTemplateValue(bsize, template, x_low, y_high)),(pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((ACR_helper:getTemplateValue(bsize, template, x_high, y_low) - ACR_helper:getTemplateValue(bsize, template, x_low, y_low)),(pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))


      gradPose[{{},2,1}] = gradPose[{{},2,1}] +  torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( ACR_helper:getTemplateValue(bsize, template, x_high, y_high), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low, y_high), (pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_x - (  torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_low),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low, y_low),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_x


      gradPose[{{},2,2}] = gradPose[{{},2,2}] + torch.cmul(gradOutput[{{},output_x,output_y}],(torch.cmul( ACR_helper:getTemplateValue(bsize, template, x_high, y_high),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low, y_high),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))))*output_y - ( torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_low),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low, y_low),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_y

      gradPose[{{},2,3}] = gradPose[{{},2,3}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( ACR_helper:getTemplateValue(bsize, template, x_high, y_high), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))) - torch.cmul( ACR_helper:getTemplateValue(bsize, template, x_low, y_high),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_high, y_low), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) + torch.cmul(ACR_helper:getTemplateValue(bsize, template, x_low, y_low), (pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))
    end
  end
    return gradTemplate, gradPose
    --]]
end


function ACR_helper:getTemplateValue(bsize, template, template_x, template_y)
  template_x_size = template:size()[2] + 1
  template_y_size = template:size()[3] + 1
  output_x = torch.floor(template_x + template_x_size / 2)
  output_y = torch.floor(template_y + template_y_size / 2)

  res = torch.zeros(bsize)
  for i = 1,bsize do
    if output_x[i] < 1 or output_x[i] > template:size()[2]
      or output_y[i] < 1 or output_y[i] > template:size()[3] then
      res[i] = 0
    else
      res[i] = template[i][output_x[i]][output_y[i]]
    end
  end
  return res
end

function ACR_helper:getInterpolatedTemplateValue(bsize, template, template_x, template_y)
  template_x = template_x - 1/2
  template_y = template_y - 1/2

  local x_high_coeff = torch.Tensor(bsize)
  x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
  x_low_coeff  =  -x_high_coeff + 1

  local y_high_coeff = torch.Tensor(bsize)
  y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1
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