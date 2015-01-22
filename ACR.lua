-- ACR module


local ACR, Parent = torch.class('nn.ACR', 'nn.Module')

function ACR:__init(bsize, output_width)
  Parent.__init(self)
  self.bsize = bsize
  self.output = torch.zeros(bsize, output_width, output_width)
end


function ACR:updateOutput(input)
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)

  for output_x = 1, self.output:size()[2] do
    for output_y = 1, self.output:size()[3] do
      output_coords = torch.Tensor({output_x, output_y, 1})

      --template_coords = pose * output_coords
      --self.output[output_x][output_y] = getInterpolatedTemplateValue(
      --                                          template,
      --                                          template_coords[1],
      --                                          template_coords[2])

      template_coords = torch.zeros(bsize, 3)
      for i=1, bsize do
        template_coords[{i, {}}] = pose[{bsize,{},{}}]*output_coords
      end

      self.output[{{}, output_x, output_y}] = torch.cmul(intensity, self:getInterpolatedTemplateValue(
                                                  template,
                                                  template_coords[{{},1}], -- x
                                                  template_coords[{{},2}])) -- y

    end
  end

  --self.output = self.output * intensity[1]

  return self.output
end

function ACR:updateGradInput(input, gradOutput)
  local bsize = self.bsize
  local template = input[1]:reshape(bsize, math.sqrt(input[1]:size()[2]), math.sqrt(input[1]:size()[2]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{},{1,9}}]:reshape(bsize,3,3)
  local intensity = iGeoPose[{{}, {10}}]:reshape(bsize)


  self.gradTemplate = self.gradTemplate or torch.Tensor(template:size())
  self.gradTemplate:fill(0)
  self.gradPose = torch.Tensor(pose:size())
  self.gradPose:fill(0)

  for output_x = 1, self.output:size()[2] do
    for output_y = 1, self.output:size()[3] do
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

      local x_high_coeff = torch.Tensor(self.bsize)
      x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
      x_low_coeff  =  -x_high_coeff + 1

      local y_high_coeff = torch.Tensor(self.bsize)
      y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1
      y_low_coeff  =  -y_high_coeff + 1

      x_low  = torch.floor(template_x)
      x_high = x_low + 1
      y_low  = torch.floor(template_y)
      y_high = y_low + 1

      for i=1,bsize do
        -----------------------------------------------------------------------------
        --------------------------- Template gradient -------------------------------
        -----------------------------------------------------------------------------
        -- calculate the derivatives for the template
        x_vec = torch.Tensor({x_low_coeff[i], x_high_coeff[i]})
        y_vec = torch.Tensor({y_low_coeff[i], y_high_coeff[i]})

        -- outer product
        dOutdPose = torch.ger(x_vec, y_vec)
        for i, x in ipairs({x_low[i], x_high[i]}) do
          for j, y in ipairs({y_low[i], y_high[i]}) do
              if x >= 1 and x <= template:size()[2]
                and y >= 1 and y <= template:size()[3] then
                self.gradTemplate[i][x][y] = self.gradTemplate[i][x][y] + dOutdPose[i][j]
              end
          end
        end
      end


      --[[template_x = template_x - 1/2
      template_y = template_y - 1/2
      x_high_coeff = template_x % 1
      x_low_coeff  = 1 - x_high_coeff

      y_high_coeff = template_y % 1
      y_low_coeff  = 1 - y_high_coeff

      x_low  = math.floor(template_x)
      x_high = x_low + 1
      y_low  = math.floor(template_y)
      y_high = y_low + 1

      -- calculate the derivatives for the template
      x_vec = torch.Tensor({x_low_coeff, x_high_coeff})
      y_vec = torch.Tensor({y_low_coeff, y_high_coeff})

      -- outer product
      dOutdPose = torch.ger(x_vec, y_vec)
      for i, x in ipairs({x_low, x_high}) do
        for j, y in ipairs({y_low, y_high}) do
            if x >= 1 and x <= template:size()[1]
              and y >= 1 and y <= template:size()[2] then
              self.gradTemplate[x][y] = self.gradTemplate[x][y] + dOutdPose[i][j]
            end
        end
      end
      --]]

      -- self.gradTemplate[{{x_low, x_high}, {y_low, y_high}}]:addr(gradOutput[output_x][output_y], )


      -- calculate the derivatives for the pose

      -- should be equivalent to below
      -- self:getTemplateValue(template, x_high, y_high) *pose[2][3]*output_x - self:getTemplateValue(template, x_high, y_low) *pose[2][3]*output_x - self:getTemplateValue(template, x_low, y_high) *pose[2][3]*output_x + self:getTemplateValue(template, x_low, y_low) *pose[2][3]*output_x + self:getTemplateValue(template, x_high, y_low) *output_x*y_high - self:getTemplateValue(template, x_high, y_high) *output_x*y_low - self:getTemplateValue(template, x_low, y_low) *output_x*y_high + self:getTemplateValue(template, x_low, y_high) *output_x*y_low + self:getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x^2 - self:getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x^2 - self:getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x^2 + self:getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x^2 + self:getTemplateValue(template, x_high, y_high) *pose[2][2]*output_x*output_y - self:getTemplateValue(template, x_high, y_low) *pose[2][2]*output_x*output_y - self:getTemplateValue(template, x_low, y_high) *pose[2][2]*output_x*output_y + self:getTemplateValue(template, x_low, y_low) *pose[2][2]*output_x*output_y
      -- self:getTemplateValue(template, x_high, y_high) *pose[2][3]*output_y - self:getTemplateValue(template, x_high, y_low) *pose[2][3]*output_y - self:getTemplateValue(template, x_low, y_high) *pose[2][3]*output_y + self:getTemplateValue(template, x_low, y_low) *pose[2][3]*output_y + self:getTemplateValue(template, x_high, y_low) *output_y*y_high - self:getTemplateValue(template, x_high, y_high) *output_y*y_low - self:getTemplateValue(template, x_low, y_low) *output_y*y_high + self:getTemplateValue(template, x_low, y_high) *output_y*y_low + self:getTemplateValue(template, x_high, y_high) *pose[2][2]*output_y^2 - self:getTemplateValue(template, x_high, y_low) *pose[2][2]*output_y^2 - self:getTemplateValue(template, x_low, y_high) *pose[2][2]*output_y^2 + self:getTemplateValue(template, x_low, y_low) *pose[2][2]*output_y^2 + self:getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x*output_y - self:getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x*output_y - self:getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x*output_y + self:getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x*output_y
      -- self:getTemplateValue(template, x_high, y_high) *pose[2][3] - self:getTemplateValue(template, x_high, y_low) *pose[2][3] - self:getTemplateValue(template, x_low, y_high) *pose[2][3] + self:getTemplateValue(template, x_low, y_low) *pose[2][3] + self:getTemplateValue(template, x_high, y_low) *y_high - self:getTemplateValue(template, x_high, y_high) *y_low - self:getTemplateValue(template, x_low, y_low) *y_high + self:getTemplateValue(template, x_low, y_high) *y_low + self:getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x - self:getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x - self:getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x + self:getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x + self:getTemplateValue(template, x_high, y_high) *pose[2][2]*output_y - self:getTemplateValue(template, x_high, y_low) *pose[2][2]*output_y - self:getTemplateValue(template, x_low, y_high) *pose[2][2]*output_y + self:getTemplateValue(template, x_low, y_low) *pose[2][2]*output_y
      -- self:getTemplateValue(template, x_high, y_high) *pose[1][3]*output_x - self:getTemplateValue(template, x_high, y_low) *pose[1][3]*output_x - self:getTemplateValue(template, x_low, y_high) *pose[1][3]*output_x + self:getTemplateValue(template, x_low, y_low) *pose[1][3]*output_x - self:getTemplateValue(template, x_high, y_high) *output_x*x_low + self:getTemplateValue(template, x_low, y_high) *output_x*x_high + self:getTemplateValue(template, x_high, y_low) *output_x*x_low - self:getTemplateValue(template, x_low, y_low) *output_x*x_high + self:getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x^2 - self:getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x^2 - self:getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x^2 + self:getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x^2 + self:getTemplateValue(template, x_high, y_high) *pose[1][2]*output_x*output_y - self:getTemplateValue(template, x_high, y_low) *pose[1][2]*output_x*output_y - self:getTemplateValue(template, x_low, y_high) *pose[1][2]*output_x*output_y + self:getTemplateValue(template, x_low, y_low) *pose[1][2]*output_x*output_y
      -- self:getTemplateValue(template, x_high, y_high) *pose[1][3]*output_y - self:getTemplateValue(template, x_high, y_low) *pose[1][3]*output_y - self:getTemplateValue(template, x_low, y_high) *pose[1][3]*output_y + self:getTemplateValue(template, x_low, y_low) *pose[1][3]*output_y - self:getTemplateValue(template, x_high, y_high) *x_low*output_y + self:getTemplateValue(template, x_low, y_high) *x_high*output_y + self:getTemplateValue(template, x_high, y_low) *x_low*output_y - self:getTemplateValue(template, x_low, y_low) *x_high*output_y + self:getTemplateValue(template, x_high, y_high) *pose[1][2]*output_y^2 - self:getTemplateValue(template, x_high, y_low) *pose[1][2]*output_y^2 - self:getTemplateValue(template, x_low, y_high) *pose[1][2]*output_y^2 + self:getTemplateValue(template, x_low, y_low) *pose[1][2]*output_y^2 + self:getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x*output_y - self:getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x*output_y - self:getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x*output_y + self:getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x*output_y
      -- self:getTemplateValue(template, x_high, y_high) *pose[1][3] - self:getTemplateValue(template, x_high, y_low) *pose[1][3] - self:getTemplateValue(template, x_low, y_high) *pose[1][3] + self:getTemplateValue(template, x_low, y_low) *pose[1][3] - self:getTemplateValue(template, x_high, y_high) *x_low + self:getTemplateValue(template, x_low, y_high) *x_high + self:getTemplateValue(template, x_high, y_low) *x_low - self:getTemplateValue(template, x_low, y_low) *x_high + self:getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x - self:getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x - self:getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x + self:getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x + self:getTemplateValue(template, x_high, y_high) *pose[1][2]*output_y - self:getTemplateValue(template, x_high, y_low) *pose[1][2]*output_y - self:getTemplateValue(template, x_low, y_high) *pose[1][2]*output_y + self:getTemplateValue(template, x_low, y_low) *pose[1][2]*output_y

      -- add dCost/dOut(x,y) * dOut(x,y)/dPose for this (x,y)
      self.gradPose[{{},1,1}] = self.gradPose[{{},1,1}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( (self:getTemplateValue(template, x_high, y_high)*output_x - self:getTemplateValue(template, x_low, y_high)*output_x), (pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((self:getTemplateValue(template, x_high, y_low)*output_x - self:getTemplateValue(template, x_low, y_low)*output_x), (pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))

      self.gradPose[{{},1,2}] = self.gradPose[{{},1,2}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul((self:getTemplateValue(template, x_high, y_high)*output_y - self:getTemplateValue(template, x_low, y_high)*output_y), (pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((self:getTemplateValue(template, x_high, y_low)*output_y - self:getTemplateValue(template, x_low, y_low)*output_y),(pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))

      self.gradPose[{{},1,3}] = self.gradPose[{{},1,3}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( (self:getTemplateValue(template, x_high, y_high) - self:getTemplateValue(template, x_low, y_high)),(pose[{{},2,3}] - y_low + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))) - torch.cmul((self:getTemplateValue(template, x_high, y_low) - self:getTemplateValue(template, x_low, y_low)),(pose[{{},2,3}] - y_high + pose[{{},2,1}]*output_x + pose[{{},2,2}]*output_y))


      self.gradPose[{{},2,1}] = self.gradPose[{{},2,1}] +  torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( self:getTemplateValue(template, x_high, y_high), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(self:getTemplateValue(template, x_low, y_high), (pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_x - (  torch.cmul(self:getTemplateValue(template, x_high, y_low),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(self:getTemplateValue(template, x_low, y_low),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_x


      self.gradPose[{{},2,2}] = self.gradPose[{{},2,2}] + torch.cmul(gradOutput[{{},output_x,output_y}],(torch.cmul( self:getTemplateValue(template, x_high, y_high),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(self:getTemplateValue(template, x_low, y_high),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))))*output_y - ( torch.cmul(self:getTemplateValue(template, x_high, y_low),(pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(self:getTemplateValue(template, x_low, y_low),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)))*output_y

      self.gradPose[{{},2,3}] = self.gradPose[{{},2,3}] + torch.cmul(gradOutput[{{},output_x,output_y}], torch.cmul( self:getTemplateValue(template, x_high, y_high), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))) - torch.cmul( self:getTemplateValue(template, x_low, y_high),(pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) - torch.cmul(self:getTemplateValue(template, x_high, y_low), (pose[{{},1,3}] - x_low + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y)) + torch.cmul(self:getTemplateValue(template, x_low, y_low), (pose[{{},1,3}] - x_high + pose[{{},1,1}]*output_x + pose[{{},1,2}]*output_y))

    end
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

function ACR:getTemplateValue(template, template_x, template_y)
  template_x_size = template:size()[2] + 1
  template_y_size = template:size()[3] + 1
  output_x = torch.floor(template_x + template_x_size / 2)
  output_y = torch.floor(template_y + template_y_size / 2)

  res = torch.zeros(self.bsize)
  for i = 1,self.bsize do
    if output_x[i] < 1 or output_x[i] > template:size()[2]
      or output_y[i] < 1 or output_y[i] > template:size()[3] then
      res[i] = 0
    else
      res[i] = template[i][output_x[i]][output_y[i]]
    end
  end
  return res
end

function ACR:getInterpolatedTemplateValue(template, template_x, template_y)
  template_x = template_x - 1/2
  template_y = template_y - 1/2

  local x_high_coeff = torch.Tensor(self.bsize)
  x_high_coeff:map(template_x, function(xhc, txx) return math.fmod(txx, 1) end) --x_high_coeff = template_x % 1
  x_low_coeff  =  -x_high_coeff + 1

  local y_high_coeff = torch.Tensor(self.bsize)
  y_high_coeff:map(template_y, function(yhc, tyy) return math.fmod(tyy,1) end) --y_high_coeff = template_y % 1
  y_low_coeff  =  -y_high_coeff + 1

  x_low  = torch.floor(template_x)
  x_high = x_low + 1
  y_low  = torch.floor(template_y)
  y_high = y_low + 1

  --return self:getTemplateValue(template, x_low,  y_low)  * x_low_coeff  * y_low_coeff  +
  --       self:getTemplateValue(template, x_high, y_low)  * x_high_coeff * y_low_coeff  +
  --       self:getTemplateValue(template, x_low,  y_high) * x_low_coeff  * y_high_coeff +
  --       self:getTemplateValue(template, x_high, y_high) * x_high_coeff * y_high_coeff

  return torch.cmul(self:getTemplateValue(template, x_low,  y_low) , torch.cmul(x_low_coeff  , y_low_coeff))  +
         torch.cmul(self:getTemplateValue(template, x_high, y_low) , torch.cmul(x_high_coeff , y_low_coeff )) +
         torch.cmul(self:getTemplateValue(template, x_low,  y_high), torch.cmul(x_low_coeff  , y_high_coeff ))+
         torch.cmul(self:getTemplateValue(template, x_high, y_high), torch.cmul(x_high_coeff , y_high_coeff ))
end










