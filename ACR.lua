-- ACR module


local ACR, Parent = torch.class('nn.ACR', 'nn.Module')

function ACR:__init(output_width)
  Parent.__init(self)
  self.output = torch.zeros(output_width, output_width)
end


function ACR:updateOutput(input)
  local template = input[1]:reshape(math.sqrt(input[1]:size()[1]), math.sqrt(input[1]:size()[1]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{1,9}}]:reshape(3,3)
  local intensity = iGeoPose[{{10}}]

  for output_x = 1, self.output:size()[1] do
    for output_y = 1, self.output:size()[2] do
      output_coords = torch.Tensor({output_x, output_y, 1})
      template_coords = pose * output_coords
      self.output[output_x][output_y] = getInterpolatedTemplateValue(
                                                template,
                                                template_coords[1],
                                                template_coords[2])
    end
  end
  self.output = self.output * intensity[1]
  return self.output
end

function ACR:updateGradInput(input, gradOutput)
  local template = input[1]:reshape(math.sqrt(input[1]:size()[1]), math.sqrt(input[1]:size()[1]))
  local iGeoPose = input[2]
  local pose = iGeoPose[{{1,9}}]:reshape(3,3)
  local intensity = iGeoPose[{{10}}]

  self.gradTemplate = self.gradTemplate or torch.Tensor(template:size())
  self.gradTemplate:fill(0)
  self.gradPose = torch.Tensor(pose:size())
  self.gradPose:fill(0)

  for output_x = 1, self.output:size()[1] do
    for output_y = 1, self.output:size()[2] do
      -- calculate the correspondence between template and output
      output_coords = torch.Tensor({output_x, output_y, 1})
      template_coords = pose * output_coords
      template_x = template_coords[1]
      template_y = template_coords[2]

      template_x = template_x - 1/2
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

      -- self.gradTemplate[{{x_low, x_high}, {y_low, y_high}}]:addr(gradOutput[output_x][output_y], )


      -- calculate the derivatives for the pose

      -- should be equivalent to below
      -- getTemplateValue(template, x_high, y_high) *pose[2][3]*output_x - getTemplateValue(template, x_high, y_low) *pose[2][3]*output_x - getTemplateValue(template, x_low, y_high) *pose[2][3]*output_x + getTemplateValue(template, x_low, y_low) *pose[2][3]*output_x + getTemplateValue(template, x_high, y_low) *output_x*y_high - getTemplateValue(template, x_high, y_high) *output_x*y_low - getTemplateValue(template, x_low, y_low) *output_x*y_high + getTemplateValue(template, x_low, y_high) *output_x*y_low + getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x^2 - getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x^2 - getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x^2 + getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x^2 + getTemplateValue(template, x_high, y_high) *pose[2][2]*output_x*output_y - getTemplateValue(template, x_high, y_low) *pose[2][2]*output_x*output_y - getTemplateValue(template, x_low, y_high) *pose[2][2]*output_x*output_y + getTemplateValue(template, x_low, y_low) *pose[2][2]*output_x*output_y
      -- getTemplateValue(template, x_high, y_high) *pose[2][3]*output_y - getTemplateValue(template, x_high, y_low) *pose[2][3]*output_y - getTemplateValue(template, x_low, y_high) *pose[2][3]*output_y + getTemplateValue(template, x_low, y_low) *pose[2][3]*output_y + getTemplateValue(template, x_high, y_low) *output_y*y_high - getTemplateValue(template, x_high, y_high) *output_y*y_low - getTemplateValue(template, x_low, y_low) *output_y*y_high + getTemplateValue(template, x_low, y_high) *output_y*y_low + getTemplateValue(template, x_high, y_high) *pose[2][2]*output_y^2 - getTemplateValue(template, x_high, y_low) *pose[2][2]*output_y^2 - getTemplateValue(template, x_low, y_high) *pose[2][2]*output_y^2 + getTemplateValue(template, x_low, y_low) *pose[2][2]*output_y^2 + getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x*output_y - getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x*output_y - getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x*output_y + getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x*output_y
      -- getTemplateValue(template, x_high, y_high) *pose[2][3] - getTemplateValue(template, x_high, y_low) *pose[2][3] - getTemplateValue(template, x_low, y_high) *pose[2][3] + getTemplateValue(template, x_low, y_low) *pose[2][3] + getTemplateValue(template, x_high, y_low) *y_high - getTemplateValue(template, x_high, y_high) *y_low - getTemplateValue(template, x_low, y_low) *y_high + getTemplateValue(template, x_low, y_high) *y_low + getTemplateValue(template, x_high, y_high) *pose[2][1]*output_x - getTemplateValue(template, x_high, y_low) *pose[2][1]*output_x - getTemplateValue(template, x_low, y_high) *pose[2][1]*output_x + getTemplateValue(template, x_low, y_low) *pose[2][1]*output_x + getTemplateValue(template, x_high, y_high) *pose[2][2]*output_y - getTemplateValue(template, x_high, y_low) *pose[2][2]*output_y - getTemplateValue(template, x_low, y_high) *pose[2][2]*output_y + getTemplateValue(template, x_low, y_low) *pose[2][2]*output_y
      -- getTemplateValue(template, x_high, y_high) *pose[1][3]*output_x - getTemplateValue(template, x_high, y_low) *pose[1][3]*output_x - getTemplateValue(template, x_low, y_high) *pose[1][3]*output_x + getTemplateValue(template, x_low, y_low) *pose[1][3]*output_x - getTemplateValue(template, x_high, y_high) *output_x*x_low + getTemplateValue(template, x_low, y_high) *output_x*x_high + getTemplateValue(template, x_high, y_low) *output_x*x_low - getTemplateValue(template, x_low, y_low) *output_x*x_high + getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x^2 - getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x^2 - getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x^2 + getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x^2 + getTemplateValue(template, x_high, y_high) *pose[1][2]*output_x*output_y - getTemplateValue(template, x_high, y_low) *pose[1][2]*output_x*output_y - getTemplateValue(template, x_low, y_high) *pose[1][2]*output_x*output_y + getTemplateValue(template, x_low, y_low) *pose[1][2]*output_x*output_y
      -- getTemplateValue(template, x_high, y_high) *pose[1][3]*output_y - getTemplateValue(template, x_high, y_low) *pose[1][3]*output_y - getTemplateValue(template, x_low, y_high) *pose[1][3]*output_y + getTemplateValue(template, x_low, y_low) *pose[1][3]*output_y - getTemplateValue(template, x_high, y_high) *x_low*output_y + getTemplateValue(template, x_low, y_high) *x_high*output_y + getTemplateValue(template, x_high, y_low) *x_low*output_y - getTemplateValue(template, x_low, y_low) *x_high*output_y + getTemplateValue(template, x_high, y_high) *pose[1][2]*output_y^2 - getTemplateValue(template, x_high, y_low) *pose[1][2]*output_y^2 - getTemplateValue(template, x_low, y_high) *pose[1][2]*output_y^2 + getTemplateValue(template, x_low, y_low) *pose[1][2]*output_y^2 + getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x*output_y - getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x*output_y - getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x*output_y + getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x*output_y
      -- getTemplateValue(template, x_high, y_high) *pose[1][3] - getTemplateValue(template, x_high, y_low) *pose[1][3] - getTemplateValue(template, x_low, y_high) *pose[1][3] + getTemplateValue(template, x_low, y_low) *pose[1][3] - getTemplateValue(template, x_high, y_high) *x_low + getTemplateValue(template, x_low, y_high) *x_high + getTemplateValue(template, x_high, y_low) *x_low - getTemplateValue(template, x_low, y_low) *x_high + getTemplateValue(template, x_high, y_high) *pose[1][1]*output_x - getTemplateValue(template, x_high, y_low) *pose[1][1]*output_x - getTemplateValue(template, x_low, y_high) *pose[1][1]*output_x + getTemplateValue(template, x_low, y_low) *pose[1][1]*output_x + getTemplateValue(template, x_high, y_high) *pose[1][2]*output_y - getTemplateValue(template, x_high, y_low) *pose[1][2]*output_y - getTemplateValue(template, x_low, y_high) *pose[1][2]*output_y + getTemplateValue(template, x_low, y_low) *pose[1][2]*output_y

      -- add dCost/dOut(x,y) * dOut(x,y)/dPose for this (x,y)
      self.gradPose[1][1] = self.gradPose[1][1] + gradOutput[output_x][output_y] * (getTemplateValue(template, x_high, y_high)*output_x - getTemplateValue(template, x_low, y_high)*output_x)*(pose[2][3] - y_low + pose[2][1]*output_x + pose[2][2]*output_y) - (getTemplateValue(template, x_high, y_low)*output_x - getTemplateValue(template, x_low, y_low)*output_x)*(pose[2][3] - y_high + pose[2][1]*output_x + pose[2][2]*output_y)
      self.gradPose[1][2] = self.gradPose[1][2] + gradOutput[output_x][output_y] * (getTemplateValue(template, x_high, y_high)*output_y - getTemplateValue(template, x_low, y_high)*output_y)*(pose[2][3] - y_low + pose[2][1]*output_x + pose[2][2]*output_y) - (getTemplateValue(template, x_high, y_low)*output_y - getTemplateValue(template, x_low, y_low)*output_y)*(pose[2][3] - y_high + pose[2][1]*output_x + pose[2][2]*output_y)
      self.gradPose[1][3] = self.gradPose[1][3] + gradOutput[output_x][output_y] * (getTemplateValue(template, x_high, y_high) - getTemplateValue(template, x_low, y_high))*(pose[2][3] - y_low + pose[2][1]*output_x + pose[2][2]*output_y) - (getTemplateValue(template, x_high, y_low) - getTemplateValue(template, x_low, y_low))*(pose[2][3] - y_high + pose[2][1]*output_x + pose[2][2]*output_y)
      self.gradPose[2][1] = self.gradPose[2][1] + gradOutput[output_x][output_y] * output_x*(getTemplateValue(template, x_high, y_high)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_low, y_high)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y)) - output_x*(getTemplateValue(template, x_high, y_low)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_low, y_low)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y))
      self.gradPose[2][2] = self.gradPose[2][2] + gradOutput[output_x][output_y] * output_y*(getTemplateValue(template, x_high, y_high)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_low, y_high)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y)) - output_y*(getTemplateValue(template, x_high, y_low)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_low, y_low)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y))
      self.gradPose[2][3] = self.gradPose[2][3] + gradOutput[output_x][output_y] * getTemplateValue(template, x_high, y_high)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_low, y_high)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y) - getTemplateValue(template, x_high, y_low)*(pose[1][3] - x_low + pose[1][1]*output_x + pose[1][2]*output_y) + getTemplateValue(template, x_low, y_low)*(pose[1][3] - x_high + pose[1][1]*output_x + pose[1][2]*output_y)
    end
  end
  self.gradTemplate = self.gradTemplate * intensity[1]
  self.gradPose = self.gradPose * intensity[1]
  self.gradPose = self.gradPose:reshape(9)
  self.gradPose:resize(10)
  self.gradPose[10] = gradOutput:sum()
  self.gradInput = {self.gradTemplate, self.gradPose}

  -- if self.gradInput:nElement() == 0 then
  --   self.gradInput = torch.zeros(input:size())
  -- end
  return self.gradInput
end

-- function getTemplateGradient(template, pose, output_x, output_y)

-- end

function getTemplateValue(template, template_x, template_y)
  template_x_size = template:size()[1] + 1
  template_y_size = template:size()[2] + 1
  output_x = math.floor(template_x + template_x_size / 2)
  output_y = math.floor(template_y + template_y_size / 2)
  if output_x < 1 or output_x > template:size()[1]
    or output_y < 1 or output_y > template:size()[2] then
    return 0
  else
    return template[output_x][output_y]
  end
end

function getInterpolatedTemplateValue(template, template_x, template_y)
  template_x = template_x - 1/2
  template_y = template_y - 1/2
  x_high_coeff = template_x % 1
  x_low_coeff  = 1 - x_high_coeff
  y_high_coeff = template_y % 1
  y_low_coeff  = 1 - y_high_coeff

  x_low  = math.floor(template_x)
  x_high = x_low + 1
  y_low  = math.floor(template_y)
  y_high = y_low + 1

  return getTemplateValue(template, x_low,  y_low)  * x_low_coeff  * y_low_coeff  +
         getTemplateValue(template, x_high, y_low)  * x_high_coeff * y_low_coeff  +
         getTemplateValue(template, x_low,  y_high) * x_low_coeff  * y_high_coeff +
         getTemplateValue(template, x_high, y_high) * x_high_coeff * y_high_coeff
end










