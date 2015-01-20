-- ACR module


local ACR, Parent = torch.class('nn.ACR', 'nn.Module')

function ACR:__init(output_width)
  Parent.__init(self)
  self.output = torch.zeros(output_width, output_width)
end


function ACR:updateOutput(input)
  local template = input[1]
  local pose = input[2]

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
  return self.output
end

function getTemplateValue(template, template_x, template_y)
  template_x_size = template:size()[1] + 1
  template_y_size = template:size()[2] + 1
  x = math.floor(template_x + template_x_size / 2)
  y = math.floor(template_y + template_y_size / 2)
  if x < 1 or x > template:size()[1]
    or y < 1 or y > template:size()[2] then
    return 0
  else
    return template[x][y]
  end
end

function getInterpolatedTemplateValue(template, template_x, template_y)
  template_x = template_x - 1/2
  template_y = template_y - 1/2
  x_high_coeff = template_x % 1
  x_low_coeff  = 1 - x_high_coeff
  y_high_coeff = template_y % 1
  y_low_coeff  = 1 - y_high_coeff

  x_high = math.floor(template_x + 1)
  x_low  = math.floor(template_x)
  y_high = math.floor(template_y + 1)
  y_low  = math.floor(template_y)

  return getTemplateValue(template, x_low,  y_low)  * x_low_coeff  * y_low_coeff  +
         getTemplateValue(template, x_high, y_low)  * x_high_coeff * y_low_coeff  +
         getTemplateValue(template, x_low,  y_high) * x_low_coeff  * y_high_coeff +
         getTemplateValue(template, x_high, y_high) * x_high_coeff * y_high_coeff
end










