-- numerical gradient tests

function checkEncoderGrads(criterion, model, data)
  criterion:forward(model:forward(data), data)
  model:backward(data, criterion:backward(model.output, data))

  local EPSILON = 1e-7--0.0001
  grad_diff = 0

  local enc = model.modules[1].modules[4]
  local enc_truegrad_weight = torch.zeros(enc.weight:size())
  local enc_truegrad_bias = torch.zeros(enc.bias:size())

  print(enc_truegrad_weight:size())
  print(enc.weight:size())

  for id=1, enc_truegrad_weight:size()[1] do
    for ii = 1, enc_truegrad_weight:size()[2] do
      local t = enc.weight[{id, ii }]
      enc.weight[{id,ii}] = t + EPSILON
      J_pos = criterion:forward(model:forward(data), data)

      enc.weight[{id,ii}] = t - EPSILON
      J_neg = criterion:forward(model:forward(data), data)

      enc_truegrad_weight[{id,ii}] = (J_pos - J_neg)/(2*EPSILON)
      print('.')
      enc.weight[{id,ii}] = t

      print('Calculated:', enc.gradWeight[{id,ii}], ' FiniteD:', enc_truegrad_weight[{id,ii}])
    end
  end

  print('[GRADIENT CHECKER: TEMPLATE] Error: ', grad_diff)  
end



function checkTemplateGrads(criterion, model, data, num_acrs)
  criterion:forward(model:forward(data), data)
  model:backward(data, criterion:backward(model.output, data))

  local EPSILON = 0.0001
  grad_diff = 0
  for ac=1,num_acrs do
    print('ACR #', ac)
    local ac_bias = model.modules[3].modules[ac].modules[3].modules[1].modules[1]
    local ac_bias_truegrad = torch.zeros(ac_bias.bias:size())
    for id=1, ac_bias.bias:size()[1] do
      print('batchid #', id)
      for ii = 1, ac_bias.bias:size()[2] do
        local t = ac_bias.bias[{id, ii }]
        ac_bias.bias[{id,ii}] = t + EPSILON
        J_pos = criterion:forward(model:forward(data), data)

        ac_bias.bias[{id,ii}] = t - EPSILON
        J_neg = criterion:forward(model:forward(data), data)

        ac_bias_truegrad[{id,ii}] = (J_pos - J_neg)/(2*EPSILON)
        print('.')
        ac_bias.bias[{id,ii}] = t
      end
      local diff = torch.sum(torch.pow(ac_bias_truegrad - ac_bias.gradBias,2))
      print('ACR error:', diff)
      print(ac_bias_truegrad)
      print(ac_bias.gradBias)
      grad_diff = grad_diff + diff
    end
  end

  print('[GRADIENT CHECKER: TEMPLATE] Error: ', grad_diff)


end

