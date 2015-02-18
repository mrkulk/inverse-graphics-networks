--[[ An implementation of RMSprop ]]

function rmsprop( x, dfdx, gradAverage, meta_learning_alpha)
	local flag =0
	if torch.type(x) == torch.type(1) then
		newx = torch.zeros(1)
		newx[1] = x; x = newx
		flag = 1

		newfddx = torch.zeros(1)
		newfddx[1] = dfdx; dfdx = newfddx
	end

    gradAverageArr=torch.zeros(2)
    gamma = {math.exp(3),math.exp(6)}
    for i=1,2 do
        gradAverageArr[i] = 1/gamma[i] * torch.pow(dfdx:norm(), 2) + (1-(1/gamma[i]))*gradAverage
    end
    gradAverage = torch.max(gradAverageArr)
    gradAverage = torch.clamp(torch.Tensor({gradAverage}), 1e-5, 1e20)[1]

    -- print('RATIO', meta_learning_alpha / gradAverage)
    -- print(dfdx:norm())
    -- print('\n')
    x = x - (dfdx*meta_learning_alpha / (gradAverage))

    -- print('QQ', x)
    if flag == 1 then
    	return x[1]
    else
    	return x
    end
end



-- function rmsprop( x, dfdx, config, state)
--    -- get parameters
--    local config = config or {}
--    local state = state or config
--    local lr = config.learningRate or 1e-3
--    local b1 = config.momentumDecay or 1
--    local b2 = config.updateDecay or 1

--    state.evalCounter = state.evalCounter or 0
--    state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
--    state.v = state.v or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)

--    -- Decay term
--    state.m:mul(1 - b1)

--    -- New term
--    state.momentum_dfdx = state.momentum_dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
--    state.momentum_dfdx:copy(dfdx)

--    state.m:add(state.momentum_dfdx:mul(b1))

--    -- Decay term of update
--    state.v:mul(1 - b2)

--    -- New update
--    dfdx:cmul(dfdx):mul(b2)

--    state.v:add(dfdx)

--    -- calculate update step
--    state.evalCounter = state.evalCounter + 1

--    --Create new momentum Tensors for cutorch compatibility
--    state.momentum_update = state.momentum_update or torch.Tensor():typeAs(state.m):resizeAs(state.m)
--    state.momentum_update:copy(state.m)

--    state.update = state.update or torch.Tensor():typeAs(state.v):resizeAs(state.v)
--    state.update:copy(state.v)

--    state.momentum_update:cdiv(state.update:add(1e-8):sqrt())

--    local gamma = (math.sqrt(1 - math.pow(1 - b2,state.evalCounter))/(1 - math.pow(1 - b1, state.evalCounter)))
--    state.momentum_update:mul(gamma)

--    x:add(-lr, state.momentum_update)

--    -- return x*, f(x) before optimization
--    return x
-- end
