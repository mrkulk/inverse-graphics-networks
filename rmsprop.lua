--[[ An implementation of RMSprop ]]

function rmsprop( x, dfdx, gradAverage)
	meta_learning_alpha = 0.05

	gradAverageArr=torch.zeros(2)
	gamma = {math.exp(3),math.exp(6)} 
	for i=1,2 do
    	gradAverageArr[i] = 1/gamma[i] * torch.pow(dfdx:norm(), 2) + (1-(1/gamma[i]))*gradAverage
    end
    gradAverage = torch.max(gradAverageArr)
    x = x - (dfdx*meta_learning_alpha / gradAverage)
    return x

end