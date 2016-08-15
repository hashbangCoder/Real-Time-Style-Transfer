-- file's code mix of and borrowed from jcjhonson/neural_style.lua and kaishengtai/neuralart
local lossUtils = {} 
require 'loadcaffe'
local os = require 'os'

local function gramMatrix(device)
	local net = nn.Sequential()
	net:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
    concat:add(nn.Identity())
	concat:add(nn.Identity())
	net:add(concat)
	net:add(nn.MM(false, true))
	net = (device>=0) and cudnn.convert(net,cudnn):cuda() or net
	return net
end


function lossUtils.tvLoss(input,tvFactor)
	local x_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {1, -2}, {2, -1}}]
	local y_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {2, -1}, {1, -2}}]
	local grad = input.new():resize(input:size()):zero()
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, x_diff)
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, y_diff)
	return grad:mul(tvFactor)
end


function lossUtils.styleLoss(input,styleMap,sFactor,device)

	local gramNet = gramMatrix(device)
	local styleGram = gramNet:forward(styleMap):clone()
	local crit = (device>=0) and nn.MSECriterion():cuda() or nn.MSECriterion()
	local imageGram = gramNet:forward(input)
	local loss = crit:forward(imageGram:view(-1),styleGram:view(-1))
	local grad = crit:backward(imageGram:view(-1),styleGram:view(-1)):view(#imageGram)
	grad = torch.mm(grad, input:view(input:size(2), -1)):view(input:size())
	return loss*sFactor, grad:mul(sFactor)
end


function lossUtils.contentLoss(input,contentMap,cFactor,device)
	local crit = (device>=0) and nn.MSECriterion():cuda() or nn.MSECriterion()
	local loss =  crit:forward(input:view(-1),contentMap:view(-1))
--	print(torch.type(input),torch.type(contentMap),loss)
	local grad = crit:backward(input:view(-1),contentMap:view(-1)):view(#input)
	return loss*cFactor, grad:mul(cFactor)

end


return lossUtils



