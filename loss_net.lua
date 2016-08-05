-- file's code mix of and borrowed from jcjhonson/neural_style.lua and kaishengtai/neuralart
local lossUtils = {} 
require 'loadcaffe'
local os = require 'os'

local function gramMatrix(gpu)
	local net = nn.Sequential()
	net:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
    concat:add(nn.Identity())
	concat:add(nn.Identity())
	net:add(concat)
	net:add(nn.MM(false, true))
	net = (gpu>=0) and cudnn.convert(net,cudnn):cuda() or net
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


function lossUtils.styleLoss(input,styleMap,sFactor,normFlag,gpu)
	local gramNet = nn.Sequential()
	gramNet:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
    concat:add(nn.Identity())
	concat:add(nn.Identity())
	gramNet:add(concat)
	gramNet:add(nn.MM(false, true))
	gramNet = (gpu>=0) and cudnn.convert(gramNet,cudnn):cuda() or gramNet

	--local gramNet = gramMatrix(gpu)
	local styleGram = gramNet:forward(styleMap):clone()
	local crit = (gpu>=0) and nn.MSECriterion():cuda() or nn.MSECriterion()
	local imageGram = gramNet:forward(input)
	local loss = crit:forward(imageGram:view(-1),styleGram:view(-1))
	local grad = crit:backward(imageGram:view(-1),styleGram:view(-1)):view(#imageGram)
	if normFlag then
		local norm = torch.norm(grad,1) + 1e-7
		grad:div(norm)
		loss = loss/norm
	end
	grad = torch.mm(grad, input:view(input:size(2), -1)):view(input:size())
	return loss*sFactor, grad:mul(sFactor)
end


function lossUtils.contentLoss(input,contentMap,cFactor,normFlag,gpu)
	local crit = (gpu>=0) and nn.MSECriterion():cuda() or nn.MSECriterion()
	local loss =  crit:forward(input,contentMap)
--	print(torch.type(input),torch.type(contentMap),loss)
	local grad = crit:backward(input,contentMap):view(#input)
	if normFlag then
		local norm = torch.norm(grad,1) + 1e-7
		loss = loss/norm
		grad = grad:div(norm)
	end
	return loss*cFactor, grad:mul(cFactor)

end


--
--function feature_loss(input,target,model)
--	model:forward(input)
--	local output = model:get(9).output and (model:get(9).name == 'relu2_2') or nil
--	assert(output,'Check image output for relu2_2')
--	
--	model:forward(target)
--	local target_out = model:get(9).output and (model:get(9).name == 'relu2_2') or nil
--	assert(target_out,'Check target output for relu2_2')
--
--	feat_loss =  (output:prod()^-1)*((output - target_out):pow(2):sum())
--	if feat_loss then return feat_loss else error('Feature Loss is zero') end
--end
--
--function style_loss(input,target,model)
--    local style_relus = {'relu1_2','relu2_2','relu3_3','relu4_3'}
--	local style_relus = {2,4,7,10}
--	local all_relus = model:findModules('nn.ReLU')
--	local input_relu_maps = {}
--	local target_relu_maps = {}
--	model:forward(input)
--	for i,v in ipairs(style_relus) do
--		for j,r in ipairs(all_relus) do
--			if not (r.name == v) then 
--			else
--				--convert to gram matrix
--				local out = r.output
--				local out_resh = out:reshape(out:size(1),out:size(2)*out:size(3)) 
--				table.insert(input_relu_outputs,torch.mm(out_resh,out_resh:t())/out_resh:size():prod())
--		    end
--
--	model:forward(target)
--	for i,v in ipairs(style_relus) do
--		for j,r in ipairs(all_relus) do
--			if not (r.name == v) then 
--			else
--				--convert to gram matrix
--				local out = r.output
--				local out_resh = out:reshape(out:size(1),out:size(2)*out:size(3)) 
--				table.insert(target_relu_outputs,torch.mm(out_resh,out_resh:t())/out_resh:size():prod())
--			end
--	for i,v in ipairs(style_relus) do
--		local out = all_relus[v].output
--		local out_resh = out:reshape(out:size(1),out:size(2)*out:size(3)) 
--		table.insert(input_relu_maps,torch.mm(out_resh,out_resh:t())/out_resh:size():prod())
--	end
--	model:fprward(target)
--	for i,v in ipairs(style_relus) do
--		local out = all_relus[v].output
--		local out_resh = out:reshape(out:size(1),out:size(2)*out:size(3)) 
--		table.insert(target_relu_maps,torch.mm(out_resh,out_resh:t())/out_resh:size():prod())
--	end	
--	style_loss = 0
--	for i =1,#input_relu_maps do
--		style_loss = style_loss +  (input_relu_maps[i] - target_relu_maps[i]):pow(2):sum()
--	end
--	return style_loss
--end


--
--function variation_loss(modelOp,beta)
--	local filtV = torch.Tensor({1,-1}):reshape(2,1)
--	local filtH = torch.Tensor({1,-1}):reshape(1,2)
--	opV = torch.conv2(modelOp,filtV,'V'):pow(2):sum()
--	opH = torch.conv2(modelOp,filtH,'V'):pow(2):sum()
--	return (opV + opH)^(beta/2)
return lossUtils



