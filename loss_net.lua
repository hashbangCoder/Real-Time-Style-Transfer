-- A lot of file's code is borrowed from jcjhonson/neural_style.lua and kaishengtai/neuralart

require 'loadcaffe'


function gramMatrix()
	local net = nn.Sequential()
	net:add(nn.View(-1):setNumInputDims(2))
	local concat = nn.ConcatTable()
    concat:add(nn.Identity())
	concat:add(nn.Identity())
	net:add(concat)
	net:add(nn.MM(false, true))
	return net
end


function tvLoss(input)
	local x_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {1, -2}, {2, -1}}]
	local y_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {2, -1}, {1, -2}}]
	local grad = gen.new():resize(gen:size()):zero()
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, x_diff)
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, y_diff)
	return grad
end


function styleLoss(input,styleMap,sFactor,normFlag)
	local styleGram = gramMatrix(stylemap)
	local crit = nn.MSECriterion()
	local imageGram = gramMatrix(input)
	local loss = crit:forward(imageGram:view(-1),styleGram:view(-1))
	local grad = crit:backward(imageGram:view(-1),styleGram:view(-1)):view(#imageGram)
	if normFlag then
		local norm = input:nElement()^2
		grad:div(norm)
		loss = loss/norm
	end
	grad = torch.mm(grad, input:view(k, -1)):view(input:size())
	return loss*sFactor, grad
end


function contentLoss(input,contentMap,cFactor,normFlag)
	local crit = nn.MSECriterion()
	local loss =  crit:forward(input,contentMap)
	local grad = crit:backward(input,contentMap):view(#input)
	if normFlag then
		local norm = input:nElement()^2
		loss = loss/norm
		grad = grad:div(norm)
	end
	return loss*cFactor, grad

end

function load_vgg(backend)
	local model =  loadcaffe.load('VGG/VGG_ILSVRC_16_layers_deploy.prototxt','VGG/VGG_ILSVRC_16_layers.caffemodel',backend)
	for i=1,#model do
		-- TODO : Add functionality for avg. pooling
		model:get(i).accGradParameters = function() end
	end
	return model
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
--



