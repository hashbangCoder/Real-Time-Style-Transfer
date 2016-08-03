-- Residual block code snippet from fb.resnet.torch on github.
local models = {}
local nn = require 'nn'
local loadcaffe = require 'loadcaffe'

-- Load pretrained 16-layer VGG model and freeze layers
function models.load_vgg(backend,avg_pool)
	local model =  loadcaffe.load('VGG/VGG_ILSVRC_16_layers_deploy.prototxt','VGG/VGG_ILSVRC_16_layers.caffemodel',backend)
	for i=23,#model do
		model:remove()
	end
	--assert(model:get(#model).name == 'relu4_2','VGG Model is loaded incorrectly')
	for i=1,#model do
		model:get(i).accGradParameters = function() end
	end
	-- Change to average pooling option
	if avg_pool then
		local poolBackend = (backend == 'cudnn') and 'cudnn.SpatialMaxPooling' or 'nn.SpatialMaxPooling'
		local bknd = require(backend)
		model:replace(function(module)
			if torch.typename(module) == poolBackend then
				return bknd.SpatialAveragePooling(module.kW,module.kH,module.dW,module.dW)
			else return module
			end
		end)
	end

	return model
end


local function shortcut()
	return nn.Identity()
end

-- Residual block for the network.
local function resblock(n,nInputPlane,inputSize)
	local block = nn.Sequential()
	local s = nn.Sequential()
	s:add(nn.SpatialConvolution(nInputPlane,n,3,3,1,1,1,1))
	s:add(nn.SpatialBatchNormalization(n))
	s:add(nn.ReLU(true))
	s:add(nn.SpatialConvolution(n,n,3,3,1,1,1,1))
    block:add(nn.ConcatTable()
		:add(s)
    	:add(nn.Identity()))
	block:add(nn.CAddTable(true))
	--assert(inputSize == s:get(#s).output:size(),'Size of image must not change inside resblock')
	return block
end

-- Non-residual, flattened module. Alternative to resblock.
local function flatblock(n,nInputPlane,inputSize)
	local submod = nn.Sequential()
	submod:add(nn.SpatialConvolution(nInputPlane,n,3,3,1,1,1,1))	
	submod:add(nn.SpatialBatchNormalization(n))
	submod:add(nn.ReLU(true))
	submod:add(nn.SpatialConvolution(n,n,3,3,1,1,1,1))	
	submod:add(nn.SpatialBatchNormalization(n))
	return submod
end


-- Build the image transformation network with or without residual
function models.transform_net(res_flag)
	local model = nn.Sequential()
	model:add(nn.SpatialConvolution(3,32,9,9,1,1,4,4))
	model:add(nn.SpatialBatchNormalization(32))
	model:add(nn.ReLU(true))
	model:add(nn.SpatialConvolution(32,64,3,3,2,2,1,1))
	model:add(nn.SpatialBatchNormalization(64))
	model:add(nn.ReLU(true))
	model:add(nn.SpatialConvolution(64,128,3,3,2,2,1,1))
	model:add(nn.SpatialBatchNormalization(128))
	model:add(nn.ReLU(true))

	if res_flag then
		for i=1,5 do
			-- one resblock module adds all the residual submodules to the Sequential container
			model:add(resblock(128,128,model:get(#model).output:size()))
		end
	else
		for i =1,5 do
			model:add(flatblock(128,128,model:get(#model).output:size()))
		end
	end

	model:add(nn.SpatialFullConvolution(128,64,3,3,2,2,1,1,1,1))
	model:add(nn.SpatialBatchNormalization(64))
	model:add(nn.ReLU(true))
	model:add(nn.SpatialFullConvolution(64,32,3,3,2,2,1,1,1,1))
	model:add(nn.SpatialBatchNormalization(32))
	model:add(nn.ReLU(true))
	model:add(nn.SpatialFullConvolution(32,3,9,9,1,1,4,4))
	model:add(nn.SpatialBatchNormalization(3))
	model:add(nn.ReLU(true))
	return model
end

return models
