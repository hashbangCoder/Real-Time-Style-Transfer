require 'torch'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'optim'
require 'xlua'
local models = require 'models.lua'
local lossUtils = require 'loss_net.lua'
local utils = require 'utils.lua'


------------------- CommandLine options ---------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Real Time Neural Style Transfer in Torch')
cmd:text()
cmd:text('Options : ')

cmd:option('-test','test_image.jpg','Test Image path')

cmd:option('-style','','Style Image')
cmd:option('-dir','Data/train2014/','Path to Content Images')
cmd:option('-style_maps','4,9,16,22','Layer outputs for style')
cmd:option('-content','9','Layer outputs for content')
cmd:option('-style_param',10,'Hyperparameter for style emphasis')
cmd:option('-feature_param',1,'Hyperparameter for feature emphasis')
cmd:option('-tv_param',5e-4,'Hyperparameter for total variation regularization')

cmd:option('-iter',12,'Number of iteration to train')
cmd:option('-save_freq',5000,'How frequently to save output ')
cmd:option('-output','Output/','Save output to')
cmd:option('-batch_size',4,'#Images per batch')
cmd:option('-lr',1e-3,'Learning rate for optimizer')
cmd:option('-beta',0.5,'Beta value for Adam optim')

cmd:option('-gpu',0,'GPU ID')
cmd:option('-res',true,'Flag to use residual model or not. Residual architecture converges faster!')
cmd:option('-pooling','avg','Pooling type for CNN  - "avg" or "max"')
cmd:option('-im_format','jpg','Image format - jpg|png')

cmd:option('-debug',false,'Turn debugger on/off')
cmd:option('-log','Logs/','File to log results')
-----------------------------------------------------------------------------------


cmd:text()
local opt = cmd:parse(arga)
cmd:log(opt.log .. 'main.log',opt)
--if opt.debug then
--	local system = require 'sys'
--	system.execute('export $') 
--end

if opt.gpu > 0 then
	require 'cutorch'
	require 'cudnn'
	require 'cunn'
	cutorch.setDevice(opt.gpu+1)
	cudnn.fastest = true
	backend = 'cudnn'
else 
	require 'nn'
	backend = 'nn'
end

-- Load models and convert to cuda()
local pooltype = true and (opt.pooling == 'avg') or false
lossNet = models.load_vgg(backend,pooltype)
transformNet  = model.transform_net(opt.res)
if opt.gpu > 0 then
	transformNet = cudnn.convert(transformNet,cudnn)
	lossNet = lossNet:cuda()
	transformNet = transformNet:cuda()
end

--Get image file names
local imageFiles = {}
for file in paths.files(opt.dir) do
	if file:find('jpg' .. '$') then
		table.insert(paths.concat(opt.dir),file)
	end
end


local loadIm = image.loadJPG and (opt.im_format == 'jpg') or image.loadPNG
local saveIm = image.saveJPG and (opt.im_format == 'jpg') or image.savePNG
local styleIm = utils.pp(loadIm(opt.style)):cuda()
local content_layers = opt.content_maps:split(',')
assert(#content_layers == 1,'More than one content activation layers received')
local style_layers = opt.style_maps:split(',')
local shuffleInd = torch.randperm(#imageFiles)

local logger = optim.Logger(opt.log .. 'train.log')
trainLogger:setNames{'Train Loss'}
local optimState = {
		learningRate = opt.lr,
		beta1 = opt.beta,
	}

local optimMethod = optim.adam

--Get style Loss 
local function getStyleMap() 
	local sm = {}
	for _,v in ipairs(style_layers) do
		table.insert(sm,lossNet:get(tonumber(v)).output) 
	return sm
end

lossNet:forward(styleIm)
local styleMap = getStyleMap()

----------------------------------- start training ------------------------------------------
for i=1,opt.iter,opt.batch_size do
	local k = i%#imageFiles 
	local mini_batch = {}
	for j = k,k+opt.batch_size do
		-- Get content image from mini-batch
		local inputIm = utils.pp(loadIm(paths.concat(opt.dir),imageFiles[shuffleInd[j]])):cuda()
		table.insert(mini_batch,inputIm)
	end
	xlua.progress(i,#imageFiles)
	if i%opt.save_freq == 0 then
		utils.saveImage(i)
	end
end
----------------------------------------------------------------------------------------------

-- Function for calculating loss and gradients
local function feval(mini_batch)
	collectgarbage()
	local loss = 0
	local grad = (opt.gpu > 0) and torch.CudaTensor():resize(#mini_batch[1]):zero() or torch.Tensor():resize(#mini_batch[1]):zero()
	for m=1,#mini_batch do 
		local inputIm = mini_batch[m]
		--Forward content image through VGG and get contentMap
		lossNet:forward(inputIm)
		local contentMap = lossNet:get(tonumber(content_layers[1])).output

		--Forward same content through tarnsformnet and get output image style and contentmaps
		local outputIm = transformNet:forward(inputIm)	
		lossNet:forward(outputIm)
		local outputStyleMap = getStyleMap()
		local outputContentMap = lossNet:get(tonumber(content_layers[1])).output

		local tvGrad =  lossUtils.tvLoss(outputIm)
		grad:add(tvGrad)
		local cLoss,Cgrad = lossUtils.contentLoss(outputContentMap,contentMap,opt.feature_param,false)
		grad:add(cGrad)
		loss = loss + cLoss
		for	m = 1,#styleMap do
			local sLoss,sGrad = lossUtils.styleLoss()
			loss = loss + sLoss
			grad:add(sGrad)
		end
		transform_net:backward(loss,grad)
	end
	loss = loss/#mini_batch
	grad = grad:div(#mini_batch)
	logger:add{loss}
	return loss,grad
end



