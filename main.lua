
require 'torch'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'optim'
require 'xlua'
models = require 'models.lua'
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
--cmd:option('-style_maps','4,9,16,25','Layer outputs for style')
cmd:option('-style_maps','4,9,16,23','Layer outputs for style')
cmd:option('-content_maps','16','Layer outputs for content')
cmd:option('-style_param',1e7,'Hyperparameter for style emphasis')
cmd:option('-feature_param',1,'Hyperparameter for feature emphasis')
cmd:option('-tv_param',5e-3,'Hyperparameter for total variation regularization')
cmd:option('-size',256,'Size of output')

cmd:option('-iter',160000,'Number of iteration to train')
cmd:option('-batch_size',4,'#Images per batch')
cmd:option('-save_freq',20000,'How frequently to save output ')
cmd:option('-saved_params','transformNet.t7','Save output to')
cmd:option('-output','Output/','Save output to')

cmd:option('-lr',1e-3,'Learning rate for optimizer')
cmd:option('-beta',0.5,'Beta value for Adam optim')

cmd:option('-gpu',1,'GPU ID, -1 for CPU (not fully supported now)')
cmd:option('-res',true,'Flag to use residual model or not. Residual architecture converges faster!')
cmd:option('-pooling','max','Pooling type for CNN  - "avg" or "max"')
cmd:option('-im_format','jpg','Image format - jpg|png')

cmd:option('-log','Logs/','File to log results')
-----------------------------------------------------------------------------------

cmd:text()
local opt = cmd:parse(arg)
cmd:log(opt.log .. 'main.log',opt)
--if opt.debug then
--	local system = require 'sys'
--	system.execute('export $') 
--end
if opt.gpu >= 0 then
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
local pooltype = (opt.pooling == 'avg') and true or false
local loadIm = (opt.im_format == 'jpg') and image.loadJPG or image.loadPNG
local saveIm = (opt.im_format == 'jpg') and image.saveJPG  or image.savePNG
if opt.style == '' or opt.content == '' then
	error('No style/content image given')
end
local styleIm = utils.pp(utils.scale_pp(loadIm(opt.style),opt.size))
styleIm:resize(1,3,styleIm:size(2),styleIm:size(3))
local testIm = utils.scale_pp(loadIm(opt.test),opt.size)
testIm:resize(1,3,testIm:size(2),testIm:size(3))


lossNet = models.load_vgg(backend,pooltype)
transformNet  = models.transform_net(opt.res)
if opt.gpu >= 0 then
	lossNet = lossNet:cuda()
	transformNet = transformNet:cuda()
	styleIm = styleIm:cuda()
	testIm = testIm:cuda()
end

--Get image file names
local imageFiles = {}
for file in paths.files(opt.dir) do
	if file:find('jpg' .. '$') then
		table.insert(imageFiles,paths.concat(opt.dir,file))
	end
end
--Content Layers
local content_layers = opt.content_maps:split(',')
local content_outs = {}
assert(#content_layers == 1,'More than one content activation layers received')
content_outs[tonumber(content_layers[1])] = true

-- Style layers
local style_layers = opt.style_maps:split(',')
local style_outs = {} 
for i,v in ipairs(style_layers) do
	style_outs[tonumber(v)] = true
end
local shuffleInd = torch.randperm(#imageFiles)

local logger = optim.Logger(opt.log .. 'trainloss.log')
logger:setNames{'Train Loss'}
local optimState = {
					learningRate = opt.lr,
					beta1 = opt.beta,
				   }
local optimMethod = optim.adam

local function saveImage(iter)
	local out = transformNet:forward(testIm)
	saveIm(paths.concat(opt.output,'testOutIter'..tostring(iter)..'.'..opt.im_format),out:squeeze():double())
	print ('Saving Test Image @ iter : '..tostring(iter))
end

--Get style Loss 
lossNet:forward(styleIm)
local styleMap = {}
for i,_ in pairs(style_outs) do
		styleMap[i] = lossNet:get(tonumber(i)).output:clone()
end
local params,gradParameters = transformNet:getParameters()

----------------------------------- start training ------------------------------------------
for i=1,opt.iter,opt.batch_size do
	local k = (i%(#imageFiles-opt.batch_size+1)==0) and 1 or i%(#imageFiles-opt.batch_size+1)
	local mini_batch = {}
	xlua.progress(i,opt.iter)
	for j = k,k+opt.batch_size-1 do
		-- Get content image from mini-batch
		local inputIm = utils.scale_pp(loadIm(paths.concat(opt.dir,imageFiles[shuffleInd[j]])),opt.size)
		table.insert(mini_batch,inputIm)
	end

	-- Function for calculating loss and gradients
	-- Run gradients through lossNet and then do final backpass through transformNet
	local function feval(params)
		collectgarbage()
		gradParameters:zero()
		local loss = 0
		for m=1,#mini_batch do 
			local gradOut = (opt.gpu >= 0) and torch.CudaTensor():resize(#lossNet.output):zero() or torch.Tensor():resize(#lossNet.output):zero()
			local contentIm = utils.pp(mini_batch[m]):clone():cuda()
			contentIm:resize(1,3,contentIm:size(2),contentIm:size(3))
			local inputIm = mini_batch[m]:cuda()
			inputIm:resize(1,3,inputIm:size(2),inputIm:size(3))

			local outputIm = transformNet:forward(inputIm):clone()
			if i%opt.save_freq < opt.batch_size and (i>opt.batch_size) then
				saveIm(paths.concat(opt.output,'transformOutIter'..tostring(i)..'.'..opt.im_format),utils.dp(outputIm:squeeze():double()))
			end
			--Forward content image through VGG and get contentMap
			lossNet:forward(contentIm)
			local contentMap = lossNet:get(tonumber(unpack(content_layers))).output:clone()
			
			local outputIm = utils.pp(outputIm:squeeze():double())
			outputIm = outputIm:resize(1,3,outputIm:size(2),outputIm:size(3)):cuda()
			lossNet:forward(outputIm)
															
			-- Backpass inspired by kaishengtai/neuralart
			for z = #lossNet.modules, 1, -1 do
				local mod_inp = (z== 1) and outputIm or lossNet.modules[z - 1].output
				local module = lossNet.modules[z]
				if content_outs[z] == true then
					local cLoss,cGrad = lossUtils.contentLoss(lossNet:get(z).output,contentMap,opt.feature_param,opt.gpu)
					loss = loss + cLoss
					--print(m,z,cLoss)
					gradOut:add(cGrad)
				end
				if style_outs[z] == true  then
					local sLoss,sGrad = lossUtils.styleLoss(lossNet:get(z).output,styleMap[z],opt.style_param,opt.gpu)
					loss = loss+sLoss
					--print(m,z,sLoss)
					gradOut:add(sGrad)
				end
				gradOut = module:backward(mod_inp,gradOut)
			end
			local tvGrad = lossUtils.tvLoss(outputIm,opt.tv_param)
			gradOut:add(tvGrad)
			transformNet:backward(inputIm,gradOut)
		end
		loss = loss/#mini_batch
		gradParameters:div(#mini_batch)
		return loss,gradParameters:view(-1)
	end

	
	local _,loss = optimMethod(feval,params,optimState)
	logger:add{loss[1]}
	if i%5000 < opt.batch_size and (i>opt.batch_size)then
		print('Iter :'..tostring(i),'Loss :'..tostring(loss[1]), 'LR :',optimState.learningRate)
	end
	if i%opt.save_freq < opt.batch_size and (i>opt.batch_size)then
		saveImage(i)
	end
	if i%100000 < opt.batch_size and (i>opt.batch_size) then
		optimState.learningRate  = optimState.learningRate/2
	end
end
----------------------------------------------------------------------------------------------
saveImage('end')
torch.save(opt.output..'Styles/'..opt.saved_params,transformNet:clearState())

