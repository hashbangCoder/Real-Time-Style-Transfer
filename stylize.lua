require('image')
require('cudnn')
require('cutorch')
require('cunn')
local utils = require('utils.lua')
cmd = torch.CmdLine()
cmd:text()
cmd:text('Real Time Neural Style Transfer in Torch')
cmd:text('Options : ')

cmd:option('-test','test_image.jpg','Test Image path')
cmd:option('-im_format','jpg','Test Image format - JPG|PNG')
cmd:option('-model','Output_1/Styles/transformNet.t7','Saved model file')
cmd:option('-gpu',1,'GPU ID')
cmd:option('-output','Stylizations/','Saved final image to')
cmd:option('-size',512,'Size of output image')
-----------------------------------------------------------------------------------

cmd:text()
local opt = cmd:parse(arg)	
cutorch.setDevice(opt.gpu+1)


local saveIm = (opt.im_format == 'jpg') and image.saveJPG  or image.savePNG
local loadIm = (opt.im_format == 'jpg') and image.loadJPG or image.loadPNG

local testIm = utils.scale_pp(loadIm(opt.test),opt.size)
testIm:resize(1,3,testIm:size(2),testIm:size(3))
testIm = testIm:cuda()

local net = torch.load(opt.model)
net = net:cuda()

local out = net:forward(testIm)
local file_name = paths.concat(opt.output,'Stylize_test'..'.'..opt.im_format)
saveIm(file_name,out:squeeze():double())
print ('Saved Image at : ' .. file_name)

