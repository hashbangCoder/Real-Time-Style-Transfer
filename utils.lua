local image = require 'image'

function pp(im)
	local im_size = im:size()
	assert(im:size(1) == 3,'Only RGB images are valid')
	return im:resize(1,3,256,256)

end

function saveImage(iter)
	local inputIm = utils.pp(loadIm(paths.concat(opt.test),imageFiles[shuffleInd[j]])):cuda()
	transformNet:evaluate()
	local out = transformNet:forward(inputIm)
	saveIm(paths.concat(opt.output,'test'..tostring(iter)),output:squeeze())
end
