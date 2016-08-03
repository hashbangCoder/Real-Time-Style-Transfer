local utils = {}
local image = require 'image'

function utils.pp(im)
	local im_size = im:size()
	if im:size(1) ~= 3 then
	print(#im)
	assert(im:size(1) == 3,'Only RGB images are valid')
	end
	return im:resize(1,3,256,256)

end

function utils.saveImage(iter)
	local inputIm = utils.pp(loadIm(paths.concat(opt.test),imageFiles[shuffleInd[j]])):cuda()
	transformNet:evaluate()
	local out = transformNet:forward(inputIm)
	saveIm(paths.concat(opt.output,'modelOutIter'..tostring(iter)),output:squeeze())
	transformNet:training()
end

return utils
