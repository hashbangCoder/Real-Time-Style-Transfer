local utils = {}
local image = require 'image'

function utils.pp(im)
	local rgbIm = im
	if im:size(1) == 1 then
		print('grayscale image')
		rgbIm = torch.cat(im,torch.cat(im,im,1),1)
	--print('Image size : ',#im)
	--assert(im:size(1) == 3,'Only RGB images are valid')
	elseif im:size(1) == 3 then
	else 
		print(im:size(1))
		error('Input image has invalid dimensions of (2) or >3')
	end
	return rgbIm:resize(1,3,256,256)

end

return utils
