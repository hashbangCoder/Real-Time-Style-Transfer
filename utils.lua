local utils = {}
local image = require 'image'

function utils.pp(im)
	-- Take in RGB return BGR type
	-- If you're wondering what these floats are for, like I did, they're the mean pixel (Blue-Green-Red) for the pretrained VGG net in order to zero-center the input image.
	local means = torch.DoubleTensor({-103.939, -116.779, -123.68})
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:mul(255.0)
	means = means:view(3, 1, 1):expandAs(im)
	-- Subtract means and resize
	im:add(1, means)
	return im
end

function utils.toBGR(im)
	local perm = torch.LongTensor{3, 2, 1}
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function utils.scale_pp(im,size)
	-- Take in grayscale/RGB return RGB-type scaled image
	if im:size(1) ==1 then
		im = torch.cat(im,torch.cat(im,im,1),1)
	elseif im:size(1) == 3 then
	else
		error('Input image is not an RGB image')
	end
	im = image.scale(im,size,size,'bicubic')
	return im
	
end

function utils.dp(im)
	-- Exact inverse of above
	local perm = torch.LongTensor{3, 2, 1}
	im = im:index(1, perm)
	return im:double()
end
return utils
