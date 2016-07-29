local image = require 'image'

function pp(im)
	local im_size = im:size()
	assert(im:size(1) == 3,'Only RGB images are valid')
	return im:resize(1,3,256,256)

end
