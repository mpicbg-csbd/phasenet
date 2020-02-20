from csbdeep.utils import _raise
import random

def cropper3D(image, params):

	"""
        crops 3d images
        
        :param  image: 3d array, image to be cropped
        :params dictionary: should contain crop_shape(tuple), jitter(boolean) and max_jitter(scalar)
	"""

	'crop_shape' in params or _raise(ValueError('Crop shape not defined'))
	crop_shape = params.get('crop_shape')
	half_crop_shape = tuple(c//2 for c in crop_shape)
	half_image_shape = tuple(i//2 for i in image.shape)
	assert all([c<l for c,l in zip(half_crop_shape,half_image_shape)]), "Crop shape is bigger than equal to image shape"

	jitter = params.get('jitter', False)
	if jitter:
		contraint_1 = tuple((l-c)//4 for c,l in zip(half_crop_shape,half_image_shape))
		contraint_2 = tuple(c//2 for c in half_crop_shape)

		max_jitter = params.get('max_jitter', tuple([min(_x,_y) for _x,_y in zip(contraint_1,contraint_2)]))
		assert all([l-m>=0 and l+m<2*l for m,l in zip(max_jitter,half_image_shape)]), "Jittering results in cropping outside border, please reduce max_jitter"
		loc = tuple(l-random.randint(-1*max_jitter[i],max_jitter[i]) for i,l in enumerate(half_image_shape))
	else:
		loc=half_image_shape
	return image[loc[0]-half_crop_shape[0]:loc[0]+half_crop_shape[0], loc[1]-half_crop_shape[1]:loc[1]+half_crop_shape[1], loc[2]-half_crop_shape[2]:loc[2]+half_crop_shape[2]]

