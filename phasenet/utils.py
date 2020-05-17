from csbdeep.utils import _raise
import random

def cropper3D(image, crop_shape, jitter=False, max_jitter=None, planes=None):

	"""
        crops 3d images
        
        :param image: 3d array, image to be cropped
        :param crop_shape: tuple, crop shape
        :param jitter: booelan, randomly move the center point within a given limit, default is False
        :param max_jitter: tuple, maximum displacement for jittering, if None then it gets a default value
	"""

	if planes is None:
		half_crop_shape = tuple(c//2 for c in crop_shape)
	else:
		half_crop_shape = tuple((crop_shape[-1]//2,crop_shape[-1]//2,crop_shape[-1]//2))

	half_image_shape = tuple(i//2 for i in image.shape)
	assert all([c<=l for c,l in zip(half_crop_shape,half_image_shape)]), "Crop shape is bigger than image shape"

	if jitter:
		contraint_1 = tuple((l-c)//4 for c,l in zip(half_crop_shape,half_image_shape))
		contraint_2 = tuple(c//2 for c in half_crop_shape)

		if max_jitter is None:
			max_jitter = tuple([min(_x,_y) for _x,_y in zip(contraint_1,contraint_2)])
		assert all([l-m>=0 and l+m<2*l for m,l in zip(max_jitter,half_image_shape)]), "Jittering results in cropping outside border, please reduce max_jitter"
		loc = tuple(l-random.randint(-1*max_jitter[i],max_jitter[i]) for i,l in enumerate(half_image_shape))
	else:
		loc=half_image_shape

	final_img = image[loc[0]-half_crop_shape[0]:loc[0]+half_crop_shape[0], loc[1]-half_crop_shape[1]:loc[1]+half_crop_shape[1], loc[2]-half_crop_shape[2]:loc[2]+half_crop_shape[2]]

	if planes is not None:
		try:
			final_img = final_img[planes+half_crop_shape[0]]
		except:
			_raise(ValueError("Plane does not exist"))

	return final_img 

