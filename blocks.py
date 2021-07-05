import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, LeakyReLU, BatchNormalization



def conv_block(input_layer, filters_shape, down_sample = False,
		activate = True, batch_norm = True, regularization = 0.0005, reg_stddev = 0.01, activate_alpha = 0.1):

	if down_sample:
		input_layer = ZeroPadding2D(((1,0),(1,0)))(input_layer)
		padding ="valid"
		strides = 2
	else:
		padding ="same"
		strides = 1

	x = Conv2D(filters=filters_shape[-1],
		kernel_size = filters_shape[0],
		strides = strides,
		padding = padding,
		use_bias = not batch_norm,
		kernel_regularizer= keras.regularizers.l2(regularization),
		kernel_initializer = keras.initializers.RandomNormal(stddev=reg_stddev),
		bias_initializer = keras.initializers.Zeros()
    )(input_layer)

	if batch_norm:
		x = BatchNormalization()(x)
	if activate:
		x = LeakyReLU(alpha=activate_alpha)(x)

	return x


def res_block(input_layer, input_channel, filter_num1, filter_num2):
	shortcut = input_layer
	x = conv_block(input_layer, filters_shape=(1,1,input_channel,filter_num1))
	x = conv_block(x, filters_shape=(3,3,filter_num1,filter_num2))

	res_output = shortcut + x
	return res_output



def darknet53(input_layer):
	x = conv_block(input_layer,(3,3,3,32))
	x = conv_block(x, (3,3,32,64), down_sample = True)

	for i in range(1):
		x = res_block(x, 64,32,64)

	x = conv_block(x, (3,3,64,128),down_sample=True)

	for i in range(2):
		x = res_block(x, 128,64,128)

	x = conv_block(x, (3,3,128,256), down_sample= True)

	for i in range(8):
		x = res_block(x,256,128,256)


	route_1 = x 

	x = conv_block(x,(3,3,256,512), down_sample= True)

	for i in range(8):
		x = res_block(x,512,256,512)

	route_2 = x
  
	x = conv_block(x,(3,3,512,1024), down_sample= True)

	for i in range(4):
		x= res_block(x,1024,512,1024)


	return route_1, route_2, x



def upsample(input_layer):
	return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),
		method='nearest')