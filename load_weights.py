import numpy as np 


FILE_PATH = "./model_weights/yolov3.weights"

def load_weights(model):

  with open(FILE_PATH, 'rb') as f:

    j=0
    for i in range(75):
      conv_layer_name = 'conv2d_%d' %i if i>0 else 'conv2d'
      bn_layer_name = 'batch_normalization_%d' %j if j>0 else 'batch_normalization'   
      conv_layer = model.get_layer(conv_layer_name)
      filters = conv_layer.filters
      kernel_size = conv_layer.kernel_size[0]
      in_dim = conv_layer.input_shape[-1] 
      if i not in [58,66,74]:
        # darknet weights: [beta, gamma, mean, variance]
        bn_weights = np.fromfile(f, dtype= np.float32, count = 4*filters)
        bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
        bn_layer = model.get_layer(bn_layer_name) 
        j+=1  
      else:
        conv_bias = np.fromfile(f,dtype= np.float32, count= filters) 
      # darknet shape is (out_dim, in_dim, height,width)
      conv_shape = (filters, in_dim,kernel_size,kernel_size)
      conv_weights = np.fromfile(f,dtype= np.float32, count= np.product(conv_shape)) 
      #tf shpae (height, width, in_dim, out_dim)
      conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])  
      if i not in [58,66,74]:
        conv_layer.set_weights([conv_weights])
        bn_layer.set_weights(bn_weights)
      else:
        conv_layer.set_weights([conv_weights,conv_bias])

    assert len(f.read(0))==0, 'failed to read all data'

  return model