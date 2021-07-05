"""
YOLO-v3 Model
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from blocks import *
from hyperparams import *
from load_weights import *


def yolo_v3(input_layer):

  route_1, route_2, x = darknet53(input_layer)

  x = conv_block(x, (1,1,1024,512))
  x = conv_block(x,(3,3,512,1024))
  x = conv_block(x, (1,1,1024, 512))
  x = conv_block(x, (3,3,512,1024))
  x = conv_block(x,(1,1,1024,512))

  conv_lobj_branch = conv_block(x,(3,3,512,1024))
  conv_lbbox = conv_block(conv_lobj_branch,(1,1,1024,3*(NUM_CLASSES+5)), activate= False, batch_norm = False)

  x = conv_block(x,(1,1,512,256))
  x = upsample(x)

  x = tf.concat([x, route_2], axis=-1)

  x = conv_block(x,(1,1,768,256)) 
  x = conv_block(x,(3,3,256, 512))
  x = conv_block(x,(1,1,512,256))
  x = conv_block(x,(3,3,256,512))
  x = conv_block(x, (1,1,512,256))

  conv_mobj_branch = conv_block(x, (3,3,256,512))
  conv_mbbox = conv_block(conv_mobj_branch ,(1,1,512,3*(NUM_CLASSES+5)), activate= False, batch_norm= False)

  x = conv_block(x, (1,1,256,128))
  x = upsample(x)

  x = tf.concat([x,route_1], axis = -1)

  x = conv_block(x, (1,1,384,128))
  x = conv_block(x, (3,3,128, 256))
  x = conv_block(x, (1,1,256, 128))
  x = conv_block(x, (3,3,128, 256))
  x = conv_block(x, (1,1,256, 128))

  conv_sobj_branch = conv_block(x,(3,3,128, 256))
  conv_sbbox = conv_block(conv_sobj_branch, (1,1,256,3*(NUM_CLASSES+5)),activate= False , batch_norm= False)

  return [conv_sbbox, conv_mbbox, conv_lbbox]



def activate_raw_outputs(model_out, i = 0):

  batch_size = tf.shape(model_out)[0]
  output_size = tf.shape(model_out)[1]

  model_out = tf.reshape(model_out, (batch_size, output_size,output_size, 3,5+NUM_CLASSES))

  txty = model_out[:,:,:,:,0:2]
  twth = model_out[:,:,:,:,2:4]
  raw_object_conf = model_out[:,:,:,:,4:5]
  raw_class_prob = model_out[:,:,:,:,5:]

  x = tf.tile(tf.range(output_size,dtype=tf.int32)[tf.newaxis,:],[output_size,1])
  y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])

  xy_grid = tf.concat([x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis = -1)
  xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
  xy_grid = tf.cast(xy_grid,tf.float32)

  bxby = (tf.sigmoid(txty)+xy_grid)*STRIDES[i]
  bwbh = (tf.exp(twth)*ANCHORS[i])*STRIDES[i]
  xywh = tf.concat([bxby,bwbh], axis = -1)

  object_conf = tf.sigmoid(raw_object_conf )
  class_prob = tf.sigmoid(raw_class_prob)

  return tf.concat([xywh, object_conf, class_prob], axis = -1)



def Model():
  
  input_layer = tf.keras.layers.Input([416,416,3])
  feature_maps = yolo_v3(input_layer) 
  bbox_tensors = [] 

  for i , fm in enumerate(feature_maps):
    bbox_tensor = activate_raw_outputs(fm, i)
    bbox_tensors.append(bbox_tensor)  

  model = tf.keras.Model(input_layer, bbox_tensors)
  model = load_weights(model) 
  return model


if __name__ == "__main__":
  model = Model()
  print(model.summary())