"""
Module to store all the hyperparameters of the model
"""

import numpy as np

NUM_CLASSES = 80                                                 # Number of Classes that the model is goin to detect

STRIDES = np.array([8,16,32])                                    # Final stride for each scale detection (Total 3 scale detections)

ANCHORS = (1.25,1.625, 2.0,3.75,                                 # Anchor Boxes (Defined by Height and Width only)
  4.125,2.875, 1.875,3.8125, 3.875,2.8125, 
  3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 
  11.65625,10.1875
)
ANCHORS = np.array(ANCHORS).reshape(3,3,2)