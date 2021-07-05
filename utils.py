import tensorflow as tf 
import numpy as np 
import cv2


def get_class_names(class_file_path):
    names = {}
    with open(class_file_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def box_detector(pred):
    
    center_x,center_y,width,height,obj_confidence,classes = tf.split(pred,[1,1,1,1,1,-1], axis=-1)
    top_left_x=(center_x-width/2.)/ 416                # Normalized Coordinates
    top_left_y = (center_y - height/2.)/416.0          # Normalized Coordinates
    bottom_right_x = (center_x + width/2.)/416.0       # Normalized Coordinates
    bottom_right_y = (center_y + height/2.)/416.0      # Normalized Coordinates

    bboxes = tf.concat([top_left_y,top_left_x,bottom_right_y,bottom_right_x],axis=-1)
    scores = obj_confidence*classes
    scores = np.array(scores)

    scores = scores.max(axis=-1)
    class_index = np.argmax(classes, axis=-1)

    selected_indices = tf.image.non_max_suppression(bboxes,scores, max_output_size= 20)
    # selected_indices = np.array(selected_indices)
    class_index = tf.gather(class_index, selected_indices)
    # class_names = class_index[selected_indices]
    bboxes = tf.gather(bboxes,selected_indices)
    scores = tf.gather(scores, selected_indices)
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    class_index = np.array(class_index)
    # class_names = np.array(class_names)
    # boxes = boxes[selected_indices,:]

    # scores = scores[selected_indices]
    bboxes = bboxes*416                       # Denormalize the coordinates

    return bboxes ,class_index, scores


def drawbox(boxes, class_index,scores,names,img):
    data = np.concatenate([boxes,scores[:,np.newaxis],class_index[:,np.newaxis]],axis=-1)
    data = data[np.logical_and(data[:, 0] >= 0, data[:, 0] <= 416)]
    data = data[np.logical_and(data[:, 1] >= 0, data[:, 1] <= 416)]
    data = data[np.logical_and(data[:, 2] >= 0, data[:, 2] <= 416)]
    data = data[np.logical_and(data[:, 3] >= 0, data[:,3] <= 416)]
    data = data[data[:,4]>0.4]                       # Score Threshold

    img = cv2.resize(img, (416, 416))
    for i,row in enumerate(data):
        img=cv2.rectangle(img,(int(row[1]),int(row[0])),(int(row[3]),int(row[2])) ,(51, 51, 255),1)    # Boxes are in the format (y1,x1, y2,x2), Color is in BGR format
        img = cv2.putText(img,(names[row[5]]+": "+"{:.4f}".format(row[4])),(int(row[1]),int(row[0])),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255,0 ),1)
    return  img