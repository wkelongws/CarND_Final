from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
from numpy import zeros, newaxis
import rospkg
import numpy as np
import tensorflow as tf 
class TLClassifier(object):
    def __init__(self):
        
        
        ros_root = rospkg.get_ros_root()

        r = rospkg.RosPack()
        path = r.get_path('tl_detector')
        print(path)
        self.model = load_model(path + '/model.h5') 

        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        #print(model)
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        imrs = cv2.resize(image, (400, 400)) 
        imrs = imrs.astype(float)
        imrs = imrs / 255.0

        imrs = imrs[newaxis,:,:,:]
        #print(imrs)
        with self.graph.as_default():
            preds = self.model.predict(imrs)
        #print('Predicted:' ,preds)
        predicted_class = np.argmax(preds, axis=1)

        # print('Predicted Class:' ,predicted_class[0])
        lid = predicted_class[0]

        if(lid == 1):
           return TrafficLight.RED

        return TrafficLight.UNKNOWN
