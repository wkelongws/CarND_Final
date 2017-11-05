#!/usr/bin/env python
import rospy
import math
import tf
import cv2
import yaml
from std_msgs.msg import Int32
from styx_msgs.msg import Lane
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.slps = []
        self.listener = None
              
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.waypoints = rospy.wait_for_message('/base_waypoints', Lane).waypoints # Only need to get base_waypoints once
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # List of positions that correspond to the line to stop in front of a given intersection. Convert to more appropriate format
        for slp in self.config['stop_line_positions']:
            tl = TrafficLight()
            tl.pose.pose.position.x, tl.pose.pose.position.y, tl.pose.pose.position.z = slp[0], slp[1], 0
            self.slps.append( self.get_closest_waypoint(tl.pose.pose.position) )

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
 
        self.light_classifier = TLClassifier()
       
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        if not self.lights or not self.waypoints or not self.pose: pass

        light_wp, state = self.process_traffic_lights()
        if state == TrafficLight.UNKNOWN:
            state_str = 'Green/Yellow'
        else:
            state_str = 'Red'

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # Does the detected state not match the previous state?
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1 # (-light_wp)
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, position):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        return min( xrange(len(self.waypoints)), key = lambda p: self.distance_between_points(position, self.waypoints[p].pose.pose.position) )

    def distance_between_points(self, a, b):
        dist = math.sqrt( (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2 )
        return dist

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")e

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        
        if(not self.has_image):
                    self.prev_light_loc = None
                    return TrafficLight.RED

        # fixing convoluted camera encoding...
        if hasattr(self.camera_image, 'encoding'):
            self.attribute = self.camera_image.encoding
            if self.camera_image.encoding == '8UC3':
                self.camera_image.encoding = "rgb8"
        else:
            self.camera_image.encoding = 'rgb8'
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        if (self.listener is not None):
            x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        if (self.waypoints is None):
            return -1, TrafficLight.UNKNOWN
                
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position)

        #TODO find the closest visible traffic light (if one exists)
        #Finding the closest light infront
            func = lambda p: self.slps[p]-car_position if self.slps[p] >= car_position else len(self.waypoints) + self.slps[p]-car_position
            idx = min(xrange(len(self.slps)), key = func)
            light_wp = self.slps[idx]
            #light = self.stop_line_positions[idx]
            light = self.lights[idx]
        if light:
            state = light.state #Remove once traffic light classifier is implemented
            state = self.get_light_state(light)
            return light_wp, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
