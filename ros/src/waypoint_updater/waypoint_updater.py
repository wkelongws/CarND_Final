#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
from std_msgs.msg import Int32

import math
import tf
from copy import deepcopy
from scipy.interpolate import CubicSpline
import random

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # 50 # Number of waypoints we will publish. You can change this number


STOP_DIST = 2. # (in meters) Distance from closest traffic light to decide whether to stop or go through intersection
#MAX_SPEED = 2.77# meters/second = 10 kph  #11.11# meters/second = 40 kph
#SLOW_DIST = self.max_speed*4 # (in meters) Distance from closest traffic light must be for car to start slowing down
ACCEL = 1.0 # Velocity increment (m/s) to be applied at each waypoint 



class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.base_waypoints = None
        self.final_waypoints = None
        self.current_pose = None
        self.next_waypoint_index = None
        self.light_wp = None
        self.max_speed = None
        self.slow_dist = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, callback=self.current_velocity_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()
        #rospy.spin()

    def pose_cb(self, msg):
        self.current_pose = msg.pose
    
    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints
        if self.base_waypoints:
            self.wp_sub.unregister()
            self.wp_sub = None
            self.max_speed = self.get_waypoint_velocity(self.base_waypoints[0]) #Get velocity from waypoint_loader
            self.slow_dist = self.max_speed*4

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message
        self.light_wp = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def loop(self):
        # Set the frequency of the waypoint updater to run at 5Hz
        rate = rospy.Rate(5)
        
        # While ROS is running
        while not rospy.is_shutdown():
            is_initialized = self.base_waypoints and self.current_pose and self.light_wp
            
            # If we're initialized
            if is_initialized: # and self.current_velocity:

                # Get the index of the closest waypoint.    
                start_idx = self.closest_waypoint(self.current_pose.position)

                # If this waypoint is behind the current pose then update to next waypoint
                self.next_waypoint_index = self.ahead_waypoint(start_idx)
                
                # Set the waypoints' speeds and publish results
                self.publish()
            rate.sleep()

    """ Publish the final_waypoints """
    def publish(self):

        # Create Lane object and set timestamp
        final_waypoints_msg = Lane()
        final_waypoints_msg.header.stamp = rospy.Time.now()

        # Update waypoints and set their velocities. 
        self.set_final_waypoints()
        self.set_final_waypoints_speed()

        final_waypoints_msg.waypoints = self.final_waypoints

        # Publish results
        self.final_waypoints_pub.publish(final_waypoints_msg)

    """ Get the next LOOKAHEAD_WPS waypoints """
    def set_final_waypoints(self):
        # Grab the next LOOKAHEAD_WPS waypoints from the base waypoints.
        self.final_waypoints = deepcopy(self.base_waypoints[self.next_waypoint_index: self.next_waypoint_index + LOOKAHEAD_WPS])

        # If wraparound occurs, handle it by grabbing waypoints from beginning of base waypoints. 
        rem_points = LOOKAHEAD_WPS - len(self.final_waypoints)
        if rem_points > 0: 
            self.final_waypoints = self.final_waypoints + deepcopy(self.base_waypoints[0:rem_points])
    
    """ Set the waypoint speeds based on traffic light information """
    def set_final_waypoints_speed(self):
        
        # Get the distance between the next waypoint and the traffic light.
        dist = self.distance(self.base_waypoints, self.next_waypoint_index, abs(self.light_wp))

        no_red_light_detected = self.light_wp < 0
        if no_red_light_detected:
            # rospy.logwarn("Upcoming TL waypoint %d" % -self.light_wp)
            # rospy.logwarn("No red light detected")
            state_id = random.randint(1,3)
            if state_id == 1:
                rospy.logwarn("No red light detected")
            elif state_id == 2:
                rospy.logwarn("Happy driving ~")
            else:
                rospy.logwarn("Go Go Go ~")
            # Set all waypoint velocities to MAX SPEED.
            #for wp in self.final_waypoints: 
            #   self.set_waypoint_velocity(wp, self.max_speed) #Accelelerate to top speed
            #    return
            speed = self.current_velocity

            target_speed = self.max_speed

            # Creep up on the target mode...  Unfortunately, it causes us to creep once the light turns green too since we haven't yet passed the light.
            Creep_Enabled = False
            if Creep_Enabled and dist < self.slow_dist:
                target_speed = self.max_speed / 2.0
            

            for wp in self.final_waypoints:
                if speed < target_speed:                
                    speed = min(target_speed, speed+ACCEL)
                else:
                    speed = max(target_speed, speed-ACCEL)
                self.set_waypoint_velocity(wp, speed) #Accelelerate to top speed
            
            return

        # Otherwise, a red light was detected.

        

        # Report message of upcoming traffic light.
        # rospy.logwarn("Next wp: %s, Coming redlight wp: %s, distance: %s",self.next_waypoint_index, self.light_wp, dist)
        rospy.logwarn("Attention, Coming redlight in %s meters", dist)

        # If we're within stopping distance, then set the future waypoints to 0 velocity to stop.
        if dist <= self.slow_dist: 
            speed = max(self.current_velocity,max(dist*0.2,0.25))
            if dist > STOP_DIST:
                decel = speed / (dist - STOP_DIST)
                # rospy.logwarn("Decelerating to stop light with : %f",decel)
            else:
                speed = 0
                decel = 0
                # rospy.logwarn("Stopped")            
            for wp in self.final_waypoints:
                speed = max(0, speed - decel)
                self.set_waypoint_velocity(wp, speed) #Accelelerate to top speed

        else: 
            # Otherwise, keep going at top-speed.
            speed = self.current_velocity
            # if speed < self.max_speed:
            #     rospy.logwarn("Accelerating to top speed again...")
            for wp in self.final_waypoints:
                if speed > self.max_speed:              
                    speed = min(self.max_speed, speed+ACCEL)
                else:
                    speed = max(self.max_speed, speed-ACCEL)
                self.set_waypoint_velocity(wp, speed) #Accelelerate to top speed
    

    """ Get the velocity at the specified waypoint index """
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    """ Set the velocity at the specified waypoint index """ 
    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity


    """ Get the piece-wise distance from some waypoint indices wp1 to wp2 """  
    def distance(self, waypoints, wp1, wp2):
        dist, wp3 = 0.0, -1
        #In case of wraparound
        if wp2 < wp1: wp3, wp2 = wp2, len(waypoints)-1
        for i in xrange(wp1,wp2):
            dist += self.distance_between_points(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        for i in xrange(-1,wp3):
            dist += self.distance_between_points(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        return dist

    """ Euclidean distance function """
    def distance_between_points(self, a, b):
        dist = math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        return dist

    """ Retrieve the closest waypoint.  Note: this waypoint index may be behind the vehicle. """
    def closest_waypoint(self, position):
        return min(xrange(len(self.base_waypoints)), key = lambda p: self.distance_between_points(position, self.base_waypoints[p].pose.pose.position))


    """ Utility to get yaw angle """
    def get_euler_yaw(self):
        quaternion = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]


    """ Get the next upcoming waypoint index """
    def ahead_waypoint(self, wp_idx):
        ahead_idx = wp_idx

        # Get coordinates of the waypoint at the specified index.
        map_wp_x = self.base_waypoints[wp_idx].pose.pose.position.x
        map_wp_y = self.base_waypoints[wp_idx].pose.pose.position.y

        # Get current position
        x,y = self.current_pose.position.x, self.current_pose.position.y

        # Get yaw angle 
        yaw = self.get_euler_yaw()

        # Compute expression to determine if closest waypoint is behind us or not. 
        localize_x = (map_wp_x - x) * math.cos(yaw) + (map_wp_y - y) * math.sin(yaw)
        
        # If the waypoint is behind us, then increment the waypoint index.
        if localize_x < 0.0: 
            ahead_idx = ahead_idx + 1

        # Set the internal property
        self.ahead_waypoint_index = ahead_idx

        # And return the index as a return argument.
        return ahead_idx

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
