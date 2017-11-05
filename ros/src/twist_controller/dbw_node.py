#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane, Waypoint
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5.)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.) #Max steering wheel angle in radians

        min_speed = 0.0
        self.brake_max_torque = abs(decel_limit*vehicle_mass*wheel_radius)
        self.brake_deadband_perc = abs(brake_deadband/decel_limit)
        self.brake_deadband = brake_deadband

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
    #Set class variables to default values
        self.reset()
        # TODO: Create `TwistController` object
        self.controller = Controller(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)        

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, callback=self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, callback=self.twist_cmd_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, callback=self.dbw_enabled_cb, queue_size=1)
        #rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        self.loop()

    def reset(self):
        self.linear_velocity = None
        self.angular_velocity = None
        self.current_linear = None
        self.pose = None
        self.final_waypoints = None
        self.dbw_enabled = None
        self.time_elapsed = 0.0
        self.previous_time = rospy.get_time()

        self.previous_linear = None
        self.decel = 0.0

    #Callback functions
    def current_velocity_cb(self, msg):
        self.current_linear = msg.twist.linear.x

    def twist_cmd_cb(self, msg):
        self.linear_velocity = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

    # def pose_cb(self, msg):
    #     pass
    #     self.pose = [msg.pose.position.x, msg.pose.position.y]

    # def waypoints_cb(self, msg):
    #     pass
    #     self.final_waypoints = [[msg.waypoints[0].pose.pose.position.x,msg.waypoints[0].pose.pose.position.y],[msg.waypoints[5].pose.pose.position.x,msg.waypoints[5].pose.pose.position.y]]

    """
    Primary Logic for drive-by-wire node.

    This logic will call the Controller object with the current state information (speed, etc) to obtain throttle, brake, and steering commands.

    If DBW is enabled, then the values for throttle, braking, and steering are published.

    If DBW becomes disabled, then all values are reset.

    """
    def loop(self):
        rate = rospy.Rate(5) #Was originally set to 50Hz
        while not rospy.is_shutdown():
        #Only publish if drive-by-wire is enabled
            if self.dbw_enabled:
                self.time_elapsed = rospy.get_time() - self.previous_time if self.previous_time else 0.0
                self.previous_time = rospy.get_time()
                self.decel = (self.previous_linear-self.current_linear)/self.time_elapsed if self.previous_linear and self.time_elapsed > 0.0 else 0.0
                self.previous_linear = self.current_linear
                throttle, brake, steer = self.controller.control(self.linear_velocity, self.angular_velocity, self.current_linear, self.pose, self.final_waypoints, self.dbw_enabled, self.time_elapsed)
                #If throttle needs to applied or the % of max brakeis less than the deadband % than not apply breaking
                if throttle > 0.0 or (brake < self.brake_deadband_perc and self.current_linear > 0.5): brake = 0.0
                #if throttle > 0.0 or (self.decel < self.brake_deadband and self.current_linear > 0.5): brake = 0.0
                else: brake = self.brake_max_torque*brake
                
                rospy.loginfo('linear: %s, current: %s', self.linear_velocity, self.current_linear)
                rospy.loginfo('angular: %s, decel: %s, time: %s', self.angular_velocity, self.decel, self.previous_time)
                rospy.loginfo('throttle: %s, brake: %s, steer: %s', throttle, brake, steer)

                self.publish(throttle, brake, steer)
            else: self.reset()
            rate.sleep()
            

    def publish(self, throttle, brake, steer):
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)
    
        if throttle > 0.0:
            tcmd = ThrottleCmd()
            tcmd.enable = True
            tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
            tcmd.pedal_cmd = throttle
            self.throttle_pub.publish(tcmd)
        else:
            bcmd = BrakeCmd()
            bcmd.enable = True
            bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
            bcmd.pedal_cmd = brake
            self.brake_pub.publish(bcmd)

if __name__ == '__main__':
    DBWNode()
