from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
from rospy import logwarn

from math import atan2, sin, cos

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

#Got this value from the slack channel
#Will remove hardcoded number at a later date if it is useless and replace with the parameter from ROS
class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.max_angle = max_steer_angle
        #Using the Udacity provided controller for steering. It doesn't need to to be reset after initialization
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.set_controllers()

    def control(self,linear_velocity, angular_velocity, current_linear, pose, way_points, dbw_enabled, time_elapsed):
    #Reset if drive by wire is not enabled and return 0's
        if not dbw_enabled:
            self.set_controllers()
            return 0.0, 0.0, 0.0

    #Calculate CTE for throttle
        cte = linear_velocity - current_linear
        throttle = self.pid_throttle.step(cte, time_elapsed)
        brake = self.pid_brake.step(-cte, time_elapsed)

    #Steer uses provided yaw controller
        #angular_velocity = self.lowpass_steer.filt(angular_velocity)
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_linear)
    #Calculate lateral cross track error for steering control
    #vector = [way_points[1][0] - way_points[0][0], way_points[1][1] - way_points[0][1]]
    #angle = atan2(vector[1], vector[0])
    #cte = -sin(angle)*(pose[0] - way_points[0][0]) + cos(angle)*(pose[1] - way_points[0][1])
    #pid_steer_output = self.pid_steer.step(-cte, time_elapsed)
    #steer = max( min( steer + pid_steer_output, self.max_angle), -self.max_angle)

    #Apply low pass filter to smooth out steering
        steer = self.lowpass_steer.filt(steer)
    # Return throttle, brake, steer
        return throttle, brake, steer

    def set_controllers(self):
        #PID Controllers for throttle and brake
        self.pid_throttle = PID(0.35,0.0,0.0,0.0,1.0)
        self.pid_brake = PID(0.3,0.0,0.0,0.0,1.0)
        #Low pass filter to smooth out the steering
        self.lowpass_steer = LowPassFilter(0.2,1.0)
