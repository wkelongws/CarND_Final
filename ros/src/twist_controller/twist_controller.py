from math import atan2, sin, cos
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
from rospy import logwarn

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.max_angle = max_steer_angle
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
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_linear)

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
