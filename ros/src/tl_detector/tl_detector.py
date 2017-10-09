#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import math
import cv2
import yaml
import numpy as np
from scipy import spatial

STATE_COUNT_THRESHOLD = 2
MAX_DECEL = 1.

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.KDTree = None
        self.camera_image = None
        self.lights = []
        self.stop_lines = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        sub7 = rospy.Subscriber('/next_wp', Int32, self.next_wp_cb, queue_size=1)
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.light_classifier.init()
        
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.next_wp = None
        rospy.spin()

    def next_wp_cb(self, val):
        self.next_wp = val.data

    def pose_cb(self, msg):
        self.pose = msg
            
    def waypoints_cb(self, lane):
        if self.waypoints != lane.waypoints:
            data = []
            for wp in lane.waypoints:
                data.append((wp.pose.pose.position.x, wp.pose.pose.position.y))
            self.KDTree = spatial.KDTree(data)
            self.waypoints = lane.waypoints
            self.stop_lines = None

    def find_stop_line_position(self, light):
        stop_line_positions = self.config['stop_line_positions']
        min_distance = 100000
        result = None
        light_pos = light.pose.pose.position
        for pos in stop_line_positions:
            distance = self.euclidean_distance_2d(pos, light_pos)
            if (distance < min_distance):
                min_distance = distance
                result = pos
        return result

    def traffic_cb(self, msg):
        if not self.stop_lines and self.KDTree:
            stop_lines = []
            for light in msg.lights:
                stop_line_pos = self.find_stop_line_position(light)
                closest_index = self.KDTree.query(np.array([stop_line_pos]))[1][0]
                closest_wp = self.waypoints[closest_index]
                if not self.is_ahead(closest_wp.pose.pose, stop_line_pos):
                    closest_index = max(closest_index - 1, 0)
                stop_lines.append(closest_index)
            self.lights = msg.lights
            self.stop_lines = stop_lines

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state in [TrafficLight.RED, TrafficLight.YELLOW] else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        state = self.light_classifier.get_classification(cv_image)
        if state == TrafficLight.UNKNOWN and self.last_state:
            state = self.last_state
        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None
        light_wp = -1

        if self.waypoints and 
            self.next_wp and 
            self.stop_lines:
            next_wp = self.waypoints[min(self.next_wp, len(self.waypoints)-1)]
            target_velocity = next_wp.twist.twist.linear.x
            search_distance = target_velocity * target_velocity / 2 / MAX_DECEL
            min_distance = search_distance
            for i in range(len(self.stop_lines)):
                stop_line_wp_index = self.stop_lines[i]
                if stop_line_wp_index >= self.next_wp:
                    stop_line_wp = self.waypoints[stop_line_wp_index]
                    distance = self.euclidean_distance_2d(next_wp.pose.pose.position, stop_line_wp.pose.pose.position)
                    if distance < min_distance:
                        light_wp = stop_line_wp_index
                        light = self.lights[i]

        if light_wp > -1:
            state = self.get_light_state(light)
            return light_wp, state
        return -1, TrafficLight.UNKNOWN

    def is_ahead(self, origin_pose, test_position):
        test_x = self.get_x(test_position)
        test_y = self.get_y(test_position)

        orig_posit = origin_pose.position
        orig_orient = origin_pose.orientation
        quaternion = (orig_orient.x, orig_orient.y, orig_orient.z, orig_orient.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        orig_x = ((test_x - orig_posit.x) * math.cos(yaw) \
                + (test_y - orig_posit.y) * math.sin(yaw))
        return orig_x > 0.

    def euclidean_distance_2d(self, position1, position2):
        a_x = self.get_x(position1)
        a_y = self.get_y(position1)
        b_x = self.get_x(position2)
        b_y = self.get_y(position2)
        return math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)

    def get_x(self, pos):
        return pos.x if isinstance(pos, Point) else pos[0] 
    def get_y(self, pos):
        return pos.y if isinstance(pos, Point) else pos[1] 

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
