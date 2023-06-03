#!/usr/bin/env python
import os
import rospy
import numpy as np
import tf
import math
import time
import struct

import actionlib

from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs import point_cloud2

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
# from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseGoal

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

from std_msgs.msg import Header

from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker

# Parameters

# # ACRE-Sorghum
# intensity_threshold = 100                                # Change
# # forward_radius_threshold = 1.2                                # Change
# forward_radius_threshold = 1.6                                # Change

# backward_radius_threshold = 2.2
# # line_extraction_period = 10    # per 4s = 20/(300rpm/60)

# angle_resolution = 1    # 360 / 360
# # dx = 1
# velocity = 0.03     # 0.15
# dx = velocity
# theta_weight = 0.06 #0.75  #0.6 #0.5
# near_area_dist = 0.4
# next_goal_dist_threshold = 0.2#0.21
# angular_velocity_in_place = 0.8
# direction_threshold = -0.8

# micro_control_weight = 0.5

# forward_angle_start = 40
# forward_angle_range = 45   # angle(degree) = angle_range * angle_resolution

# backward_angle_start = 100  #100
# backward_angle_range = 60   #70 # angle(degree) = angle_range * angle_resolution


# ACRE-Sorghum
# intensity_threshold = 100                                # Change
# # forward_radius_threshold = 1.2                                # Change
# forward_radius_threshold = 1.6#1.6                                # Change

# backward_radius_threshold = 2.2
# # line_extraction_period = 10    # per !4s = 20/(300rpm/60)

# angle_resolution = 1    # 360 / 360
# dx = 1       # 0.08
# velocity = 0.1     # 0.15
# theta_weight = 1 #0.75  #0.6 #0.5
# near_area_dist = 0.5
# next_goal_dist_threshold = 0.22
# angular_velocity_in_place = 1.0
# direction_threshold = -0.8

# micro_control_weight = 0.6

# forward_angle_start = 30
# forward_angle_range = 90   # angle(degree) = angle_range * angle_resolution

# backward_angle_start = 80  #100
# backward_angle_range = 20  #30 #70 # angle(degree) = angle_range * angle_resolution

# ACRE-Corn
intensity_threshold = 100  # Change
# forward_radius_threshold = 1.2                                # Change
forward_radius_threshold = 1.6  # Change

backward_radius_threshold = 2.2
# line_extraction_period = 10    # per 4s = 20/(300rpm/60)

# angle_resolution = 1    # 360 / 360
# dx = 1
velocity = 0.3  # 0.15
dx = velocity
theta_weight = 0.05  # 0.75  #0.6 #0.5
near_area_dist = 0.4
next_goal_dist_threshold = 0.2  # 0.21
angular_velocity_in_place = 0.8
direction_threshold = -0.8

micro_control_weight = 0.5

# forward_angle_start = 10
# forward_angle_range = 75   # angle(degree) = angle_range * angle_resolution

# backward_angle_start = 190 #100  #100
# backward_angle_range = 60   #70 # angle(degree) = angle_range * angle_resolution

MAX_HEIGHT = -0.1
MIN_HEIGHT = -0.3  # -0.2

# Define arrays
pose_robot = np.array([0, 0])
cur_dir_robot = np.array([1, 0])

dist_txt = os.path.join('/home/iot4ag/Desktop', "dist.txt")
a_file = open(dist_txt, "w")
pose_txt = os.path.join('/home/iot4ag/Desktop', "pose.txt")
b_file = open(pose_txt, "w")


# trans_pose = [0, 0, 0]
# rot_pose = [0, 0, 0, 0]

# Row tracer class
class NaviIntegration:

    # AMCL pose callback function
    def PoseListener(self):

        # listener = tf.TransformListener()

        # (trans,rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))

        # print(trans)
        # print(rot)

        # localization pose based on AMCL
        self.amcl_pose_x = self.trans_pose[0]
        self.amcl_pose_y = self.trans_pose[1]

        np.savetxt(b_file, [self.amcl_pose_x], fmt='%f')
        np.savetxt(b_file, [self.amcl_pose_y], fmt='%f')

        # localization pose direction based on AMCL
        amcl_orientation_x = self.rot_pose[0]
        amcl_orientation_y = self.rot_pose[1]
        amcl_orientation_z = self.rot_pose[2]
        amcl_orientation_w = self.rot_pose[3]

        # # localization pose based on AMCL
        # self.amcl_pose_x = amclpose.pose.pose.position.x
        # self.amcl_pose_y = amclpose.pose.pose.position.y

        # # localization pose direction based on AMCL
        # amcl_orientation_x = amclpose.pose.pose.orientation.x
        # amcl_orientation_y = amclpose.pose.pose.orientation.y
        # amcl_orientation_z = amclpose.pose.pose.orientation.z
        # amcl_orientation_w = amclpose.pose.pose.orientation.w

        # Change from Quarternion to Euler angle (x, y, z, w -> roll, pitch, yaw)
        amcl_orientation_list = [amcl_orientation_x, amcl_orientation_y, amcl_orientation_z, amcl_orientation_w]
        (self.amcl_roll, self.amcl_pitch, self.amcl_yaw) = tf.transformations.euler_from_quaternion(
            amcl_orientation_list)

        # print("AMCL", self.amcl_roll, self.amcl_pitch, self.amcl_yaw)

    # Goal info callback function
    def GoalCallback(self, goal):

        # Goal position
        self.goal_pose_x = goal.goal.target_pose.pose.position.x
        self.goal_pose_y = goal.goal.target_pose.pose.position.y

        # Goal orientation (Quarternion)
        goal_orientation_x = goal.goal.target_pose.pose.orientation.x
        goal_orientation_y = goal.goal.target_pose.pose.orientation.y
        goal_orientation_z = goal.goal.target_pose.pose.orientation.z
        goal_orientation_w = goal.goal.target_pose.pose.orientation.w

        # Change from Quarternion to Euler angle (x, y, z, w -> roll, pitch, yaw)
        goal_orientation_list = [goal_orientation_x, goal_orientation_y, goal_orientation_z, goal_orientation_w]
        (self.goal_roll, self.goal_pitch, self.goal_yaw) = tf.transformations.euler_from_quaternion(
            goal_orientation_list)

    # Laser scan callback function
    # def laserCallback(self, laserscan):

    #     # Compute the distance between the robot and the surroundings
    #     # self.direction_function()

    #     # print("size", len(laserscan.ranges))
    #     # print("left: ", laserscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start])
    #     # print("Intensity: ", laserscan.intensities)
    #     #print("##################")

    #     (self.trans_pose,self.rot_pose) = self.listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
    #     # (self.trans_pose,self.rot_pose) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))

    #     self.PoseListener()

    #     # Compute the distance between the robot and the surroundings
    #     self.direction_function()

    #     if self.dot_dir > 0:    # Forward
    #         # Only the laser scans in the front side

    #         # print("left: ", laserscan.ranges)
    #         # print("right: ", laserscan.ranges[-300:-1])

    #         # self.left_laser_ranges = laserscan.ranges[forward_angle_start:forward_angle_start+forward_angle_range]   # 0 ~ 360
    #         # self.left_laser_intensities = laserscan.intensities[forward_angle_start:forward_angle_start+forward_angle_range]

    #         # self.right_laser_ranges = laserscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start]
    #         # self.right_laser_intensities = laserscan.intensities[-(forward_angle_start+forward_angle_range):-forward_angle_start]

    #         # Updated (11/29/2022)
    #         self.left_laser_ranges = laserscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start]   # 0 ~ 360
    #         self.left_laser_intensities = laserscan.intensities[-(forward_angle_start+forward_angle_range):-forward_angle_start]

    #         self.right_laser_ranges = laserscan.ranges[forward_angle_start:forward_angle_start+forward_angle_range]   # 0 ~ 360
    #         self.right_laser_intensities = laserscan.intensities[forward_angle_start:forward_angle_start+forward_angle_range]

    #         print(laserscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range])

    #         # print(self.left_laser_ranges)
    #         # print("left: ", forward_angle_start)
    #         # print("right: ", -forward_angle_start)

    #     else:                   # Backward
    #         # Only the laser scans in the front side
    #         self.left_laser_ranges = laserscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range]   # 0 ~ 360
    #         self.left_laser_intensities = laserscan.intensities[backward_angle_start:backward_angle_start+backward_angle_range]

    #         self.right_laser_ranges = laserscan.ranges[-(backward_angle_start+backward_angle_range):-backward_angle_start]
    #         self.right_laser_intensities = laserscan.intensities[-(backward_angle_start+backward_angle_range):-backward_angle_start]

    #         #print(laserscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range])
    #         # print("**********", laserscan.ranges)
    #         # print("left: ", laserscan.ranges[-(backward_angle_start+backward_angle_range):-backward_angle_start])

    #         # print("left: ", self.left_laser_ranges)

    #         # print("right: ", self.right_laser_ranges)

    #     # print(self.right_laser_ranges)

    #     self.dist_computation()

    #     self.move()

    # Laser scan callback function
    def lidarCallback(self, lidarscan):

        # Compute the distance between the robot and the surroundings
        # self.direction_function()

        # print("size", len(lidarscan.ranges))
        # print("left: ", lidarscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start])
        # print("Intensity: ", lidarscan.intensities)

        (self.trans_pose, self.rot_pose) = self.listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
        # (self.trans_pose,self.rot_pose) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))

        self.PoseListener()

        # Compute the distance between the robot and the surroundings
        self.direction_function()

        assert isinstance(lidarscan, PointCloud2)
        # cloud_points = [point_cloud2.read_points(lidarscan, skip_nans=True, field_names = ("x", "y", "z"))]
        cloud_points = list(point_cloud2.read_points(lidarscan, skip_nans=True, field_names=("x", "y", "z")))

        # NEED TO MODIFY DEPENDING ON THE DRIVING DIRECTION
        # (1) Filtering the pointclouds which are useful for autonomous navigation
        if self.dot_dir > 0:  # Forward
            # points = [list(cloud_point) for cloud_point in cloud_points if (np.sqrt(pow(cloud_point[0],2)+pow(cloud_point[1],2)) < 8 and cloud_point[0] <= 1 and cloud_point[2] < MAX_HEIGHT and cloud_point[2] > MIN_HEIGHT)]
            points = [list(cloud_point) for cloud_point in cloud_points if
                      cloud_point[0] <= 0 and cloud_point[2] < MAX_HEIGHT and cloud_point[2] > MIN_HEIGHT]
        else:  # Backward
            # points = [list(cloud_point) for cloud_point in cloud_points if (np.sqrt(pow(cloud_point[0],2)+pow(cloud_point[1],2)) < 8 and cloud_point[0] >= 0 and cloud_point[2] < MAX_HEIGHT and cloud_point[2] > MIN_HEIGHT)]
            points = [list(cloud_point) for cloud_point in cloud_points if
                      cloud_point[0] >= 0 and cloud_point[2] < MAX_HEIGHT and cloud_point[2] > MIN_HEIGHT]
            # points = [list(cloud_point) for cloud_point in cloud_points if (cloud_point[0] >= -3 and cloud_point[0] <= 3 and cloud_point[2] < MAX_HEIGHT and cloud_point[2] > MIN_HEIGHT)]

        # (2) Clustering the pointclouds for each rows
        clustering = DBSCAN(eps=0.3, min_samples=3).fit_predict(points)
        num_clusters = max(clustering) + 1

        self.cluster_list = [[] for _ in range(num_clusters)]

        for i in range(len(points)):
            if clustering[i] != -1:
                self.cluster_list[clustering[i]].append(points[i])

        # print(clusters_list[0])
        # print(num_clusters)

        #############################################################
        # To check: Visualization
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgb', 12, PointField.UINT32, 1),
                  # PointField('rgba', 12, PointField.UINT32, 1),
                  ]

        header = Header()
        header.frame_id = "velodyne1"
        pc2_1 = point_cloud2.create_cloud(header, fields, points)
        # pc2_2 = point_cloud2.create_cloud(header, fields, self.cluster_list[0])

        pc2_1.header.stamp = rospy.Time.now()
        # pc2_2.header.stamp = rospy.Time.now()

        pc2_pub1 = rospy.Publisher('/navigation_pc2_1', PointCloud2, queue_size=10)
        pc2_pub1.publish(pc2_1)

        # pc2_pub2 = rospy.Publisher('/navigation_pc2_2', PointCloud2, queue_size=10)
        # pc2_pub2.publish(pc2_2)

        #############################################################

        # if self.dot_dir > 0:    # Forward
        #     # Only the laser scans in the front side

        #     # print("left: ", lidarscan.ranges)
        #     # print("right: ", lidarscan.ranges[-300:-1])

        #     # self.left_laser_ranges = lidarscan.ranges[forward_angle_start:forward_angle_start+forward_angle_range]   # 0 ~ 360
        #     # self.left_laser_intensities = lidarscan.intensities[forward_angle_start:forward_angle_start+forward_angle_range]

        #     # self.right_laser_ranges = lidarscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start]
        #     # self.right_laser_intensities = lidarscan.intensities[-(forward_angle_start+forward_angle_range):-forward_angle_start]

        #     # Updated (11/29/2022)
        #     self.left_laser_ranges = lidarscan.ranges[-(forward_angle_start+forward_angle_range):-forward_angle_start]   # 0 ~ 360
        #     self.left_laser_intensities = lidarscan.intensities[-(forward_angle_start+forward_angle_range):-forward_angle_start]

        #     self.right_laser_ranges = lidarscan.ranges[forward_angle_start:forward_angle_start+forward_angle_range]   # 0 ~ 360
        #     self.right_laser_intensities = lidarscan.intensities[forward_angle_start:forward_angle_start+forward_angle_range]

        #     # print(lidarscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range])

        #     # print(self.left_laser_ranges)
        #     # print("left: ", forward_angle_start)
        #     # print("right: ", -forward_angle_start)

        # else:                   # Backward
        #     # Only the laser scans in the front side
        #     self.left_laser_ranges = lidarscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range]   # 0 ~ 360
        #     self.left_laser_intensities = lidarscan.intensities[backward_angle_start:backward_angle_start+backward_angle_range]

        #     self.right_laser_ranges = lidarscan.ranges[-(backward_angle_start+backward_angle_range):-backward_angle_start]
        #     self.right_laser_intensities = lidarscan.intensities[-(backward_angle_start+backward_angle_range):-backward_angle_start]

        # print(laserscan.ranges[backward_angle_start:backward_angle_start+backward_angle_range])
        # print("**********", laserscan.ranges)
        # print("left: ", laserscan.ranges[-(backward_angle_start+backward_angle_range):-backward_angle_start])

        # print("left: ", self.left_laser_ranges)

        # print("right: ", self.right_laser_ranges)

        # print(self.right_laser_ranges)

        self.dist_computation()

        self.move()

    # Class Initialization
    def __init__(self):

        self.listener = tf.TransformListener()

        # rospy.init_node('navi_integration')

        points_seq = rospy.get_param('move_base_seq/p_seq')
        # Only yaw angle required (no rotations around x and y axes) in deg:
        yaweulerangles_seq = rospy.get_param('move_base_seq/yea_seq')

        # List of goal quaternions:
        quat_seq = list()

        # List of goal poses:
        self.pose_seq = list()
        self.goal_cnt = 0

        self.dot_dir_list = [0, 0, 0]

        for yawangle in yaweulerangles_seq:
            # Unpacking the quaternion list and passing it as arguments to Quaternion message constructor
            quat_seq.append(Quaternion(*(quaternion_from_euler(0, 0, yawangle * math.pi / 180, axes='sxyz'))))

        n = 3

        # Returns a list of lists [[point1], [point2],...[pointn]]
        points = [points_seq[i:i + n] for i in range(0, len(points_seq), n)]
        for point in points:
            # Exploit n variable to cycle in quat_seq
            self.pose_seq.append(Pose(Point(*point), quat_seq[n - 3]))
            n += 1

        # Create action client
        # self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        # rospy.loginfo("Waiting for move_base action server...")
        # wait = self.client.wait_for_server(rospy.Duration(0.0))
        # #wait = self.client.wait_for_server()

        # if not wait:
        #     rospy.logerr("Action server not available!")
        #     rospy.signal_shutdown("Action server not available!")
        #     return

        # rospy.loginfo("Connected to move base server")
        # rospy.loginfo("Starting goals achievements ...")

        while not rospy.is_shutdown():
            # rate = rospy.Rate(60) # 60Hz

            # /amcl_pose subscriber
            # self.amclpose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.AmclPoseCallback)
            # self.AmclPoseCallback()
            # /move_base/goal subscriber
            # self.goal_sub = rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.GoalCallback)

            # /move_base_simple/goal subscriber
            # self.goal_pose_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.GoalPoseCallback)

            self.movebase_client()  # Set the goal point

            # se!lf.PoseListener()
            # self.movebase_client()      # Set the goal point

            # # Compute the distance between the robot and the surroundings
            # self.direction_function()

            # /front/scan subscriber
            # self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserCallback)
            self.lidar_sub = rospy.Subscriber("/velodyne1_points", PointCloud2, self.lidarCallback)
            # rospy.Subscriber("/scan", LaserScan, self.laserCallback)
            # print('!!!!')
            # print(self.left_laser_ranges)
            # self.dist_computation()
            # self.move()

            # print(self.amcl_pose_x)
            # self.dist_computation()
            # self.move()

            rospy.spin()  # spin() simply keeps python from exiting until this node is stopped

    # Distance computation function
    def dist_computation(self):

        # (3) Line regression to express each row
        row_list = []
        dist_list = []
        bool_init = 1

        lines_marker = Marker()

        lines_marker.header.frame_id = "velodyne1"
        lines_marker.type = lines_marker.LINE_LIST
        lines_marker.action = lines_marker.ADD

        lines_marker.scale.x = 0.03
        lines_marker.scale.y = 0.03
        lines_marker.scale.z = 0.03

        lines_marker.color.a = 1.0
        lines_marker.color.r = 1.0
        lines_marker.color.g = 1.0
        lines_marker.color.b = 0.0

        lines_marker.pose.orientation.x = 0.0
        lines_marker.pose.orientation.y = 0.0
        lines_marker.pose.orientation.z = 0.0
        lines_marker.pose.orientation.w = 1.0

        lines_marker.pose.position.x = 0.0
        lines_marker.pose.position.y = 0.0
        lines_marker.pose.position.z = 0.0

        for cluster in self.cluster_list:
            if len(cluster) > 100:
                x = np.array([point[0] for point in cluster])
                y = np.array([point[1] for point in cluster])

                slope, intercept = np.polyfit(x, y, 1)
                row_list.append([slope, intercept])

                # (4) Compute the distance from the lidar to each row
                # d = |ax1+by1+c|/sqrt(a^2+b^2)
                dist = intercept / math.sqrt(pow(slope, 2) + 1)
                dist_list.append(dist)

                first_point = Point()
                second_point = Point()

                first_point.x = np.min(x)
                idx_min_x = np.argmin(x)
                first_point.y = y[idx_min_x]

                second_point.x = np.max(x)
                idx_max_x = np.argmax(x)
                second_point.y = y[idx_max_x]

                lines_marker.points.append(first_point)
                lines_marker.points.append(second_point)

            #     print(slope, intercept)
            #     bool_init = 0

        # print("Distance:", dist_list)

        # left_row_list = [[dist, dist_list.index(dist)] for dist in dist_list if dist < 0]
        # right_row_list = [[dist, dist_list.index(dist)] for dist in dist_list if dist > 0]

        line_publisher = rospy.Publisher('/rows', Marker, queue_size=10)
        line_publisher.publish(lines_marker)

        left_row_list = [-dist for dist in dist_list if dist < 0]
        right_row_list = [dist for dist in dist_list if dist > 0]

        # print("LEFT: ", left_row_list)
        # print("RIGHT: ", right_row_list)

        min_left_dist = min(left_row_list)
        min_right_dist = min(right_row_list)

        # list_left_laser = []
        # list_right_laser = []

        # # 1. Choose the laser scans in the interesting area
        # # (1) Left side
        # #print(self.left_laser_ranges)
        # for left_point_range in self.left_laser_ranges:

        #     if self.dot_dir > 0:    # Forward
        #         radius_threshold = forward_radius_threshold
        #     else:
        #         radius_threshold = backward_radius_threshold

        #     if left_point_range < radius_threshold:                 # (i) Near the robot

        #         left_point_idx = self.left_laser_ranges.index(left_point_range)
        #         left_point_intensity = self.left_laser_intensities[left_point_idx]

        #         # if left_point_intensity > intensity_threshold:      # (ii) High intensity         # Change when using lds-01
        #         list_left_laser.append([left_point_idx, left_point_range, left_point_intensity])

        # # print("!!!!!!!!!!", list_left_laser)
        # # (2) Right side
        # for right_point_range in self.right_laser_ranges:

        #     if right_point_range < radius_threshold:                # (i) Near the robot
        #         right_point_idx = self.right_laser_ranges.index(right_point_range)
        #         right_point_intensity = self.right_laser_intensities[right_point_idx]

        #         # if right_point_intensity > intensity_threshold:     # (ii) High intensity        # Change when using lds-01
        #         list_right_laser.append([right_point_idx, right_point_range, right_point_intensity])

        # # 2. Compute the distance between the robot and the chosen laser scans
        # # (1) Left side
        # #print("Left", list_left_laser)
        # #print("Right", list_right_laser)
        # left_trigger = False
        # right_trigger = False

        # for left_point in list_left_laser:
        #     angle = angle_resolution * left_point[0]          # Change
        #     angle = math.radians(angle)
        #     dist = abs(left_point[1] * math.sin(angle))
        #     # print("Left", dist)
        #     if (dist != 0.0):
        #         # continue

        #     # Find the minimum distance
        #         # if left_point == list_left_laser[0]:
        #         if left_trigger == False:
        #             min_left_dist = dist
        #             left_trigger = True
        #         else:
        #             if min_left_dist > dist:
        #                 min_left_dist = dist

        print("left_min: ", min_left_dist)

        # # (2) Right side
        # for right_point in list_right_laser:
        #     angle = angle_resolution * right_point[0]                 # Change
        #     angle = math.radians(angle)
        #     dist = abs(right_point[1] * math.sin(angle))
        #     # print("Left", dist)

        #     if (dist != 0.0):
        #         # continue

        #         # Find the minimum distance
        #         # if right_point == list_right_laser[0]:
        #         if right_trigger == False:
        #             min_right_dist = dist
        #             right_trigger = True
        #         else:
        #             if min_right_dist > dist:
        #                 min_right_dist = dist

        print("right_min: ", min_right_dist)

        np.savetxt(a_file, [min_left_dist], fmt='%f')
        np.savetxt(a_file, [min_right_dist], fmt='%f')

        # 3. Determine the direction of driving
        # Compute the average value of left/right minimum distance
        half_dist = 0.5 * (min_left_dist + min_right_dist)
        diff_dist = min_left_dist - half_dist

        # print("half_dist = :", half_dist)

        # Compute the direction and angular velocity
        self.theta = math.atan(diff_dist / dx)

        # print("diff_dist = :", diff_dist)
        # print("theta = :", self.theta)
        self.rot_mat = np.array(
            [[math.cos(self.theta), -math.sin(self.theta)], [math.sin(self.theta), math.cos(self.theta)]])
        self.goal_dir_robot = np.matmul(self.rot_mat, cur_dir_robot)

        # print("goal direction: ", self.goal_dir_robot)

    # The function to command a linear/angular velocity
    def direction_function(self):

        # print("Goal: ", self.goal_cnt+1)

        # print("***************************************")

        self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]

        # rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
        # rospy.loginfo(str(self.pose_seq[self.goal_cnt]))

        # Compare between directions of AMCL pose and goal
        dir_amcl = np.array([math.cos(self.amcl_yaw), math.sin(self.amcl_yaw)])
        dir_robot_goal = np.array([self.goal.target_pose.pose.position.x - self.amcl_pose_x,
                                   self.goal.target_pose.pose.position.y - self.amcl_pose_y])

        # print("Direction AMCL:", self.amcl_yaw)

        self.dot_dir = np.dot(dir_amcl, dir_robot_goal) / (np.linalg.norm(dir_amcl) * np.linalg.norm(dir_robot_goal))
        self.cross_dir = np.cross(dir_amcl, dir_robot_goal)

        # Compute the distance from the robot to the goal point
        self.dist_amcl_goal = math.sqrt(
            math.pow((self.goal.target_pose.pose.position.x - self.amcl_pose_x), 2) + math.pow(
                (self.goal.target_pose.pose.position.y - self.amcl_pose_y), 2))

        # print(self.amcl_pose_x, self.amcl_pose_y)
        # print(self.goal.target_pose.pose)
        # print(self.dist_amcl_goal)

        self.trigger = np.ones(len(self.pose_seq), dtype=bool)

        print("Distance:", self.dist_amcl_goal)
        print("Direction:", self.dot_dir)

        self.dot_dir_list.append(self.dot_dir)

        # print("GOAL: ", self.goal_cnt+1)

        # if self.dist_amcl_goal < near_area_dist:     # When the robot is near the goal point: "Fine" controller

        #     if self.dot_dir > 0.3:
        #         vel_msg.linear.x = 0
        #         vel_msg.linear.y = 0
        #         vel_msg.linear.z = 0

        #         vel_msg.angular.x = 0
        #         vel_msg.angular.y = 0

        #         if self.cross_dir >= 0:
        #             vel_msg.angular.z = - angular_velocity_in_place * micro_control_weight
        #             print("Turn right in place")
        #         else:
        #             vel_msg.angular.z = angular_velocity_in_place * micro_control_weight
        #             print("Turn left in place")

        #     elif (self.dot_dir < 0.3) & (self.dot_dir > 0.0):
        #         vel_msg.linear.x = velocity * micro_control_weight
        #         vel_msg.linear.y = 0
        #         vel_msg.linear.z = 0

        #         vel_msg.angular.x = 0
        #         vel_msg.angular.y = 0
        #         vel_msg.angular.z = 0

        #         print("Go forward slowly")

        #     else :
        #         vel_msg.linear.x = -velocity * micro_control_weight
        #         vel_msg.linear.y = 0
        #         vel_msg.linear.z = 0

        #         vel_msg.angular.x = 0
        #         vel_msg.angular.y = 0
        #         vel_msg.angular.z = 0

        #         print("Go backward slowly")

        #     #print(self.dist_amcl_goal)

        #     if (self.dist_amcl_goal < next_goal_dist_threshold) & self.trigger[self.goal_cnt]:     # When the robot is far from the goal point: "row tracer" controller

        #         self.trigger[self.goal_cnt] = False
        #         self.goal_cnt += 1

        #         #print(self.goal_cnt)
        #         #print(len(self.pose_seq))

        #         if self.goal_cnt >= len(self.pose_seq):
        #             rospy.loginfo("Final goal pose reached!")
        #             rospy.signal_shutdown("Final goal pose reached!")
        #             return
        #         else:
        #             print("!!!!!!Change the goal!!!!!!!!")
        #             self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        # print("DOT_DIR: ", self.dot_dir)

        # else:                                # When the robot is far from the goal point: "row tracer" controller
        # Set cmd_vel

    def move(self):

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # /cmd_vel publisher
        cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # rate = rospy.Rate(60)

        vel_msg = Twist()

        if (self.dot_dir_list[-1] * self.dot_dir_list[-2] < 0) and (self.dot_dir_list[-2] * self.dot_dir_list[-3] < 0):

            self.trigger[self.goal_cnt] = False
            self.goal_cnt += 1

            # print(self.goal_cnt)

            if self.goal_cnt >= len(self.pose_seq):
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return
            else:
                print("!!!!!!Change the goal!!!!!!!!")
                self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                self.rotation_trigger1 = True
                self.rotation_trigger2 = True

        else:

            if self.dot_dir < 0:  # When the robot needs a rotation in place to reach the goal position

                if self.dist_amcl_goal > near_area_dist:  # When the robot is near the goal point: "Fine" controller

                    beginTime = rospy.Time.now()
                    secondsIWantToSendMessagesFor = rospy.Duration(1);
                    endTime = secondsIWantToSendMessagesFor + beginTime

                    while ((rospy.Time.now() < endTime) & self.rotation_trigger1 == True):

                        vel_msg.linear.x = 0
                        vel_msg.linear.y = 0
                        vel_msg.linear.z = 0

                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0

                        if self.cross_dir >= 0:
                            vel_msg.angular.z = - (math.pi - np.arccos(self.dot_dir))
                        else:
                            vel_msg.angular.z = math.pi - np.arccos(self.dot_dir)

                        cmd_vel_publisher.publish(vel_msg)

                    if self.rotation_trigger1 == True:
                        rospy.sleep(1)

                    self.rotation_trigger1 = False

                    vel_msg.linear.x = - velocity
                    vel_msg.linear.y = 0
                    vel_msg.linear.z = 0

                    vel_msg.angular.x = 0
                    vel_msg.angular.y = 0
                    vel_msg.angular.z = - self.theta * theta_weight

                    print("Backward row tracing...:", vel_msg.angular.z)

                    print(self.dist_amcl_goal)
                    print(near_area_dist)

                else:
                    # while (self.dot_dir > -0.8):
                    # start = time.time()
                    # stop = time.time()

                    beginTime = rospy.Time.now()
                    secondsIWantToSendMessagesFor = rospy.Duration(1);
                    endTime = secondsIWantToSendMessagesFor + beginTime

                    while ((rospy.Time.now() < endTime) & self.rotation_trigger2 == True):

                        vel_msg.linear.x = 0
                        vel_msg.linear.y = 0
                        vel_msg.linear.z = 0

                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0

                        if self.cross_dir >= 0:
                            vel_msg.angular.z = - (math.pi - np.arccos(self.dot_dir))
                        else:
                            vel_msg.angular.z = math.pi - np.arccos(self.dot_dir)

                        cmd_vel_publisher.publish(vel_msg)

                        # print(rospy.Time.now())
                        # stop = time.time()
                        # print("START", beginTime)
                        # print("STOP: ", endTime)

                        # print("rotation_trigger2: ", self.rotation_trigger2)
                        # print("DOT_DIR: ", self.dot_dir)
                        print("Rotating in place (backward), angular velocity: ", vel_msg.angular.z, " ",
                              rospy.Time.now())

                    if self.rotation_trigger2 == True:
                        rospy.sleep(1)

                    self.rotation_trigger2 = False

                    # if self.cross_dir >= 0:

                    # else:
                    #     vel_msg.angular.z = - angular_velocity_in_place * micro_control_weight
                    # vel_msg.angular.z = - angular_velocity_in_place * micro_control_weight

                    # vel_msg.linear.x = - velocity * micro_control_weight
                    # vel_msg.linear.y = 0
                    # vel_msg.linear.z = 0

                    # vel_msg.angular.x = 0
                    # vel_msg.angular.y = 0
                    # vel_msg.angular.z = 0

                    # print("Go backward slowly")

                # vel_msg.angular.x = 0
                # vel_msg.angular.y = 0
                # vel_msg.angular.z = - angular_velocity_in_place

            else:

                if self.dist_amcl_goal > near_area_dist:  # When the robot is near the goal point: "Fine" controller

                    beginTime = rospy.Time.now()
                    secondsIWantToSendMessagesFor = rospy.Duration(1);
                    endTime = secondsIWantToSendMessagesFor + beginTime

                    # print(rospy.Time.now())
                    while ((rospy.Time.now() < endTime) & self.rotation_trigger1 == True):

                        vel_msg.linear.x = 0
                        vel_msg.linear.y = 0
                        vel_msg.linear.z = 0

                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0

                        if self.cross_dir >= 0:
                            vel_msg.angular.z = np.arccos(self.dot_dir)
                        else:
                            vel_msg.angular.z = -np.arccos(self.dot_dir)

                        cmd_vel_publisher.publish(vel_msg)

                        # stop = time.time()
                        # print("!!!!!!!!!!")
                        # print("START", beginTime)
                        # print("STOP: ", endTime)

                        # print(rospy.Time.now(), endTime)
                        # print("DOT_DIR: ", self.dot_dir)
                        print("Rotating in place (forward), angular velocity: ", vel_msg.angular.z, rospy.Time.now())

                    if self.rotation_trigger1 == True:
                        rospy.sleep(1)

                    self.rotation_trigger1 = False

                    vel_msg.linear.x = velocity
                    vel_msg.linear.y = 0
                    vel_msg.linear.z = 0

                    vel_msg.angular.x = 0
                    vel_msg.angular.y = 0
                    vel_msg.angular.z = self.theta * theta_weight

                    print("Forward row tracing...:", vel_msg.angular.z)

                    print(self.dist_amcl_goal)
                    print(near_area_dist)

                # else:
                #     #while (self.dot_dir < 0.8):
                #     beginTime = rospy.Time.now()
                #     secondsIWantToSendMessagesFor = rospy.Duration(1);
                #     endTime = secondsIWantToSendMessagesFor + beginTime

                #     # print(rospy.Time.now())
                #     while((rospy.Time.now() < endTime) & self.rotation_trigger2 == True):

                #         vel_msg.linear.x = 0
                #         vel_msg.linear.y = 0
                #         vel_msg.linear.z = 0

                #         vel_msg.angular.x = 0
                #         vel_msg.angular.y = 0

                #         if self.cross_dir >= 0:
                #             vel_msg.angular.z = np.arccos(self.dot_dir)
                #         else:
                #             vel_msg.angular.z = -np.arccos(self.dot_dir)

                #         cmd_vel_publisher.publish(vel_msg)

                #         # stop = time.time()
                #         #print("!!!!!!!!!!")
                #         #print("START", beginTime)
                #         #print("STOP: ", endTime)

                #         # print("DOT_DIR: ", self.dot_dir)
                #         print("Rotating in place, angular velocity: ", vel_msg.angular.z)

                #     if self.rotation_trigger2 == True:
                #         rospy.sleep(1)

                #     self.rotation_trigger2 = False
                #         #self.rotation_trigger2 = False

                #     # if self.cross_dir >= 0:
                #     #     vel_msg.angular.z = -angular_velocity_in_place * micro_control_weight
                #     # else:
                #     #     vel_msg.angular.z = angular_velocity_in_place * micro_control_weight

                #     # cmd_vel_publisher.publish(vel_msg)

                #     # print("DOT_DIR: ", self.dot_dir)
                #     # print("Rotating in place, angular velocity: ", vel_msg.angular.z)

                #     # vel_msg.linear.x = velocity * micro_control_weight
                #     # vel_msg.linear.y = 0
                #     # vel_msg.linear.z = 0

                #     # vel_msg.angular.x = 0
                #     # vel_msg.angular.y = 0
                #     # vel_msg.angular.z = 0

                #     # print("Go forward slowly")

            cmd_vel_publisher.publish(vel_msg)

            # Change to the next goal point
            # print(self.dist_amcl_goal, next_goal_dist_threshold, self.trigger[self.goal_cnt])

            # if (self.dist_amcl_goal < next_goal_dist_threshold) & self.trigger[self.goal_cnt]:     # When the robot is far from the goal point: "row tracer" controller
            if (self.dist_amcl_goal < near_area_dist) & self.trigger[
                self.goal_cnt]:  # When the robot is far from the goal point: "row tracer" controller

                self.trigger[self.goal_cnt] = False
                self.goal_cnt += 1

                # print(self.goal_cnt)

                if self.goal_cnt >= len(self.pose_seq):
                    rospy.loginfo("Final goal pose reached!")
                    rospy.signal_shutdown("Final goal pose reached!")
                    return
                else:
                    print("!!!!!!Change the goal!!!!!!!!")
                    self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                    self.rotation_trigger1 = True
                    self.rotation_trigger2 = True

                # else:

                #     vel_msg.linear.x = velocity
                #     vel_msg.linear.y = 0
                #     vel_msg.linear.z = 0

                #     vel_msg.angular.x = 0
                #     vel_msg.angular.y = 0
                #     vel_msg.angular.z = self.theta * theta_weight

                #     print("Forward row tracing...")

        # cmd_vel_publisher.publish(vel_msg)

    def movebase_client(self):
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = "odom"
        self.goal.target_pose.header.stamp = rospy.Time.now()

        self.rotation_trigger1 = True
        self.rotation_trigger2 = True
        # self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]


# Main function
if __name__ == "__main__":

    rospy.init_node('navigation_goal')

    # listener = tf.TransformListener()

    # rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            NaviIntegration()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
