#!/usr/bin/env python3

import os
import time
import struct
import numpy as np
import math
import tf

import rospy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Header
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
import actionlib
import yaml


class NaviIntegration:
    # Class Initialization
    def __init__(self):
        self.waypoints = []  # List to store the selected waypoints
        # self.navigation_started = False  # Flag to track if navigation has started

        self.MAX_HEIGHT = -0.1
        self.MIN_HEIGHT = -0.3
        self.NUM_POINTS_CLUSTER = 100

        self.VELOCITY = 0.3
        self.dx = self.VELOCITY
        self.THETA_WEIGHT = 0.15  # 0.05: Slightly rotate while driving # 0.5: Farily rotate
        self.NEAR_AREA_DIST = 0.1

        self.prev_vel_msg = Twist()
        self.prev_vel_msg.linear.x = self.VELOCITY
        self.prev_vel_msg.linear.y = 0
        self.prev_vel_msg.linear.z = 0

        self.prev_vel_msg.angular.x = 0
        self.prev_vel_msg.angular.y = 0
        self.prev_vel_msg.angular.z = 0

        self.listener = tf.TransformListener()

        # #List of goal quaternions:
        # self.quat_seq = list()

        # #List of goal poses:
        # self.pose_seq = list()
        # self.goal_cnt = 0
        self.quat_seq = []
        self.pose_seq = []
        self.goal_cnt = 0

        self.dot_dir_list = [0, 0, 0]
        self.cur_dir_robot = np.array([1, 0])

        # self.lidar_sub = rospy.Subscriber("/velodyne1_points", PointCloud2, self.lidarCallback)
        self.lidar_sub = rospy.Subscriber("/ns1/velodyne_points", PointCloud2, self.lidarCallback)
        self.waypoint_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.waypointCallback)

        # Create a MarkerArray message for visualization
        self.waypoint_marker_pub = rospy.Publisher('/waypoint_marker', Marker, queue_size=10)

        # /cmd_vel publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base_client.wait_for_server()

        while not rospy.is_shutdown():
            self.movebase_client()  # Set the goal point
            rospy.spin()  # spin() simply keeps python from exiting until this node is stopped

    def waypointCallback(self, data):
        # Extract the position and orientation from the received PoseStamped message
        position = data.pose.position
        orientation = data.pose.orientation

        waypoint = [position.x, position.y, position.z]
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]

        self.waypoints.append(waypoint)
        self.pose_seq.append(Pose(Point(*waypoint), Quaternion(*quaternion)))

        for i, pose in enumerate(self.pose_seq):
            position = pose.position
            print(f"Waypoint {i + 1}:")
            print(f"Position: ({position.x}, {position.y}, {position.z})")

        # Visualize waypoints
        self.waypointVisualization()

        # Save the waypoints to a YAML file
        self.saveWaypointsToYaml()

    def waypointVisualization(self):
        marker = Marker()
        marker_id = 1  # ID counter for markers

        for waypoint in self.waypoints:
            # Create a Marker message for each waypoint
            p = Point()
            marker.header.frame_id = "map"  # Set the frame ID of the marker
            marker.id = marker_id  # Assign a unique ID to the marker
            marker.type = marker.SPHERE_LIST  # Use a sphere marker
            marker.action = marker.ADD  # Add the marker

            # Set the position of the marker
            p.x = waypoint[0]
            p.y = waypoint[1]
            p.z = waypoint[2]

            print("VISUAL:", p.x, p.y, p.z)
            # Set the scale of the marker (adjust as needed)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            # Set the color of the marker (adjust as needed)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Append the marker to the MarkerArray
            marker.points.append(p)
            # marker_id += 1  # Increment the ID counter

        # Publish the MarkerArray for visualization
        self.waypoint_marker_pub.publish(marker)

    def saveWaypointsToYaml(self):
        data = {}
        data['p_seq'] = [point for waypoint in self.waypoints for point in [waypoint[0], waypoint[1], waypoint[2]]]

        # Specify the absolute path to save the waypoints.yaml file
        absolute_path = "/home/kimkt0408/catkin_ws/src/autonomous_navigation/param/"

        # Generate the file path by joining the absolute path and the file name
        file_path = os.path.join(absolute_path, 'waypoints.yaml')

        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    # AMCL pose callback function
    def PoseListener(self):
        try:
            (self.trans_pose, self.rot_pose) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            # (self.trans_pose, self.rot_pose) = self.listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
            self.pose_x, self.pose_y, _ = self.trans_pose
            _, _, self.pose_yaw = tf.transformations.euler_from_quaternion(self.rot_pose)

        except tf.Exception as e:
            rospy.logwarn("Could not get transform from /map to /base_link: %s" % e)
            # rospy.logwarn("Could not get transform from /odom to /base_link: %s" % e)

    # Goal info callback function
    def GoalCallback(self, goal):
        # Goal location (translation)
        self.goal_pose_x = goal.goal.target_pose.pose.position.x
        self.goal_pose_y = goal.goal.target_pose.pose.position.y

        # Goal location (orientation, Quarternion)
        goal_orientation_x = goal.goal.target_pose.pose.orientation.x
        goal_orientation_y = goal.goal.target_pose.pose.orientation.y
        goal_orientation_z = goal.goal.target_pose.pose.orientation.z
        goal_orientation_w = goal.goal.target_pose.pose.orientation.w

        # Change from Quarternion to Euler angle (x, y, z, w -> roll, pitch, yaw)
        goal_orientation_list = [goal_orientation_x, goal_orientation_y, goal_orientation_z, goal_orientation_w]
        _, _, self.goal_yaw = tf.transformations.euler_from_quaternion(goal_orientation_list)

    # Laser scan callback function
    def lidarCallback(self, lidarscan):
        (self.trans_pose, self.rot_pose) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        # (self.trans_pose,self.rot_pose) = self.listener.lookupTransform('/odom', '/base_link', rospy.Time(0))

        if len(self.pose_seq):
            self.PoseListener()

            # Compute the distance between the robot and the surroundings
            self.direction_function()

            assert isinstance(lidarscan, PointCloud2)
            cloud_points = list(point_cloud2.read_points(lidarscan, skip_nans=True, field_names=("x", "y", "z")))

            # (1) Filtering the usable pointclouds for autonomous navigation (different depends on the driving direction)
            if self.dot_dir > 0:  # Forward
                points = [list(cloud_point) for cloud_point in cloud_points if
                          cloud_point[0] <= 10.0 and cloud_point[2] < self.MAX_HEIGHT and cloud_point[
                              2] > self.MIN_HEIGHT]
            else:  # Backward
                points = [list(cloud_point) for cloud_point in cloud_points if
                          cloud_point[0] >= -10.0 and cloud_point[2] < self.MAX_HEIGHT and cloud_point[
                              2] > self.MIN_HEIGHT]

                # (2) Clustering the pointclouds from each row
            clustering = DBSCAN(eps=0.3, min_samples=3).fit_predict(points)

            num_clusters = max(clustering) + 1

            self.cluster_list = [[] for _ in range(num_clusters)]

            for i in range(len(points)):
                if clustering[i] != -1:
                    self.cluster_list[clustering[i]].append(points[i])

            # To check: Visualization
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1)]

            header = Header()
            header.frame_id = "velodyne1"
            pc2_navigation = point_cloud2.create_cloud(header, fields, points)

            pc2_navigation.header.stamp = rospy.Time.now()

            pc2_pub1 = rospy.Publisher('/pc2_navigation', PointCloud2, queue_size=10)
            pc2_pub1.publish(pc2_navigation)

            self.dist_computation()
            self.move()

    # Compute the distances from the robot to both sides of rows
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
            if len(cluster) > self.NUM_POINTS_CLUSTER:
                x = np.array([point[0] for point in cluster])
                y = np.array([point[1] for point in cluster])

                slope, intercept = np.polyfit(x, y, 1)
                # print("SLOPE:", slope)
                row_list.append([slope, intercept])

                # (4) Compute the distance from the lidar to each row
                if np.abs(slope) < 1.0:
                    # d = |ax1+by1+c|/sqrt(a^2+b^2)
                    dist = intercept / math.sqrt(pow(slope, 2) + 1)
                    # print("LINE: ", dist)
                else:
                    # Compute the distance from each point to the origin
                    distances = np.sqrt(x ** 2 + y ** 2) * y / np.abs(y)
                    dist = min(distances, key=abs)
                    # print("POINT: ", dist)

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

        line_publisher = rospy.Publisher('/row_lines', Marker, queue_size=10)
        line_publisher.publish(lines_marker)

        left_row_list = [-dist for dist in dist_list if dist < 0]
        right_row_list = [dist for dist in dist_list if dist > 0]

        # Check whether each row point cloud exists or not
        if (len(left_row_list) * len(right_row_list) == 0):
            self.bool_row = 0

        else:
            self.bool_row = 1

            min_left_dist = min(left_row_list)
            min_right_dist = min(right_row_list)

            print("left_min: ", min_left_dist)
            print("right_min: ", min_right_dist)

            # np.savetxt(a_file, [min_left_dist], fmt='%f')
            # np.savetxt(a_file, [min_right_dist], fmt='%f')

            # 3. Determine the direction of driving
            # Compute the average value of left/right minimum distance
            half_dist = 0.5 * (min_left_dist + min_right_dist)
            diff_dist = min_left_dist - half_dist

            # Compute the direction and angular velocity
            self.theta = math.atan(diff_dist / self.dx)

            self.rot_mat = np.array(
                [[math.cos(self.theta), -math.sin(self.theta)], [math.sin(self.theta), math.cos(self.theta)]])
            self.goal_dir_robot = np.matmul(self.rot_mat, self.cur_dir_robot)

    # The function to command a linear/angular velocity
    def direction_function(self):
        self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]

        # Compare between directions of AMCL pose and goal
        dir_amcl = np.array([math.cos(self.pose_yaw), math.sin(self.pose_yaw)])
        dir_robot_goal = np.array(
            [self.goal.target_pose.pose.position.x - self.pose_x, self.goal.target_pose.pose.position.y - self.pose_y])

        self.dot_dir = np.dot(dir_amcl, dir_robot_goal) / (np.linalg.norm(dir_amcl) * np.linalg.norm(dir_robot_goal))
        self.cross_dir = np.cross(dir_amcl, dir_robot_goal)

        # Compute the distance from the robot to the goal point
        self.dist_amcl_goal = math.sqrt(math.pow((self.goal.target_pose.pose.position.x - self.pose_x), 2) + math.pow(
            (self.goal.target_pose.pose.position.y - self.pose_y), 2))

        self.trigger = np.ones(len(self.pose_seq), dtype=bool)

        print("Distance:", self.dist_amcl_goal)
        print("Direction:", self.dot_dir)

        self.dot_dir_list.append(self.dot_dir)

    def move(self):
        # # /cmd_vel publisher
        # self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        vel_msg = Twist()

        # If the robot move forward and backward repeatedly near the goal location
        if (self.dot_dir_list[-1] * self.dot_dir_list[-2] < 0) and (self.dot_dir_list[-2] * self.dot_dir_list[-3] < 0):

            self.trigger[self.goal_cnt] = False
            self.goal_cnt += 1

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
            if self.bool_row == 1:
                if self.dot_dir < 0:  # When the robot needs a rotation in place to reach the goal position
                    if self.dist_amcl_goal > self.NEAR_AREA_DIST:  # When the robot is near the goal point: "Fine" controller
                        beginTime = rospy.Time.now()
                        endTime = rospy.Duration(1.0) + beginTime

                        while ((rospy.Time.now() < endTime) & self.rotation_trigger1 == True):
                            vel_msg.linear.x = 0
                            vel_msg.linear.y = 0
                            vel_msg.linear.z = 0

                            vel_msg.angular.x = 0
                            vel_msg.angular.y = 0

                            if self.cross_dir >= 0:
                                vel_msg.angular.z = -(math.pi - np.arccos(self.dot_dir))
                            else:
                                vel_msg.angular.z = math.pi - np.arccos(self.dot_dir)

                            self.cmd_vel_pub.publish(vel_msg)
                        print("Rotating in place (backward), angular velocity: ", vel_msg.angular.z, " ",
                              rospy.Time.now())

                        if self.rotation_trigger1 == True:
                            rospy.sleep(0.1)

                        self.rotation_trigger1 = False

                        vel_msg.linear.x = -self.VELOCITY
                        vel_msg.linear.y = 0
                        vel_msg.linear.z = 0

                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                        vel_msg.angular.z = -self.theta * self.THETA_WEIGHT

                        print("Backward row tracing...:", vel_msg.angular.z)

                    else:
                        beginTime = rospy.Time.now()
                        endTime = rospy.Duration(1.0) + beginTime

                        while ((rospy.Time.now() < endTime) & self.rotation_trigger2 == True):
                            vel_msg.linear.x = 0
                            vel_msg.linear.y = 0
                            vel_msg.linear.z = 0

                            vel_msg.angular.x = 0
                            vel_msg.angular.y = 0

                            if self.cross_dir >= 0:
                                vel_msg.angular.z = -(math.pi - np.arccos(self.dot_dir))
                            else:
                                vel_msg.angular.z = math.pi - np.arccos(self.dot_dir)

                            self.cmd_vel_pub.publish(vel_msg)
                        print("Slightly rotating in place (backward), angular velocity: ", vel_msg.angular.z, " ",
                              rospy.Time.now())

                        if self.rotation_trigger2 == True:
                            rospy.sleep(0.1)

                        self.rotation_trigger2 = False

                else:
                    if self.dist_amcl_goal > self.NEAR_AREA_DIST:  # When the robot is near the goal point: "Fine" controller

                        beginTime = rospy.Time.now()
                        endTime = rospy.Duration(1.0) + beginTime

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

                            self.cmd_vel_pub.publish(vel_msg)

                        print("Rotating in place (forward), angular velocity: ", vel_msg.angular.z, rospy.Time.now())

                        if self.rotation_trigger1 == True:
                            rospy.sleep(0.1)

                        self.rotation_trigger1 = False

                        vel_msg.linear.x = self.VELOCITY
                        vel_msg.linear.y = 0
                        vel_msg.linear.z = 0

                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                        vel_msg.angular.z = self.theta * self.THETA_WEIGHT

                        print("Forward row tracing...:", vel_msg.angular.z)

                    else:
                        beginTime = rospy.Time.now()
                        endTime = rospy.Duration(1.0) + beginTime

                        while ((rospy.Time.now() < endTime) & self.rotation_trigger2 == True):
                            vel_msg.linear.x = 0
                            vel_msg.linear.y = 0
                            vel_msg.linear.z = 0

                            vel_msg.angular.x = 0
                            vel_msg.angular.y = 0

                            if self.cross_dir >= 0:
                                vel_msg.angular.z = np.arccos(self.dot_dir)
                            else:
                                vel_msg.angular.z = -np.arccos(self.dot_dir)

                            self.cmd_vel_pub.publish(vel_msg)
                        print("Slightly rotating in place (forward), angular velocity: ", vel_msg.angular.z, " ",
                              rospy.Time.now())

                        if self.rotation_trigger2 == True:
                            rospy.sleep(0.1)

                        self.rotation_trigger2 = False

            else:
                print("One of the rows disappeared....")
                vel_msg = self.prev_vel_msg

            self.cmd_vel_pub.publish(vel_msg)
            self.prev_vel_msg = vel_msg

            # Change to the next goal point
            if (self.dist_amcl_goal < self.NEAR_AREA_DIST) & self.trigger[
                self.goal_cnt]:  # When the robot is far from the goal point: "row tracer" controller

                self.trigger[self.goal_cnt] = False
                self.goal_cnt += 1

                if self.goal_cnt >= len(self.pose_seq):
                    rospy.loginfo("Final goal pose reached!")
                    rospy.signal_shutdown("Final goal pose reached!")
                    return

                else:
                    print("!!!!!!Change the goal!!!!!!!!")
                    self.goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                    self.rotation_trigger1 = True
                    self.rotation_trigger2 = True
        print("==================================================")

    def movebase_client(self):
        self.goal = MoveBaseGoal()
        # self.goal.target_pose.header.frame_id = "odom"
        self.goal.target_pose.header.frame_id = "map"
        self.goal.target_pose.header.stamp = rospy.Time.now()

        self.rotation_trigger1 = True
        self.rotation_trigger2 = True

    def shutdown_callback(self):
        # Cancel the goal and stop the move_base action server
        self.move_base_client.cancel_all_goals()
        self.move_base_client.stop_tracking_goal()

        vel_msg = Twist()
        self.cmd_vel_pub.publish(vel_msg)

        print("==== MOVE_BASE_CLIENT CANCELLED ALL GOALS! ====")


# Main function
if __name__ == "__main__":
    rospy.init_node('navigation_goal')

    # Create an instance of NaviIntegration
    navi_integration = NaviIntegration()

    # Register the shutdown callback
    rospy.on_shutdown(navi_integration.shutdown_callback)

    # Spin until the node is terminated
    rospy.spin()