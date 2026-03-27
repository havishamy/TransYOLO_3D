#!/usr/bin/env python3
"""
real_time_triangulate_and_approach.py

功能：
1. 移动到预设位姿（initial_pose）
2. 在附近做若干小位移采集多帧检测（YOLO），并记录每帧 camera pose (base frame)
3. 多视角三角化得到物体 3D 位置（base frame）
4. 将物体作为碰撞体加入规划场景，规划并移动到物体“另一侧”，带避障

使用示例：
rosrun <your_pkg> real_time_triangulate_and_approach.py --model /path/to/best.pt --video /path/to/video.mp4
或 mode ros 从相机实时读取（需要 YOLO 能处理 cv frame）
"""
import sys
import time
import argparse
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import cv2
import socket
import json
import pyrealsense2 as rs

# YOLO (ultralytics)
#from ultralytics import YOLO
import os
# ROS / MoveIt
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler,euler_from_quaternion,quaternion_matrix
import moveit_commander
from moveit_commander import PlanningSceneInterface
from scipy import ndimage
from scipy.optimize import minimize  # 最小二乘法
from collections import deque

class Config:
  
    # 跟踪配置
    TRACK_MAX_AGE = 5  # 目标最大消失帧数（超过则删除）
    TRACK_IOU_THRESH = 0.3  # 跟踪匹配IoU阈值
    MAX_TRACKED_TARGETS = 10  # 最大跟踪目标数

    # 3D定位配置
    RAY_QUEUE_MAX_LEN = 30  # 每个目标的最大射线缓存数（动态滑动窗口）
    MIN_RAYS_FOR_FILTER = 4 # 射线异常值筛选阈值（单位：米）
    RAY_DISTANCE_THRESH = 0.15
    MIN_RAYS_FOR_3D = 3  # 计算3D位置所需的最小射线数
    CYLINDER_RADIUS = 0.05  # 障碍物建模（圆柱体半径，单位：米）

    # RealSense相机配置
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30     

    # 相机内参（需通过标定获取，示例值）
    CAMERA_INTRINSIC = np.array([[604.0, 0, 334.7],
              [0, 603.7, 250.7],
              [0,   0,   1]])


# --------------- math helpers (from your code) ---------------
def pose_stamped_to_xyzrpy(pose_stamped):
    """将 PoseStamped 对象转为 xyzrpy 的 numpy 数组"""
    # 提取位置（x,y,z）
    x = pose_stamped.pose.position.x
    y = pose_stamped.pose.position.y
    z = pose_stamped.pose.position.z
    
    # 提取姿态（四元数 x,y,z,w）→ 转为欧拉角（roll,pitch,yaw，弧度）
    quat = [
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(quat)
    
    # 组合为 numpy 数组
    return np.array([x, y, z, roll, pitch, yaw])

def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    x_cam = np.linalg.inv(K).dot(uv1)
    d = x_cam / np.linalg.norm(x_cam)
    return d

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)

def triangulate_rays(cam_centers: List[np.ndarray], directions: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    A_blocks = []
    b_blocks = []
    for C, d in zip(cam_centers, directions):
        d = d / np.linalg.norm(d)
        S = skew(d)
        A_blocks.append(S)
        b_blocks.append(S.dot(C))
    A = np.vstack(A_blocks)
    b = np.hstack(b_blocks)
    X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # compute mean perpendicular distance
    dists = []
    for C, d in zip(cam_centers, directions):
        v = X - C
        proj = np.dot(v, d) * d
        perp = v - proj
        dists.append(np.linalg.norm(perp))
    return X, float(np.mean(dists))

def robust_triangulate(cam_centers, directions, min_views=2, max_iter=6, thresh=0.05):
    indices = list(range(len(cam_centers)))
    for it in range(max_iter):
        if len(indices) < min_views:
            break
        C_sel = [cam_centers[i] for i in indices]
        d_sel = [directions[i] for i in indices]
        X, mean_err = triangulate_rays(C_sel, d_sel)
        perp = []
        for Ci, di in zip(C_sel, d_sel):
            v = X - Ci
            proj = np.dot(v, di) * di
            perp.append(np.linalg.norm(v - proj))
        perp = np.array(perp)
        keep_mask = perp <= thresh
        if keep_mask.all():
            return X, float(perp.mean()), indices
        new_indices = [idx for k, idx in zip(keep_mask, indices) if k]
        if len(new_indices) == len(indices):
            return X, float(perp.mean()), indices
        indices = new_indices
    if len(indices) >= min_views:
        C_sel = [cam_centers[i] for i in indices]
        d_sel = [directions[i] for i in indices]
        X, mean_err = triangulate_rays(C_sel, d_sel)
        return X, mean_err, indices
    return None, None, []

# ----------------- Node -----------------
class RemoteYOLOClient:
    def __init__(self, server_ip, server_port,base_frame,camera_frame,K):
        # TCP连接配置
        self.server_ip = server_ip  # 服务器IP（如192.168.1.100）
        self.server_port = server_port  # 服务器端口（如8888）
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_server()
        self.config=Config()
        # TF buffer
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.k=K


    def get_camera_pose_base(self):
        """返回 4x4 转换矩阵 T_base_cam"""
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.5))
            trans = t.transform.translation
            rot = t.transform.rotation
            T = np.eye(4, dtype=np.float64)
            T[:3,:3] = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])[:3,:3]
            T[0,3] = trans.x; T[1,3] = trans.y; T[2,3] = trans.z
            return T
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[TF] lookup failed: {e}")
            return None
    
    def connect_server(self):
        """连接服务器"""
        try:
            self.socket.connect((self.server_ip, self.server_port))
            print(f"成功连接服务器：{self.server_ip}:{self.server_port}")
        except Exception as e:
            print(f"连接服务器失败：{str(e)}")
            exit(1)

    def _compute_ray(self, centroid_2d, camera_pose: np.ndarray):
        """
        计算3D射线（相机中心到目标质心的射线）
        :param centroid_2d: 图像坐标系下的质心 (x, y)
        :param camera_pose: 相机位姿（4x4齐次矩阵，世界坐标系→相机坐标系）
        :return: 射线起点（世界坐标系）、射线方向向量（单位向量）
        """
        u, v = centroid_2d
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        x_cam = np.linalg.inv(self.k).dot(uv1)
        d = x_cam / np.linalg.norm(x_cam)
        
        camera_pos_world = camera_pose[:3, 3]  # 相机在世界坐标系的位置（射线起点）
        rotation_matrix = camera_pose[:3, :3]  # 相机旋转矩阵
        ray_dir_world = rotation_matrix @ d  # 射线方向转换到世界坐标系
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)  # 单位向量

        return (camera_pos_world, ray_dir_world)
    
    def _compute_mask_centroid(self, mask: np.ndarray):
        """计算分割掩码的2D质心（图像坐标系）"""
        # 找到掩码的所有非零像素
        
        y,x= ndimage.center_of_mass(mask)
        return  (x,y)
    
    def _compute_box_centroid(self, box):
        """计算检测框的2D质心（图像坐标系）"""
        # 找到掩码的所有非零像素
        
        if len(box) != 4:
            raise ValueError(f"bounding box 必须包含 4 个坐标值，当前输入了 {len(box)} 个。")

        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return (cx, cy)
    
    def send_image_get_detections(self, image):
        """发送图像到服务器，接收识别结果"""
        # 图像编码为JPEG
        ret, encode_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            rospy.logwarn("图像编码失败")
            return []
        
        img_bytes = encode_img.tobytes()
        img_len = len(img_bytes)
        
        # 发送图像长度+图像数据
        self.socket.sendall(img_len.to_bytes(4, byteorder='big'))
        self.socket.sendall(img_bytes)
        
        # 接收结果
        try:
            result_len = int.from_bytes(self.socket.recv(4), byteorder='big')
            result_bytes = b""
            while len(result_bytes) < result_len:
                chunk = self.socket.recv(min(4096, result_len - len(result_bytes)))
                if not chunk:
                    return []
                result_bytes += chunk
            result = json.loads(result_bytes.decode('utf-8'))
            
            T_base_cam = self.get_camera_pose_base()
            if result["success"] and len(result["detections"]) > 0:
                # 返回所有检测目标的（u, v, 置信度），按置信度排序
                current_detections=[]
                for i in result["detections"]:

                    current_masks = np.array(i["mask"])
                    current_cls =  np.array(i["cls_id"])
                    current_boxes =  np.array(i["box"])
                
                
                    #centroid_2d = self._compute_mask_centroid(current_masks)
                    centroid_2d = self._compute_box_centroid(current_boxes)
                    if centroid_2d[0] < 0:
                        continue
                    current_detections.append({
                        "mask": current_masks,
                        "cls": int(current_cls),
                        "box": current_boxes,
                        "centroid_2d": centroid_2d,
                        "ray": self._compute_ray(centroid_2d, T_base_cam),
                        "tf":T_base_cam
                        })
        
                return current_detections
            else:
                rospy.logwarn(f"无有效检测结果：{result.get('error', 'No detections')}")
                return []
        except Exception as e:
            rospy.logerr(f"接收识别结果失败：{str(e)}")
            return []
    
class RealTimeTriangulateAndApproach:
    def __init__(self, args):
        rospy.init_node('rt_tri_approach', anonymous=True)
        self.rate = rospy.Rate(10)
        #self.model = YOLO(args.model)
        # Camera intrinsics
        # 1. 初始化RealSense相机
        self.realsense_pipeline = rs.pipeline()
        self.realsense_config = rs.config()
        # 配置RGB流（与相机内参K匹配，640x480）
        self.realsense_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.realsense_pipeline.start(self.realsense_config)
        rospy.loginfo("RealSense相机启动成功")

        # 2. 初始化远程YOLO客户端
        self.K = np.array([[args.Kfx, 0, args.Kcx],
                           [0, args.Kfy, args.Kcy],
                           [0, 0, 1]], dtype=np.float64)
        
        
        # Video source (if offline). If mode == 'ros' we'll use cv frames from camera topic (not implemented here)
        self.cap = cv2.VideoCapture(args.video) if args.mode == 'offline' else None
        # MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = PlanningSceneInterface(synchronous=True)
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.group.set_end_effector_link("tool0")

        self.group.set_planner_id("RRTConnectkConfigDefault")  # 该规划器天然支持避障
        self.group.set_planning_time(10.0)  

        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_max_acceleration_scaling_factor(0.1)
        # TF buffer to get camera pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = args.base_frame
        self.camera_frame = args.camera_frame

        self.yolo_client = RemoteYOLOClient(args.yolo_server_ip, args.yolo_server_port, self.base_frame, self.camera_frame,self.K)
        self.tracked_targets = defaultdict(dict)
        self.config=Config()
        self.next_track_id=0

        # Waypoint initial
        self.initial_pose = args.initial_pose  # tuple (x,y,z,roll,pitch,yaw)
        # triangulation params
        self.num_views = args.num_views
        self.min_views = max(2, args.min_views)
        self.tri_thresh = args.tri_thresh
        # approach params
        self.object_safety_radius = args.object_radius
        self.approach_distance = args.approach_distance
        # internal
        rospy.sleep(0.5)

    def __del__(self):
        """析构函数：释放资源"""
        if hasattr(self, 'realsense_pipeline'):
            self.realsense_pipeline.stop()
        if hasattr(self, 'yolo_client') and hasattr(self.yolo_client, 'socket'):
            self.yolo_client.socket.close()
        rospy.loginfo("资源已释放")

    def move_to_pose(self, xyzrpy, wait=True):
        x,y,z,roll,pitch,yaw = xyzrpy
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        q = quaternion_from_euler(roll, pitch, yaw)
        target.pose.orientation.x = q[0]; target.pose.orientation.y = q[1]
        target.pose.orientation.z = q[2]; target.pose.orientation.w = q[3]
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(target, self.group.get_end_effector_link())
        plan_ok = self.group.go(wait=wait)
        self.group.stop()
        self.group.clear_pose_targets()
        return plan_ok

    def get_camera_pose_base(self):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.5))
            # convert to 4x4
            trans = t.transform.translation
            rot = t.transform.rotation
            T = np.eye(4, dtype=np.float64)
            # quaternion to rot matrix
            q = [rot.x, rot.y, rot.z, rot.w]
            # use scipy-like conversion
            import math
            # build rot matrix via numpy
            # we can use tf tf.transformations if available, but keep lightweight:
            from tf.transformations import quaternion_matrix
            T[:3, :3] = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])[:3, :3]
            T[0,3] = trans.x; T[1,3] = trans.y; T[2,3] = trans.z
            return T
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return None

    def capture_frame_and_detections(self):
        """修改：从RealSense采集图像，通过远程YOLO识别+显示图像+检测框+中心点"""
        try:
            # 1. RealSense采集一帧图像
            frames = self.realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                rospy.logwarn("未获取到图像帧")
                return [], None
            frame = np.asanyarray(color_frame.get_data())

            # 2. 远程YOLO识别（dets格式：[(u1, v1, conf1), (u2, v2, conf2), ...]，u/v是中心点像素坐标）
            dets = self.yolo_client.send_image_get_detections(frame)

            # 3. 绘制检测框和中心点（若有识别结果）
            if len(dets) > 0:
                # 遍历所有检测目标（按置信度降序排列）
                for det in dets:
                    # 绘制中心点（红色，半径3，填充）
                    (u,v),conf=det["centroid_2d"],det["cls"]
                    cv2.circle(frame, (int(u), int(v)), 3, (0, 0, 255), -1)

                    # 绘制置信度文本（白色背景+黑色文字）
                    text = f"Conf: {conf}"
                    cv2.putText(
                        frame, text, (int(u-50), int(v-50)),  # 文本位置（检测框上方）
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,  # 字体+字号
                        (0, 0, 0), 1  # 文字颜色（黑色）+ 线宽
                    )

            # 4. 显示图像窗口（窗口名：RealSense + YOLO Detections）
            cv2.imshow("RealSense + YOLO Detections", frame)
            # 按 'q' 键可关闭窗口（需保留，否则窗口会卡死）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rospy.loginfo("已关闭图像显示窗口")
            if len(dets) > 0:
                # 生成唯一文件名：时间戳_检测目标数.jpg
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式：20240520_143025_123
                save_path = os.path.join("/home/dsj/ur5e/src/universal_robot/ur5e_moveit_config/scripts", f"detection_{timestamp}_{len(dets)}.jpg")
                # 保存图像
                cv2.imwrite(save_path, frame)
                rospy.loginfo(f"已保存检测结果图：{save_path}")

            return dets, frame
        except Exception as e:
            rospy.logerr(f"采集/识别/可视化失败：{str(e)}")
            # 异常时关闭窗口，避免资源泄漏
            cv2.destroyAllWindows()
            return [], None
        
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray):
        """计算两个掩码的IoU（用于跟踪匹配）"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def _compute_3d_position(self, rays):
        """
        最小二乘法计算多条射线的最佳交点（距所有射线距离最短的点）
        :param rays: 射线列表，每条射线格式：(起点, 方向向量)
        :return: 3D目标位置（世界坐标系）
        """
        def distance_to_rays(point, rays):
            """计算点到所有射线的距离之和"""
            total_dist = 0.0
            for (ray_origin, ray_dir) in rays:
                # 点到射线的距离公式：||(P - O) × dir|| / ||dir||
                vec_op = point - ray_origin
                cross = np.cross(vec_op, ray_dir)
                dist = np.linalg.norm(cross) / np.linalg.norm(ray_dir)
                total_dist += dist ** 2  # 平方和（便于优化）
            return total_dist

        # 初始猜测：所有射线起点的均值
        ray_origins = np.array([ray[0] for ray in rays])
        initial_guess = np.mean(ray_origins, axis=0)

        # 最小化距离之和
        result = minimize(
            fun=distance_to_rays,
            x0=initial_guess,
            args=(rays,),
            method="L-BFGS-B"
        )

        if result.success:
            return result.x  # 最优3D位置
        else:
            print(f"3D位置计算失败：{result.message}")
            return initial_guess  # 失败时返回初始猜测
 

    def update_tracks(self, current_detections):

        # 步骤2：关联当前检测与历史跟踪目标（IoU匹配）
        tracked_ids = list(self.tracked_targets.keys())
        matched_ids = set()

        for det in current_detections:
            best_iou = 0.0
            best_id = None
            # 与历史目标匹配
            for track_id in tracked_ids:
                if track_id in matched_ids:
                    continue
                hist_mask = self.tracked_targets[track_id]["last_mask"]
                iou = self._compute_mask_iou(det["mask"], hist_mask)
                if iou > self.config.TRACK_IOU_THRESH and iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                # 匹配成功：更新历史目标
                self.tracked_targets[best_id]["last_mask"] = det["mask"]
                self.tracked_targets[best_id]["last_centroid_2d"] = det["centroid_2d"]
                self.tracked_targets[best_id]["last_box"] = det["box"]
                self.tracked_targets[best_id]["cls"] = det["cls"]
                self.tracked_targets[best_id]["age"] = 0  # 重置消失帧数
                # 添加新射线到缓存（滑动窗口）
                rays = self.tracked_targets[best_id]["rays"]
                rays.append(det["ray"])
                if len(rays) > self.config.RAY_QUEUE_MAX_LEN:
                    rays.popleft()  # 丢弃最早的射线
                # 筛选异常射线
                #self.tracked_targets[best_id]["rays"] = self._filter_outlier_rays_by_distance(rays)
                # 计算3D位置（如果射线足够）
                if len(self.tracked_targets[best_id]["rays"]) >= self.config.MIN_RAYS_FOR_3D:
                    self.tracked_targets[best_id]["3d_pos"] = self._compute_3d_position(rays)
                matched_ids.add(best_id)
            else:
                # 匹配失败：新增跟踪目标
                if self.next_track_id < self.config.MAX_TRACKED_TARGETS:
                    self.tracked_targets[self.next_track_id] = {
                        "track_id": self.next_track_id,
                        "cls": det["cls"],
                        "last_mask": det["mask"],
                        "last_centroid_2d": det["centroid_2d"],
                        "last_box": det["box"],
                        "age": 0,
                        "rays": deque([det["ray"]], maxlen=self.config.RAY_QUEUE_MAX_LEN),
                        "3d_pos": None,  # 初始无3D位置
                        "tf": det["tf"]
                    }
                    self.next_track_id += 1

        # 步骤3：更新未匹配目标的消失帧数，超过阈值则删除
        for track_id in tracked_ids:
            if track_id not in matched_ids:
                self.tracked_targets[track_id]["age"] += 1
                if self.tracked_targets[track_id]["age"] > self.config.TRACK_MAX_AGE:
                    del self.tracked_targets[track_id]
                    print(f"删除跟踪目标：ID={track_id}（消失帧数超限）")


    def collect_views_for_object(self, target_class_id=None):
        """
        Move the arm slightly N times and capture detection + camera poses.
        Returns list of (C_base (3,), d_base (3,)) pairs for the selected tracked object (we pick highest confidence detections).
        """
        cam_centers = []
        directions = []

        # Move pattern: small lateral steps around current pose (robot stays safe)
        cur_pose = self.group.get_current_pose(self.group.get_end_effector_link()).pose
        cur_xyz = (cur_pose.position.x, cur_pose.position.y, cur_pose.position.z)
        # generate small offsets in Y axis (left-right) and maybe small Z jitter
        offsets = []
        span = 0.02  # total span in meters
        n=3
        #n = max(2, self.num_views)
        '''
        for i in range(n):
            t = (i/(n-1)) if n>1 else 0.5
            y_offset = (t-0.5) * span
            z_offset = 0.0
            offsets.append((0.0, y_offset, z_offset))
        '''
        for i in range(n):
            for j in range(n):
                for k in range(n):

                    tx = (i/(n-1)) if n > 1 else 0.5
                    ty = (j/(n-1)) if n > 1 else 0.5
                    tz = (k/(n-1)) if n > 1 else 0.5

                    x_offset = (tx - 0.5) * span
                    y_offset = (ty - 0.5) * span
                    z_offset = (tz - 0.5) * span

                    offsets.append((x_offset, y_offset, z_offset))

        for dx,dy,dz in offsets:
            # compute small cartesian step (relative)
            start = self.group.get_current_pose(self.group.get_end_effector_link()).pose
            target = PoseStamped()
            target.header.frame_id = self.base_frame
            target.header.stamp = rospy.Time.now()
            target.pose.position.x = start.position.x + dx
            target.pose.position.y = start.position.y + dy
            target.pose.position.z = start.position.z + dz
            # keep orientation
            target.pose.orientation = start.orientation
            self.group.set_start_state_to_current_state()
            (plan, fraction) = self.group.compute_cartesian_path([target.pose], eef_step=0.01)
            if fraction > 0.0:
                self.group.execute(plan, wait=True)
            rospy.sleep(0.2)  # give time for TF and camera to stabilize

            T_base_cam = self.get_camera_pose_base()
            if T_base_cam is None:
                rospy.logwarn("no camera pose, skipping this view")
                continue
            # capture frame and detections
            dets_frame = self.capture_frame_and_detections()
            if len(dets_frame) == 0:
                rospy.logwarn("no detections in this frame")
                continue
            dets, frame = dets_frame
            # pick best detection (highest confidence)
            if len(dets) == 0:
                rospy.logwarn("no boxes parsed")
                continue
            self.update_tracks(dets)
           

    def add_collision_box_at(self, center, size=0.08, timeout=2.0):
        """
        Add a small box as collision object at `center` (3x) in base frame.
        """
        box_pose = PoseStamped()
        box_pose.header.frame_id = self.base_frame
        box_pose.header.stamp = rospy.Time.now()
        box_pose.pose.position.x = float(center[0])
        box_pose.pose.position.y = float(center[1])
        box_pose.pose.position.z = float(center[2])
        # neutral orientation
        box_pose.pose.orientation.x = 0.0
        box_pose.pose.orientation.y = 0.0
        box_pose.pose.orientation.z = 0.0
        box_pose.pose.orientation.w = 1.0
        name = f"obj_{int(time.time())}"
        self.scene.add_box(name, box_pose, size)
        # wait for planning scene update
        start = time.time()
        while time.time() - start < timeout:
            if name in self.scene.get_known_object_names():
                rospy.loginfo(f"collision object {name} added")
                return name
            rospy.sleep(0.1)
        rospy.logwarn("collision object add timeout")
        return name
   
    def _quantize_key(self, X, q=0.05):
        """将坐标量化为网格 key，避免同一物体反复创建多个 name"""
        return (round(X[0]/q)*q, round(X[1]/q)*q, round(X[2]/q)*q)
    
    def move(self):
        for ob in self.tracked_targets:
            pose=[-0.449, -0.248, 0.236, -3.112, 0.087, 1.605]
            #ok = self.move_to_pose(self.initial_pose, wait=True)
            ok = self.move_to_pose(pose, wait=True)
            if not ok:
                rospy.logerr("Failed to move to initial pose")
                return
            direction=self.tracked_targets[ob]["3d_pos"]
            cur_pose = self.group.get_current_pose(self.group.get_end_effector_link()).pose
            target_pose_start = PoseStamped()
            target_pose_start.header.frame_id = self.base_frame
            target_pose_start.header.stamp = rospy.Time.now()
            target_pose_start.pose.position.x = direction[0]+0.15
            target_pose_start.pose.position.y = direction[1]
            target_pose_start.pose.position.z = 0.275
            target_pose_start.pose.orientation = cur_pose.orientation
            self.move_to_pose(pose_stamped_to_xyzrpy(target_pose_start), wait=True)
            target_pose_end = PoseStamped()
            target_pose_end.header.frame_id = self.base_frame
            target_pose_end.header.stamp = rospy.Time.now()
            target_pose_end.pose.position.x = direction[0]-0.15
            target_pose_end.pose.position.y = direction[1]
            target_pose_end.pose.position.z = 0.275
            target_pose_end.pose.orientation = cur_pose.orientation
            rospy.loginfo(f"Planning approach to {target_pose_end}")
            self.group.set_start_state_to_current_state()
            self.group.set_pose_target(target_pose_end, self.group.get_end_effector_link())

            best_plan = None
            best_length = float('inf')
            start_time=time.time()
            for i in range(15):
                print(f"Planning attempt {i+1}")
                plan = self.group.plan()
                
                if plan[0]:
                    # 计算路径长度
                    path_length = 0
                    waypoints = plan[1].joint_trajectory.points
                    for j in range(1, len(waypoints)):
                        # 简单计算关节空间距离
                        diff = sum((waypoints[j].positions[k] - waypoints[j-1].positions[k])**2 
                                for k in range(len(waypoints[j].positions)))
                        path_length += diff
                    
                    print(f"  Path length: {path_length}")
                    
                    if path_length < best_length:
                        best_length = path_length
                        best_plan = plan
            end_time=time.time()
            duration=end_time-start_time
            print(duration)
            if best_plan[0]:
                print(f"✓ Best path length: {best_length}")
                self.group.execute(best_plan[1], wait=True)
            else:
                print("✗ All planning attempts failed")
                wayposes = [self.group.get_current_pose(self.group.get_end_effector_link()).pose, target_pose_end.pose]
                (plan_cart, frac) = self.group.compute_cartesian_path(wayposes, eef_step=0.01, jump_threshold=0.0)
                if frac > 0.1:
                    self.group.execute(plan_cart, wait=True)
                    rospy.loginfo("Cartesian fallback executed")
                else:
                    rospy.logerr("Cartesian fallback failed, aborting")

    def run(self):
        rospy.loginfo("Start: moving to initial pose")
        pose=[-0.449, -0.248, 0.236, -3.112, 0.087, 1.605]
        ok = self.move_to_pose(pose, wait=True)
        #ok = self.move_to_pose(self.initial_pose, wait=True)
        if not ok:
            rospy.logerr("Failed to move to initial pose")
            return
        
        rospy.loginfo("Collecting multi-view observations for triangulation")
        self.collect_views_for_object()
        obj_name=[]
        for track_id, target in self.tracked_targets.items():
            if target["3d_pos"] is not None:
                rospy.loginfo(f"Triangulated object position (base): {target['3d_pos']}")
        
        # add collision object at X
        #X=[0.390, 0.137, 0.225]#target 0.545, 0.137, 0.225
                obj_name.append(self.add_collision_box_at(target["3d_pos"], size=[0.1,0.1,0.2]))
        '''
        # compute approach target: go to opposite side of object relative to base
        base_pos = np.array([0.0, 0.0, 0.0])  # base frame origin
        dir_from_base = X - base_pos
        if np.linalg.norm(dir_from_base) < 1e-6:
            rospy.logerr("Object at base origin? abort")
            return
        dir_unit = dir_from_base / np.linalg.norm(dir_from_base)
        # opposite side vector (away from base, i.e., move to -dir_unit beyond object)
        target_pos = X + (dir_unit) * (self.object_safety_radius + self.approach_distance)

        # choose orientation: keep current end-effector orientation
        cur_pose = self.group.get_current_pose(self.group.get_end_effector_link()).pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = target_pos[0]
        target_pose.pose.position.y = target_pos[1]
        target_pose.pose.position.z = target_pos[2]
        target_pose.pose.orientation = cur_pose.orientation
        '''
        self.move()
        
        # cleanup
        rospy.loginfo("Finished. Removing collision object.")
        try:
            for obj in obj_name:
                self.scene.remove_world_object(obj)
        except Exception:
            pass

# ----------------- CLI and run -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo_server_ip', default="211.86.155.209")
    p.add_argument('--yolo_server_port', type=int, default=8866)
    
    p.add_argument('--model', default='/home/dsj/ur_ws/src/ur5e_robot/manipulator/scripts/ultralytics/runs/train/yolov11_glassware/weights/best.pt')
    p.add_argument('--video', default="/home/dsj/ur_ws/1.mp4")
    p.add_argument('--mode', choices=['offline','ros'], default='offline')
    p.add_argument('--base_frame', default='base_link')
    p.add_argument('--camera_frame', default='camera_color_frame')
    p.add_argument('--Kfx', type=float, default=604.0)
    p.add_argument('--Kfy', type=float, default=603.7)
    p.add_argument('--Kcx', type=float, default=334.7)
    p.add_argument('--Kcy', type=float, default=250.7)
    p.add_argument('--initial_pose', nargs=6, type=float,
                   #default=[-0.298, -0.179, 0.45, -3.051, -0.018, 1.571])
                   default=[-0.369, -0.220, 0.440,-2.680, 0.120, 1.626])
                   #default=[-0.449, -0.248, 0.236, -3.112, 0.087, 1.605])
    p.add_argument('--num_views', type=int, default=10)
    p.add_argument('--min_views', type=int, default=3)
    p.add_argument('--tri_thresh', type=float, default=0.05)
    p.add_argument('--object_radius', type=float, default=0.05)
    p.add_argument('--approach_distance', type=float, default=0.12)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        node = RealTimeTriangulateAndApproach(args)
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被中断")
    except Exception as e:
        rospy.logerr(f"发生错误：{str(e)}")
'''
  translation: 
  x: -0.03521843246238411
  y: -0.0559378107349008
  z: -0.08072852509858351
rotation: 
  x: 0.004491958605168551
  y: 0.0008506137377920497
  z: -0.0011040972718994293
  w: 0.9999889398055226
  
  
  translation: 
  x: -0.03481036372944785
  y: -0.060405746503347735
  z: -0.0777202759206387
rotation: 
  x: 0.0022810797269975118
  y: 0.003052810172862963
  z: 0.00844768804609737
  w: 0.9999570558739026
'''
'''
tool0到camera_link
translation: 
  x: -0.03821664874165852
  y: -0.06633752107850087
  z: 0.09106752675312837
rotation: 
  x: 0.0037121794818018405
  y: 0.004207769420989438
  z: 0.006958629437176863
  w: 0.99996004513998


  相机位置改变后的标定结果
  translation: 
  x: 0.002868036794780919
  y: -0.09650841304044622
  z: 0.00484814246966523
rotation: 
  x: -0.07770584186414166
  y: -0.0004751215786401586
  z: 0.01594038162830711
  w: 0.9968487752077616
'''