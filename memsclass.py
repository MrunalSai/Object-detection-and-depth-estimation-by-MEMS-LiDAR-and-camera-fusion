# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import os, sys, glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
#try:
  #  sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
   #     sys.version_info.major,
   #     sys.version_info.minor,
   #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#except IndexError:
  #  pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
from carla import ColorConverter as cc

import numpy as np
from time import sleep
import copy
from scipy.spatial.transform import Rotation as R
from queue import Queue
from queue import Empty

def r_0(t, T):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Simple Ramp Function, 
    with only uo-ramping phase.
    
    Input:
        t - time
        T - frame duration
    Returns:
        ramp value at time t
    """
    return t / T
    
def h_mirror(t, t_h_max = 80, f = 150):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Horizontal Mirrow
    Input:
        t - time
        t_h_max - horizontal field of view
        f - eigenfrequency
    
    Returns:
        Angle of the horizontal mirrow.
    """
    return t_h_max / 2 * np.cos(2 * np.pi * f * t)
    
def v_mirror(t, t_v_max = 30, f = 150, T = None, ramp_function = r_0):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Vertical Mirrow
    Input:
        t - time
        t_v_max - vertical field of view
        f - eigenfrequency
        T - frame duration
        ramp_function - Ramp Function
    
    Returns:
        Angle of the vertical mirrow.
    """
    return ramp_function(t, T = T) * t_v_max / 2 * np.sin(2 * np.pi * f * t)

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px],
                     [0, fy, py],
                     [0, 0, 1.]])

class MEMS_Sensor(object):
    def __init__(self, parent_actor, carla_transform, image_width = 800, v_fov = 20, h_fov_total = 90, h_fov_pc = 70, n_scan_lines = 100,
                 n_points_pattern = 100000, max_range = 100, out_root = "out/MEMS", tick = 1, add_noise = False):
        """
        Class for MEMS Sensor
        """
        print("Spawn MEMS Sensor")
        
        self.out_root = out_root
        self.sensor = None
        self.add_noise = add_noise
        self.max_range = max_range
        self.carla_transform = carla_transform
        self.parent = parent_actor
        self.lidar_queue = Queue()
        
        world = self.parent.get_world()
        
        # Calculates the image height from the field of view and the image width
        image_height = np.tan(v_fov * np.pi / 180 / 2) / np.tan(h_fov_total * np.pi / 180 / 2) * image_width
        #image_height = 1080
        # Intrinsic Matrix of the Camera
        self.K = intrinsic_from_fov(image_height, image_width, h_fov_total)
        self.K_inv = np.linalg.inv(self.K)
        
        # Scan Pattern of the MEMS Sensor
        self.pixel_coords_scan_pattern = MEMS_Sensor.scan_pattern(n_scan_lines = n_scan_lines, 
                                            n_points_pattern = n_points_pattern,
                                            width = image_width, height = image_height,
                                            h_fov_total = h_fov_total, h_fov_pc = h_fov_pc, v_fov = v_fov)
        
        # Spawn of a depth map sensor
        sensor = world.get_blueprint_library().find('sensor.camera.depth')
        sensor.set_attribute('image_size_x',str(image_width))
        sensor.set_attribute('image_size_y',str(image_height))
        sensor.set_attribute('fov',str(h_fov_total))
        self.sensor = world.spawn_actor(sensor, self.carla_transform, attach_to = parent_actor)
        
        # Starts the recording
        self.sensor.listen(lambda dm: self.queue(dm))

    def queue(self,raw_depth_map):
        depth_map = np.float64(MEMS_Sensor.get_depth_map(raw_depth_map))
        depth_map = MEMS_Sensor.in_meters(depth_map)
        depth_map[depth_map > self.max_range] = np.nan
        self.frame = raw_depth_map.frame
        self.point_cloud = self.depth_map_to_point_cloud(depth_map)

        def sensor_callback(point_cloud, queue):
            """
            This simple callback just stores the data on a thread safe Python Queue
            to be retrieved from the "main thread".
            """
            queue.put(point_cloud)

        sensor_callback(self.point_cloud, self.lidar_queue)

    def save_data(self, lidar_queue):
        #depth_map = np.float64(MEMS_Sensor.get_depth_map(raw_depth_map))
        frame = raw_depth_map.frame
        #depth_map = MEMS_Sensor.in_meters(depth_map)
        #depth_map[depth_map > self.max_range] = np.nan
        
       # point_cloud = self.depth_map_to_point_cloud(depth_map)
       # print(os.path.join(self.out_root, "%06d" % frame))
       # np.save(os.path.join(self.out_root, "%06d" % frame), point_cloud)

        # sensor_callback(point_cloud, self.lidar_queue)
        mems_lidar_data = self.lidar_queue.get(True, 1.0)
        print(os.path.join(self.out_root, "%06d" % frame))
        np.save(os.path.join(self.out_root, "%06d" % frame), mems_lidar_data)
        self.frame += 1
    
    def depth_map_to_point_cloud(self, depth_map):
        """
            Calculates a 3D point cloud according to
            the scan pattern and depth map.
        """
        cam_coords = self.K_inv[:3, :3] @ (self.pixel_coords_scan_pattern)
        cam_coords = cam_coords * depth_map[np.int32(self.pixel_coords_scan_pattern[1]),
                                 np.int32(self.pixel_coords_scan_pattern[0])].flatten()
        self.lidar_pc = cam_coords[np.logical_not(np.isnan(cam_coords))].reshape(3,-1)
        
        self.lidar_pc = self.lidar_pc[[2,0,1]]
        self.lidar_pc[2] *= -1
        self.rot_transl_pc()
        if self.add_noise:
            self.noise()
        return self.lidar_pc

    def destroy(self):
        if self.sensor:
            print("MEMS Sensor destroyed")
            self.sensor.destroy()
            
    def rot_transl_pc(self):
        rot_mat = R.from_euler("xyz",[self.parent.get_transform().rotation.roll - self.sensor.get_transform().rotation.roll, # roll
                                      self.parent.get_transform().rotation.pitch - self.sensor.get_transform().rotation.pitch, # pitch
                                      - self.parent.get_transform().rotation.yaw + self.sensor.get_transform().rotation.yaw], # yaw
                               degrees=True).as_matrix()
        self.lidar_pc = np.dot(rot_mat, self.lidar_pc)
        self.lidar_pc = (self.lidar_pc.T + [self.carla_transform.location.x, self.carla_transform.location.y, self.carla_transform.location.z]).T
    
    def noise(self):
        # Randomly dropping Points
        self.lidar_pc = self.lidar_pc[:,np.random.choice(self.lidar_pc.shape[1], np.int32(self.lidar_pc.shape[1]*0.95), replace=False)]
        
        # Disturb each point along the vector of its raycast
        self.lidar_pc += self.lidar_pc * np.random.normal(0, 0.005, self.lidar_pc.shape[1])
        
    @staticmethod
    def in_meters(depth_map):
        """
        Transforms the depth map image from RGB encoding to
        values in meter.
        See: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        """
        R_channel = depth_map[:,:,0].copy()
        G_channel = depth_map[:,:,1].copy()
        B_channel = depth_map[:,:,2].copy()
        depth_map = 1000 * (R_channel + G_channel * 256 + B_channel * 256 * 256) / (256 * 256 * 256 - 1)
        return depth_map

    @staticmethod
    def get_depth_map(raw_depth_map):
        """
        Fetchs the corresponding depth map image from the CARLA server.
        """
        raw_depth_map.convert(cc.Raw)
        
        depth_map = np.frombuffer(raw_depth_map.raw_data, dtype=np.dtype("uint8"))
        depth_map = np.reshape(depth_map, (raw_depth_map.height, raw_depth_map.width, 4))
        depth_map = depth_map[:, :, :3]
        depth_map = depth_map[:, :, ::-1]
        return depth_map

    @staticmethod
    def scan_pattern(n_scan_lines, n_points_pattern, width, height, h_fov_total = 80, h_fov_pc = 70, v_fov = 30, ramp_function = r_0, f = 150):
        """
        See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
        Generates the scan pattern.
        Input:
            n_scan_lines - Number of scan lines
            n_points_pattern - Number of points on the total scan pattern
            width - depth map width (influences the accuracy)
            height - depth map height (influences the accuracy)
            h_fov_total - horizontal field of view of the scan patter in total
            h_fov_pc - horizontal field of view of the point cloud
            v_fov - vertical field of view
            ramp_function - Ramp Function
            f - eigenfrequency
        
        Returns:
            Coordinates of the points on the depth image.
        """
        T = n_scan_lines / 2 / f
        t_list = np.arange(0, T, T / n_points_pattern)
        
        h_mirror_list = np.array([h_mirror(t, t_h_max = h_fov_total, f = f) for t in t_list])
        v_mirror_list = np.array([v_mirror(t, t_v_max = v_fov, f = f, T = T, ramp_function = ramp_function) for t in t_list])
        x = (width / (h_fov_total) * (h_mirror_list + h_fov_total / 2) - 1)[(h_mirror_list > - h_fov_pc / 2) & (h_mirror_list < h_fov_pc / 2)]
        y = (height / (v_fov) * (v_mirror_list + v_fov / 2) - 1)[(h_mirror_list > - h_fov_pc / 2) & (h_mirror_list < h_fov_pc / 2)]
        
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def rt_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.array([[c_p * c_y, s_y * c_p, s_p, x],
                      [c_y * s_p * s_r - s_y * c_r, s_y * s_p * s_r + c_y * c_r,-c_p * s_r , y],
                      [-c_y * s_p * c_r - s_y * s_r, -s_y * s_p * c_r + c_y * s_r, c_p * c_r,z],
                      [0, 0, 0,1]])
    return matrix

def my_color_map(vector):
    vector = vector - vector.min()
    vector = vector / vector.max() * 255
    
    c1 = vector
    c2 = 255 - vector
    c3 = (vector + 125) % 255
    
    return np.array([c1, c2, c3])

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    def int_mat(ImageSizeY, ImageSizeX, CameraFOV):
        Focus_length = ImageSizeX /(2 * np.tan(CameraFOV * np.pi / 360))
        Center_X = ImageSizeX / 2
        Center_Y = ImageSizeY / 2
        return np.array([Focus_length, 0, Center_X, 0, Focus_length, Center_Y, 0, 0, 1]).reshape((3,3))

    def transfer_points(points, rot_t):
        points = np.concatenate([points, np.ones([1,points.shape[1]])])
        points[0:3,:] = np.dot(rot_t, points[0:4,:])[0:3, :]
        return points[0:3].T

    def proj2d(points, K, img_size, rot_t):
        points = transfer_points(points, rot_t).T
        points = np.concatenate([points[1,:], -points[2,:], points[0,:]]).reshape(3,-1)
        pts_image = np.dot(K, points)
        pts_image /= pts_image[2]
        pts_image = np.delete(pts_image, 2, 0)
        
        mask = (pts_image[0,:] <= img_size[1]) & (pts_image[1,:] <= img_size[0]) & \
                    (pts_image[0,:] >= 0) & (pts_image[1,:] >= 0) & \
                    (points[2,:] >= 0)
        return pts_image[:,mask], mask
    
    """
    Generating the Data and save it.
    """ 

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        world = client.get_world()
        
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
    finally:
        print("loading finished")

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    vehicle_bp = world.get_blueprint_library().filter("vehicle.audi.etron")[0]
       
    try:
        car = world.spawn_actor(vehicle_bp, carla.Transform(carla.Location(-82., 13.5, 1.3083857), carla.Rotation(0,0,0)))
        sleep(1)
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(0, 0, 2)) ,attach_to=car)
        
        camera.listen(lambda img: img.save_to_disk(f"out/test/img/{img.frame:06d}.png"))
        
        mems = MEMS_Sensor(car, carla.Transform(carla.Location(0, 0, 2)), out_root = "out/test/mems/",
                            n_scan_lines = 50, n_points_pattern = 50000, max_range = 75)
        
        sleep(1)
    finally:
        camera.destroy()
        mems.destroy()
        car.destroy()

    """
    Map the Data on the Image.
    """
    mems_path = os.listdir("out/test/mems/")
    mems_path.sort()
    mems_lidar = np.load("out/test/mems/" + mems_path[-1])
    
    
    img_path = os.listdir("out/test/img/")
    img_path.sort()

    img = cv2.imread("out/test/img/" + img_path[-1])
    
    
    K =  int_mat(img.shape[0],img.shape[1], 90)
    rot = rt_matrix(0,0,-2,0,0,0) # Note that the MEMS LiDAR Data origin is at the origin of the car. We have to translate the data to the origin of the camera.
    
    pc_proj, mask = proj2d(mems_lidar, K, img.shape, rot)
    dist = np.sqrt((mems_lidar[:,mask]**2).sum(0))
    cmap = my_color_map(dist).T
    
    for p, c in zip(pc_proj.T, cmap):
        cv2.circle(img,(int(p[0]),int(p[1])), 1, c, -1)

    cv2.imwrite("image_with_points.png", img)
