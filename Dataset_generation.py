#!/usr/bin/env python

import glob
import os
import sys
import time
from queue import Queue
from queue import Empty
from matplotlib import cm
import numpy as np

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])



try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
from memsclass import MEMS_Sensor



def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

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

def main():
    actor_list = []

    IM_WIDTH = 1920
    IM_HEIGHT = 1080

    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        #spawining ego vehicle

        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        transform = carla.Transform(carla.Location(x=-60.5, y=-140, z=0.8),
                                    carla.Rotation(pitch=0.000000, yaw=0.0, roll=0.000000))
        ego_vehicle = world.spawn_actor(bp, transform)

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.5
        world.apply_settings(settings)


        actor_list.append(ego_vehicle)
        print('created %s' % ego_vehicle.type_id)

        #Spawining RGB camera sensor

        blueprint = blueprint_library.find('sensor.camera.rgb')

        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '120')

        spawn_point = carla.Transform(carla.Location(x=0.0, z=1.5))

        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=ego_vehicle,
                                   attachment_type=carla.AttachmentType.Rigid)


        actor_list.append(sensor)

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]

        fov = 110
        focal = IM_WIDTH / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = IM_WIDTH / 2.0
        K[1, 2] = IM_HEIGHT / 2.0

        print(K)

        #Spawning extra actors

        #transform.location.x += 2
        transform.location.x += 10
        bp = world.get_blueprint_library().filter("vehicle.audi.etron")[0]

        npc = world.try_spawn_actor(bp, transform)
        if npc is not None:
            actor_list.append(npc)
            print('created %s' % npc.type_id)
        else:
            print("collision")

        #Spawning MEMS lidar

        mems_sensor = MEMS_Sensor(ego_vehicle, carla.Transform(carla.Location(x=0.0, z=1.5)))

        actor_list.append(mems_sensor)
        print(actor_list)
        image_queue = Queue()
        lidar_queue = Queue()
        sensor.listen(lambda data: sensor_callback(data, image_queue))

        #Synchronising the data of camera and MEMS
        for frame in range(50):
            world.tick()
            world_frame = world.get_snapshot().frame
            #print('entered loop')
            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                image_data.save_to_disk('tutorial/output/%.6d.jpg' % image_data.frame)

                mems_lidar_data = mems_sensor.lidar_queue.get(True, 1.0)
                print(os.path.join(mems_sensor.out_root, "%06d" % mems_sensor.frame))
                np.save(os.path.join(mems_sensor.out_root, "%06d" % mems_sensor.frame), mems_lidar_data)
                mems_sensor.frame += 1
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d" %
                             (frame, 50, world_frame, image_data.frame, mems_sensor.frame) + ' ')
            sys.stdout.flush()
            print(mems_lidar_data.shape)

                # Add an extra 1.0 at the end of each 3d point so it becomes of
                # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.

            mems_lidar_data = np.r_[mems_lidar_data, [np.ones(mems_lidar_data.shape[1])]]
            print(mems_lidar_data.shape)
            print(mems_lidar_data)
            rot = rt_matrix(0.0,0.0,-1.5,0,0,0)
            mems_lidar_data = np.dot(rot, mems_lidar_data)
            print(mems_lidar_data)


            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = mems_sensor.sensor.get_transform().get_matrix()
            print(lidar_2_world)

            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, mems_lidar_data)
            print(world_points.shape)

            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(sensor.get_transform().get_inverse_matrix())
            print(world_2_camera.shape)
            print(world_2_camera)

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)
            print(sensor_points.shape)
            # Now we must change from UE4's coordinate system to an "standard"
            # camera coordinate system (the same used by OpenCV):

            # ^ z                       . z
            # |                        /
            # |              to:      +-------> x
            # | . x                   |
            # |/                      |
            # +-------> y             v y

            # This can be achieved by multiplying by the following matrix:
            # [[ 0,  1,  0 ],
            #  [ 0,  0, -1 ],
            #  [ 1,  0,  0 ]]

            # Or, in this case, is the same as swapping:
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

            # Finally we can use our K matrix to do the actual 3D -> 2D.
            points_2d = np.dot(K, point_in_camera_coords)

            # normalize the x, y values by the 3rd value.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            print(points_2d.shape)

            # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            # contains all the y values of our points. In order to properly
            # visualize everything on a screen, the points that are out of the screen
            # must be discarded, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            print(points_2d.shape)
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < IM_WIDTH) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < IM_HEIGHT) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]

            print(points_2d.shape)
            points_2d[:, 0] = points_2d[:, 0].astype(np.int32)
            points_2d[:, 1] = points_2d[:, 1].astype(np.int32)

                #Saving the mapped lidar points as text file

            np.savetxt('points.txt',points_2d)
            a = np.loadtxt('points.txt')
            print(a)


            u_coord = points_2d[:, 0].astype(np.int32)
            v_coord = points_2d[:, 1].astype(np.int32)
            print(u_coord)
            print(v_coord)


            intensity = np.ones((points_2d.shape[0],), dtype=int)
            print(intensity)

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            color_map = np.array([
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int32).T
            dot_extent = 2
            if dot_extent <= 0:
                    # Draw the 2d points on the image as a single pixel using numpy.
                    im_array[v_coord, u_coord] = color_map
            else:
                     #Draw the 2d points on the image as squares of extent args.dot_extent.
                for i in range(len(points_2d)):
                    im_array[
                    v_coord[i]-dot_extent : v_coord[i]+dot_extent ,
                    u_coord[i]-dot_extent : u_coord[i]+dot_extent ] = color_map[i]

                # Save the image using Pillow module.
            image = Image.fromarray(im_array)
            image.save("_out/%08d.png" % image_data.frame)


        time.sleep(20)

    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()

        print('done.')


if __name__ == '__main__':
    main()
