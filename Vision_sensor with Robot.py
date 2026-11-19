# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:35:58 2026

@author: jinki
"""

import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import copy
import my_functions_robot as func
import ecse373_vision_utils_robot as util
import time

# Declare the following parameters
f = 1.732 # focal length in meters, found using the angle of view
pixels_per_m = 0.00781
resolution = (256, 256)


vision_mode = "gray" # or "RGB"

client = zmq.RemoteAPIClient()
sim = client.getObject('sim')

for handle in sim.getObjectsInTree(sim.handle_scene, sim.object_visionsensor_type, 0):
    print(sim.getObjectAlias(handle, 2))

# Get the camera handle
camera_handle = sim.getObject("/Franka/Vision_plate")

sensor_path = "/Franka/Vision_plate"  # Use the actual path in your scene
camera_handle = sim.getObject(sensor_path)
time.sleep(3)
# Obtain the transformation of the camera in the world frame
T_cam_world = np.array(sim.getObjectMatrix(camera_handle,-1)).reshape(3,4)

if vision_mode == "gray":
    image,resolution = sim.getVisionSensorImg(camera_handle,1)
    image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1]) 
elif vision_mode == "RGB":
    image,resolution = sim.getVisionSensorImg(camera_handle,0)
    image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1],3)
else:
    print("invalid!")

image = np.flip(image,axis=1)

plt.figure()
if vision_mode == "gray":
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
else:
    plt.imshow(image)

plt.title("CoppeliaSim camera image")
plt.xlabel("Column (u)")
plt.ylabel("Row (v)")
plt.show()

#Setting T1
#pos_world = [0.53055,0.00001,0.8218]


# 1. threshold the image to get the pixel locations where the cube is located
pixel_positions = func.threshold(image)
current_dia = max(pixel_positions[1]) - min(pixel_positions[1])
Z = -0.231 * np.log(current_dia) + 1.3372
# 2. calculate the pixel centroid
pixel_centroids = func.get_pixel_centroid(pixel_positions)
# 3. calculate the position of the centroid in the camera coordinate frame
centroid_pos_world = func.compute_pos_from_pix(pixel_centroids,resolution,f,pixels_per_m,Z)
# 4. compute pos_world by converting the centroid position from the camera coordinate frame to the world coordinate frame using the util.hand_eye_transform function
pos_world = util.hand_eye_transform(centroid_pos_world, T_cam_world)
util.move_to(sim, pos_world)
n = 1

while Z  > 0.25:
    
    T_cam_world = np.array(sim.getObjectMatrix(camera_handle,-1)).reshape(3,4)
    
    if vision_mode == "gray":
        image,resolution = sim.getVisionSensorImg(camera_handle,1)
        image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1]) 
    elif vision_mode == "RGB":
        image,resolution = sim.getVisionSensorImg(camera_handle,0)
        image = np.array(bytearray(image),dtype='uint8').reshape(resolution[0],resolution[1],3)
    else:
        print("invalid!")
    
    image = np.flip(image,axis=1)
        
    plt.figure()
    if vision_mode == "gray":
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(image)

    plt.title("CoppeliaSim camera image")
    plt.xlabel("Column (u)")
    plt.ylabel("Row (v)")
    plt.show()

    pixel_positions = func.threshold(image)
    current_dia = max(pixel_positions[1]) - min(pixel_positions[1])
    Z = -0.231 * np.log(current_dia) + 1.3372
    # 2. calculate the pixel centroid
    pixel_centroids = func.get_pixel_centroid(pixel_positions)
    # 3. calculate the position of the centroid in the camera coordinate frame
    centroid_pos_world = func.compute_pos_from_pix(pixel_centroids,resolution,f,pixels_per_m,Z)
    # 4. compute pos_world by converting the centroid position from the camera coordinate frame to the world coordinate frame using the util.hand_eye_transform function
    pos_world = util.hand_eye_transform(centroid_pos_world, T_cam_world)  
    util.move_to(sim, pos_world)
    
    n = n + 1
    time.sleep(0.001)  # Give the running simulation time to update the image


m = 2

