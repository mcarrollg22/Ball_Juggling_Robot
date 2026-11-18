#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 18:16:47 2023

@author: zxc703
"""

import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import copy

'''------------------ COMPLETE THE FUNCTIONS BELOW    --------------------------'''

def threshold(image_array,thresh=25):
    '''
    PART 1 QUESTION 1(a)
    Apply a threshold to a grayscale image array to identify pixel positions exceeding a specified threshold.

    Args:
    image_array (numpy.ndarray): A 2D NumPy array representing an image with pixel values to be thresholded.
    thresh (int, optional): The threshold value above which pixel positions are identified.

    Returns:
    tuple: A tuple containing two NumPy arrays. The first array represents the y-positions (rows) of pixels exceeding the threshold,
        and the second array represents the x-positions (columns) of pixels exceeding the threshold.
    
    '''
    pixel_positions = np.where(image_array > thresh)

    # hint the np.where function can be useful
    
    return pixel_positions


def get_pixel_centroid(pixel_positions):
    '''
    PART 1 QUESTION 1(b)
    Calculate the centroid (center of mass) of a set of pixels in an image based on their positions.

    Args:
        pixel_positions (numpy.ndarray): A 2 element list with numpy vectors of length N representing the positions of pixels in an image.
        The first element contains the y-positions (rows) of the pixels, and the second row contains the x-positions (columns) of the pixels.

    Returns:
    numpy.ndarray: A 1D NumPy array of shape (2,) representing the centroid coordinates of the pixel positions.
        The first element of the array (centroid[0]) corresponds to the x-coordinate (u-bar),
        and the second element (centroid[1]) corresponds to the y-coordinate (v-bar) of the centroid.'''
    y_pos, x_pos = pixel_positions
    
    u_bar = np.mean(x_pos)
    v_bar = np.mean(y_pos)
    
    return np.array((u_bar,v_bar))


def compute_pos_from_pix(pixel_uv,resolution,focal_length,pixels_per_m,z_distance):
    '''
    PART 1 QUESTION 1(c)
    Calculate the real-world position coordinates from pixel coordinates in a camera image.
    
    Args:
        pixel_uv (tuple): A tuple representing the pixel coordinates (u, v) in the camera image.
        resolution (int): The resolution of the camera image, typically measured in pixels.
        focal_length (float): The focal length of the camera lens in meters.
        pixels_per_inch (float): The number of pixels per inch of the camera sensor.
        z_distance (float): The distance of the object from the camera along the optical axis, in meters.
    
    Returns:
        numpy.ndarray: A 1D NumPy array of shape (3,) representing the real-world position coordinates (x, y, z) of the object.
    
    '''
    w,h = resolution
    u_o = w/2
    v_o = h/2
    
    u,v = pixel_uv
    
    x_image = (u-u_o) *pixels_per_m
    y_image = (v-v_o) * pixels_per_m
    
    x_pos = x_image * z_distance / focal_length
    y_pos = y_image * z_distance / focal_length
    
    return np.array((x_pos,y_pos,z_distance))

def threshold_RGB(image_array,target_color,thresh=25):
    '''
    PART 2 QUESTION 1
    Apply a threshold to an RGB image array to identify pixel positions exceeding a specified threshold in a specific color channel.

    Args:
        image_array (numpy.ndarray): A 3D NumPy array representing an RGB image with pixel values to be thresholded.
        target_color (int): An integer representing the color channel (0 for red, 1 for green, 2 for blue) to be thresholded.
        thresh (int, optional): The threshold value above which pixel positions are identified.

    Returns:
        tuple: A tuple containing two NumPy arrays. The first array represents the y-positions (rows) of pixels exceeding the threshold in the specified color channel,
            and the second array represents the x-positions (columns) of pixels exceeding the threshold in the specified color channel.
    
    '''
    pixel_positions=0 #placeholder
    color_image = image_array[:,:,target_color]
    pixel_positions = threshold(color_image,thresh)
    
    return pixel_positions