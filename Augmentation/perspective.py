# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:33:20 2020

@author: AshishARao, Kartikeya Vats
"""

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from ast import literal_eval


def perspective_augmentation(input_image, bboxes, axis, e):

	#input_image = cv2.imread('-1x-1.jpg')
	if(axis == 0):	
		e = e/input_image.shape[0]
		h = np.array([[1, 0, 0], [0, 1, 0], [0, e, 1]], dtype=np.float64)
	elif(axis == 1):
		e = e/input_image.shape[1]
		h = np.array([[1, 0, 0], [0, 1, 0], [e, 0, 1]], dtype=np.float64)
	else:
		e_y = e/input_image.shape[0]
		e_x = e/input_image.shape[1]
		h = np.array([[1, 0, 0], [0, 1, 0], [e_x, e_y, 1]], dtype=np.float64)

	height, width, ch = np.shape(input_image)
	output_image = cv2.warpPerspective(input_image, h, (width, height))

	bboxes_ = []

	for box in bboxes:

		xmin = int(box[0])
		ymin = int(box[1])
		xmax = int(box[2])
		ymax = int(box[3])

		tl_x = (h[0][0]*xmin + h[0][1]*ymin + h[0][2]) / ((h[2][0]*xmin + h[2][1]*ymin + h[2][2]))
		tl_y = (h[1][0]*xmin + h[1][1]*ymin + h[1][2]) / ((h[2][0]*xmin + h[2][1]*ymin + h[2][2]))

		tr_x = (h[0][0]*xmax + h[0][1]*ymin + h[0][2]) / ((h[2][0]*xmax + h[2][1]*ymin + h[2][2]))
		tr_y = (h[1][0]*xmax + h[1][1]*ymin + h[1][2]) / ((h[2][0]*xmax + h[2][1]*ymin + h[2][2]))

		br_x = (h[0][0]*xmax + h[0][1]*ymax + h[0][2]) / ((h[2][0]*xmax + h[2][1]*ymax + h[2][2]))
		br_y = (h[1][0]*xmax + h[1][1]*ymax + h[1][2]) / ((h[2][0]*xmax + h[2][1]*ymax + h[2][2]))

		bl_x = (h[0][0]*xmin + h[0][1]*ymax + h[0][2]) / ((h[2][0]*xmin + h[2][1]*ymax + h[2][2]))
		bl_y = (h[1][0]*xmin + h[1][1]*ymax + h[1][2]) / ((h[2][0]*xmin + h[2][1]*ymax + h[2][2]))

		x_rect_min = int(max(tl_x, bl_x))
		y_rect_min = int(tl_y)

	
		x_rect_max = int(min(tr_x, br_x))
		y_rect_max = int(br_y)

		bboxes_.append([x_rect_min, y_rect_min, x_rect_max, y_rect_max])


	original_corners = [(0,0), (input_image.shape[1],0), (input_image.shape[1], input_image.shape[0]), (0, input_image.shape[1])]
	new_corners = []

	for corner in original_corners:
		new_x = (h[0][0]*corner[0] + h[0][1]*corner[1] + h[0][2]) / ((h[2][0]*corner[0] + h[2][1]*corner[1] + h[2][2]))
		new_y = (h[1][0]*corner[0] + h[1][1]*corner[1] + h[1][2]) / ((h[2][0]*corner[0] + h[2][1]*corner[1] + h[2][2]))
		new_corners.append((new_x, new_y))


	tl_x = new_corners[0][0]
	tl_y = new_corners[0][1]

	tr_x = new_corners[1][0]
	tr_y = new_corners[1][1]

	br_x = new_corners[2][0]
	br_y = new_corners[2][1]

	bl_x = new_corners[3][0]
	bl_y = new_corners[3][1]

	x1_rect_min = int(max(tl_x, bl_x))
	y1_rect_min = int(tl_y)

	x1_rect_max = int(min(tr_x, br_x))
	y1_rect_max = int(br_y)

	output_image = output_image[y1_rect_min:y1_rect_max + 1, x1_rect_min:x1_rect_max + 1]

	return output_image, bboxes_
