from __future__ import division
import cv2
import math
from ast import literal_eval
from rotate import rotate_image, rotatedRectWithMaxArea
from perspective import perspective_augmentation
from data_aug.data_aug import *
from data_aug.bbox_util import *
import lxml.etree as le
import copy


def blur(img, bboxes, ksize):
	ksize = literal_eval(ksize)
	img_ = cv2.blur(img, ksize)
	bboxes_ = bboxes
	return img_, bboxes_

def rotate(img, bboxes, angle):
	#img_rotated, bboxes_rotated =  Rotate(angle)(img.copy(), bboxes.copy())
	img_rotated = rotate_image(img, angle)
	bboxes_rotated = Rotate_boxes(angle)(img.copy(), bboxes.copy())
	w, h = rotatedRectWithMaxArea(img.shape[1], img.shape[0], math.radians(angle))
	top_left = int((img_rotated.shape[1]-w)/2), int((img_rotated.shape[0]-h)/2)
	bottom_right = int((img_rotated.shape[1]+w)/2), int((img_rotated.shape[0]+h)/2)
	image_rotated_cropped = img_rotated[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
	bboxes_adjusted = [[bboxes_rotated[l][0]-top_left[0],bboxes_rotated[l][1]-top_left[1],bboxes_rotated[l][2]-top_left[0], bboxes_rotated[l][3]-top_left[1],0] for l in range(len(bboxes_rotated))]
	return image_rotated_cropped, bboxes_adjusted

def perspective(img, bboxes, axis, e):
	img_, bboxes_ = perspective_augmentation(img, bboxes, axis, e)
	bboxes_temp = []
	for box in bboxes_:
		box.append(0)
		bboxes_temp.append(box)
	bboxes_ = bboxes_temp
	return img_, bboxes_

def flip(img, bboxes):
	img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
	return img_, bboxes_

def brightness(img, bboxes):
	img_, bboxes_ = RandomHSV(None, None, (-20,20))(img.copy(), bboxes.copy())
	#img_, bboxes_ = increase_brightness(img, value, bboxes)
	return img_, bboxes_

def update_annotation(tree, bboxes, dest_path, ann_file):
	root = tree.getroot()
	i = 0
	f = 0
	ctr = 0
	for elem in root.iter('object'):
		for child in elem:
			if child.tag == 'file_id':
				child.text = ann_file[:-4]
		for child in elem:
			if child.tag == 'name':
				if child.text == 'mask_ok':
					f = 1
	for elem in root.iter('object'):
		for box in root.iter('bndbox'):
			j = 1
			for rec in box:
				if(j==1):
					rec.text = str(int(bboxes[i][0]))
				elif(j==2):
					rec.text = str(int(bboxes[i][1]))
				elif(j==3):
					rec.text = str(int(bboxes[i][2]))
				else:
					rec.text = str(int(bboxes[i][3]))
				j+=1
			i+=1
		return tree, f
	#tree.write(os.path.join(dest_path, ann_file))

def write_annotation(tree, dest_path, ann_file):
	tree.write(os.path.join(dest_path, ann_file))

def write_image(dest_path, img_file, img):
	#img = img[:,:,::-1]
	cv2.imwrite(os.path.join(dest_path, img_file), img)

def supress_bboxes(tree, max_y, max_x, bboxes_new): #test this #have a threshold to remove boxes
	
	#print(bboxes_new)

	bboxes_new_trimmed = [[0 if box[i] < 0 else box[i] for i in range(len(box))] for box in bboxes_new]
	bboxes_new_trimmed = [[max_y if (box[i] > max_y and i%2==1) else box[i] for i in range(len(box))] for box in bboxes_new_trimmed]
	bboxes_new_trimmed = [[max_x if (box[i] > max_x and i%2==0) else box[i] for i in range(len(box))] for box in bboxes_new_trimmed]
	orig_area = [[(box[2] - box[0]) * (box[3] - box[1])] for box in bboxes_new]
	trimmed_area = [[(box[2] - box[0]) * (box[3] - box[1])] for box in bboxes_new_trimmed]
	orig_area_flat = [item for sublist in orig_area for item in sublist]
	trimmed_area_flat = [item for sublist in trimmed_area for item in sublist]
	area_ratio = [x/y for x, y in zip(trimmed_area_flat, orig_area_flat)]
	to_remove = [0 if area_ratio[i] > 0.8 else 1 for i in range(len(area_ratio))]
	#print(len(to_remove))
	root = tree.getroot()
	i = 0
	p = 0
	obj_ctr = 0
	
	for obj in root.findall('object'):
		obj_ctr+=1
		if to_remove[i] == 1:
			root.remove(root[obj_ctr])
			obj_ctr-=1
		else:
			for elem in obj:
				if elem.tag == 'bndbox':
					if int(elem[0].text) < 0:
						elem[0].text = '0'
					if int(elem[1].text) < 0:
						elem[1].text = '0'
					if int(elem[2].text) < 0:
						elem[2].text = '0'
					if int(elem[3].text) < 0:
						elem[3].text = '0'

		i+=1


			# if to_remove[i] == 1:
			# 	root.remove(root[3])
			# 	i = i + 1
			# 	continue
			# for rec in box:
			# 	if int(rec.text)<0:
			# 		rec.text = '0'
			# 		print(rec.text)
			# i = i + 1

	# while(1):
	# 	if(root[i].tag == "object"):
	# 		flag = 1
	# 		print("if true")
	# 		print(to_remove)
	# 		input("p")
	# 		if(to_remove[p] == 1):
	# 			root.remove(root[i])
	# 			#print(int(root[i][2][1].text))
	# 			p+=1
	# 		else:
	# 			for k in range(4):
	# 				print(int(root[i][2][k].text))
	# 				input("hey")
	# 				if(int(root[i][2][k].text)<0):
	# 					root[i][2][k].text = '0'
	# 				if(k%2 == 1):
	# 					if(int(root[i][2][k].text)>max_y):
	# 						root[i][2][k].text = str(max_y)
	# 				else:
	# 					if(int(root[i][2][k].text)>max_x):
	# 						root[i][2][k].text = str(max_x)
	# 			p+=1	
	# 	else:
	# 		if(flag == 1):
	# 			break
	# 	i+=1
	return tree