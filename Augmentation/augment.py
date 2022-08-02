import os
import cv2
import xml.etree.ElementTree as ET
import numpy
import configparser
from ast import literal_eval
from operations import blur, rotate, perspective, flip, brightness, write_annotation, write_image, update_annotation, supress_bboxes
import copy

def read_annotations(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()
	bboxes = []
	i = 1
	while(1):
		if(i<len(root)):
			if(root[i].tag == "object"):
				if(root[i][0].text == 'none'):
					root[i][0].text == 'good'
				bboxes.append([float(root[i][2][0].text), float(root[i][2][1].text), float(root[i][2][2].text), float(root[i][2][3].text), 0])
				i+=1
		else:
			break
	return bboxes, tree


def call_augmentation_functions(aug_types, data_path, config):
	for file in os.listdir(data_path):
		#file = "d154b1e4-48d9-42c2-bb8b-dbf2762f9265.jpg"
		try:
			if not file.endswith("jpg"):
				continue
			img = cv2.imread(os.path.join(data_path, file))
			bboxes, tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml"))) # keep a deep copy
			bboxes = numpy.array([numpy.array(x) for x in bboxes])

			
			###blur###
			if(aug_types['blur'] == 'Y'):
				bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
				bboxes = numpy.array([numpy.array(x) for x in bboxes])
				ksize = config['BLUR_KSIZE']['ksize']
				img_, bboxes_ = blur(img, bboxes, ksize)
				tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "blur_" + file.replace(".jpg", ".xml"))
				write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "blur_" + file.replace(".jpg", ".xml"))
				write_image(config['DATASET_PATH']['images_destination'], "blur_" + file, img_)

			###warp###
			if(aug_types['warp'] == 'Y'):
				if(config['WARP']['warp_x'] == 'Y'):
					e_x = literal_eval(config['WARP']['e_x'])
					for e in e_x:
						bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
						bboxes = numpy.array([numpy.array(x) for x in bboxes])
						img_, bboxes_ =  perspective(img, bboxes, 1, e)
						tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "warp_x" + "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						tree = supress_bboxes(tree, img_.shape[0], img_.shape[1], bboxes_)
						write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "warp_x" + "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						write_image(config['DATASET_PATH']['images_destination'], "warp_x" + "-" + str(e) + "_" + file, img_)
						
				if(config['WARP']['warp_y'] == 'Y'):
					e_y =literal_eval(config['WARP']['e_y'])
					for e in e_y:
						bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
						bboxes = numpy.array([numpy.array(x) for x in bboxes])
						img_, bboxes_ =  perspective(img, bboxes, 0, e)
						tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "warp_y"+ "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						tree = supress_bboxes(tree, img_.shape[0], img_.shape[1], bboxes_)
						write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "warp_y" + "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						write_image(config['DATASET_PATH']['images_destination'], "warp_y" + "-" + str(e) + "_" + file, img_)

				if(config['WARP']['warp_x_y'] == 'Y'):
					e_x_y =literal_eval(config['WARP']['e_x_y'])
					for e in e_x_y:
						bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
						bboxes = numpy.array([numpy.array(x) for x in bboxes])		
						img_, bboxes_ =  perspective(img, bboxes, 2, e)
						tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "warp_x_y"+ "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						if f == 1:
							continue
						tree = supress_bboxes(tree, img_.shape[0], img_.shape[1], bboxes_)
						write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "warp_x_y" + "-" + str(e) + "_" + file.replace(".jpg", ".xml"))
						write_image(config['DATASET_PATH']['images_destination'], "warp_x_y" + "-" + str(e) + "_" + file, img_)
					

			###rotate###
			if(aug_types['rotate'] == 'Y'):
				angles = numpy.arange(int(config['ROTATION_ANGLES']['min']), int(config['ROTATION_ANGLES']['max']), int(config['ROTATION_ANGLES']['interval']))
				index = numpy.argwhere(angles==0)
				angles = numpy.delete(angles, index)
				for angle in angles:
					img_, bboxes_ = rotate(img, bboxes, angle)

					if(angle<0):
						bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
						bboxes = numpy.array([numpy.array(x) for x in bboxes])
						tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "m" + str(-angle) + "_" + file.replace(".jpg", ".xml"))
						if angle == -10 and f == 1:
							continue
						tree = supress_bboxes(tree, img_.shape[0], img_.shape[1], bboxes_)
						write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "m" + str(-angle) + "_" + file.replace(".jpg", ".xml"))
						write_image(config['DATASET_PATH']['images_destination'], "m" + str(-angle) + "_" + file, img_)
					else:
						bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
						bboxes = numpy.array([numpy.array(x) for x in bboxes])
						tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], str(angle) + "_" + file.replace(".jpg", ".xml"))
						if angle == 10 and f == 1:
							continue
						tree = supress_bboxes(tree, img_.shape[0], img_.shape[1], bboxes_)
						write_annotation(tree, config['DATASET_PATH']['annotations_destination'], str(angle) + "_" + file.replace(".jpg", ".xml"))
						write_image(config['DATASET_PATH']['images_destination'], str(angle) + "_" + file, img_)
				
			###flip###
			if(aug_types['flip'] == 'Y'):
				bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
				bboxes = numpy.array([numpy.array(x) for x in bboxes])
				img_, bboxes_ = flip(img, bboxes)
				tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "flip" + "_" + file.replace(".jpg", ".xml"))
				write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "flip" + "_" + file.replace(".jpg", ".xml"))
				write_image(config['DATASET_PATH']['images_destination'], "flip_" + file, img_)
			###bright###
			if(aug_types['bright'] == 'Y'):
				bboxes, orig_tree = read_annotations(os.path.join(data_path, file.replace("jpg", "xml")))
				bboxes = numpy.array([numpy.array(x) for x in bboxes])
				img_, bboxes_ = brightness(img, bboxes)
				tree, f = update_annotation(orig_tree, bboxes_, config['DATASET_PATH']['annotations_destination'], "bright" + "_" + file.replace(".jpg", ".xml"))
				write_annotation(tree, config['DATASET_PATH']['annotations_destination'], "bright" + "_" + file.replace(".jpg", ".xml"))
				write_image(config['DATASET_PATH']['images_destination'], "bright_" + file, img_)

		except Exception as e:
		 	print(e)
		 	pass

