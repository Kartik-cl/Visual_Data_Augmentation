import argparse

import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--image", default='../../keras-yolo3/Mask_Detection_Balanced/DataSet10/warp_x_024f6398-2bd8-4251-ac92-9f13c63d141e.jpg', help="image for prediction")
parser.add_argument("--config", default='../custom_data/cfg/yolov3-custom.cfg', help="YOLO config path")
parser.add_argument("--weights", default='backup/yolov3-custom_900.weights', help="YOLO weights path")
parser.add_argument("--names", default='./model/custom.names', help="class names path")
args = parser.parse_args()

CONF_THRESH, NMS_THRESH = 0.1, 0.1

# Load the network
net = cv2.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_best.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
img_path = '/home/kartvat3/test_data_ds3/newDemo_GT/'

for img_name in os.listdir(img_path):
	if(".xml" in img_name or ".mp4" in img_name or ".json" in img_name):
		continue
	img = cv2.imread(os.path.join(img_path, img_name))
	print(img_name)
	print(img.shape)

	height, width = img.shape[:2]

	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layer_outputs = net.forward(output_layers)


	class_ids, confidences, b_boxes = [], [], []
	for output in layer_outputs:
    		for detection in output:
        		scores = detection[5:]
			#print("scores are: ", scores)
        		class_id = np.argmax(scores)
			#print(class_id)
        		confidence = scores[class_id]

        		if confidence > CONF_THRESH:
            			center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

            			x = int(center_x - w / 2)
            			y = int(center_y - h / 2)

            			b_boxes.append([x, y, int(w), int(h)])
            			confidences.append(float(confidence))
            			class_ids.append(int(class_id))

	# Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
	if len(b_boxes) == 0:
		continue

	indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

	print(class_ids)
	print(indices)
	print(confidences)

	# Draw the filtered bounding boxes with their class to the image
	with open(args.names, "r") as f:
   		 classes = [line.strip() for line in f.readlines()]
	colors = np.random.uniform(0, 255, size=(100, 3))

	print(classes[class_ids[indices[0]]])


	for index in indices:
    		x, y, w, h = b_boxes[index]
    		cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
    		cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)
    		if classes[class_ids[index]]=='mask_ok':
        		mask_status = 'masked'
    		else:
        		mask_status = 'unmasked'
    			conf_score = str(confidences[index])
                print(mask_status + " " + conf_score + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h))
		
                with open(os.path.join('ds3_predictions', img_name.replace(".jpg", ".txt")), "a") as f:
		    f.write(mask_status + " " + conf_score + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h) + "\n")
			
                cv2.imwrite(os.path.join("ds3_results_imgs", img_name), img)

#cv2.imshow("image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
