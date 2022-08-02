from __future__ import division
import os
import cv2
import numpy as np


np.set_printoptions(precision=3,suppress=True)

imagefile='0_10725.jpg'

image=cv2.imread(imagefile)

image_shape=image.shape


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x),int(y)
    else:
        return False

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  print (rot_mat)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result,rot_mat

rotated_image,rot_mat=rotate_image(image,-15)

#
print ("##################################")
#
corner_points=np.array([[0,0,1],\
  [image_shape[1],0,1],\
    [image_shape[1],image_shape[0],1],\
      [0,image_shape[0],1]]\
        )
corner_points=corner_points.T
# print (corner_points)
# print ("##################################")
rotated_corners=rot_mat.dot(corner_points)
# print (rotated_corners)
# print ("##################################")
rotated_corners_homographic=np.vstack([rotated_corners,np.ones((1,4))])
# print (rotated_corners_homographic)
# print ("##################################")
##
def get_line_homography(point1,point2):
  line_parameters=np.cross(point1,point2)
  return line_parameters

print((corner_points[:,0]),(corner_points[:,1]))
input("hey")
#
equation_top=get_line_homography(corner_points[:,0],corner_points[:,1])
equation_right=get_line_homography(corner_points[:,1],corner_points[:,2])
equation_bottom=get_line_homography(corner_points[:,2],corner_points[:,3])
equation_left=get_line_homography(corner_points[:,3],corner_points[:,0])
# print ("##################################")
# print (equation_top)  
# print (equation_right)  
# print (equation_bottom)  
# print (equation_left)  
# print ("##################################")
rotated_equation_top=get_line_homography(rotated_corners_homographic[:,0],rotated_corners_homographic[:,1])
rotated_equation_right=get_line_homography(rotated_corners_homographic[:,1],rotated_corners_homographic[:,2])
rotated_equation_bottom=get_line_homography(rotated_corners_homographic[:,2],rotated_corners_homographic[:,3])
rotated_equation_left=get_line_homography(rotated_corners_homographic[:,3],rotated_corners_homographic[:,0])
# print ("##################################")
# print (rotated_equation_top)  
# print (rotated_equation_right)  
# print (rotated_equation_bottom)  
# print (rotated_equation_left)  
# print ("##################################")
def get_point_of_intersection(line1,line2):
  a=np.cross(line1,line2)
  # print(a)
  # input("a")
  # print (a[0],a[1],a[2])
  b=[a[0]/a[2],a[1]/a[2]]
  b=np.array(b,dtype=np.int)
  b=tuple(b)
  # print (b)
  return b


def get_intersection_crop(r_t, r_b, b_l, l_t, rotated_image):

  intesection_points = []

  r_t = np.append(np.asarray(r_t),1)
  w_r_t = np.append(np.asarray((0, r_t[1])),1)
  r_b = np.append(np.asarray(r_b),1)
  w_r_b = np.append(np.asarray((r_b[0],0)),1)
  b_l = np.append(np.asarray(b_l),1)
  w_b_l = np.append(np.asarray((image.shape[1],b_l[1])),1)
  l_t = np.append(np.asarray(l_t),1)
  w_l_t = np.append(np.asarray((l_t[0], image.shape[1])),1)
  
  # print(rotated_image.shape)
  # print(image.shape)
  # print(r_b)
  # print(w_r_b)
  # print(r_t)
  # print(w_r_t)
  # input("hiya")

  l1 = line([r_b[0],r_b[1]], [w_r_b[0], w_r_b[1]])
  l2 = line([b_l[0],b_l[1]], [w_b_l[0], w_b_l[1]])
  R = intersection(l1, l2)
  xmax,ymax = R[0], R[1]
  # intesection_points.append(R)
  l1 = line([r_t[0],r_t[1]], [w_r_t[0], w_r_t[1]])
  l2 = line([l_t[0],l_t[1]], [w_l_t[0], w_l_t[1]])
  R = intersection(l1, l2)
  xmin,ymin = R[0], R[1]

  #cv2.circle(rotated_image,R,16,(0,255,0),-1)
  # cv2.line(rotated_image, (r_b[0],r_b[1]), (w_r_b[0], w_r_b[1]), (255,0,0),9)
  # cv2.line(rotated_image, (r_t[0],r_t[1]), (w_r_t[0], w_r_t[1]), (255,0,0),9)
  # cv2.line(rotated_image, (b_l[0],b_l[1]), (w_b_l[0], w_b_l[1]), (255,0,0),9)
  # cv2.line(rotated_image, (l_t[0],l_t[1]), (w_l_t[0], w_l_t[1]), (255,0,0),9)
  cv2.rectangle(rotated_image,(xmin,ymin), (xmax, ymax), (0,0,255), 4)
  image1 = rotated_image[ymin:ymax+1, xmin:xmax+1]
  print(rotated_image.shape)
  print(image1.shape)
  input("dimensions")
  return rotated_image

  

print ("-- GETTING INTERSECTION POINTS --- ")

intersection_t_t=get_point_of_intersection(equation_top,rotated_equation_top)
print ("point where the rotated top line intersects the top margin is : ",intersection_t_t)
#cv2.circle(rotated_image,intersection_t_t,16,(0,0,255),-1)
#
intersection_r_t=get_point_of_intersection(equation_right,rotated_equation_top)
print ("point where the rotated top line intersects the right margin is : ",intersection_r_t)
#cv2.circle(rotated_image,intersection_r_t,16,(0,0,255),-1)
# # 
intersection_r_r=get_point_of_intersection(equation_right,rotated_equation_right)
print ("point where the rotated right line intersects the right margin is : ",intersection_r_r)
#cv2.circle(rotated_image,intersection_r_r,16,(0,0,255),-1)
# #
intersection_r_b=get_point_of_intersection(equation_bottom,rotated_equation_right)
print ("point where the rotated right line intersects the bottom margin is : ",intersection_r_b)
#cv2.circle(rotated_image,intersection_r_b,16,(0,0,255),-1)
# #
intersection_b_b=get_point_of_intersection(equation_bottom,rotated_equation_bottom)
print ("point where the rotated bottom line intersects the bottom margin is : ",intersection_b_b)
#cv2.circle(rotated_image,intersection_b_b,16,(0,0,255),-1)
# #
intersection_b_l=get_point_of_intersection(equation_left,rotated_equation_bottom)
print ("point where the rotated bottom line intersects the left margin is : ",intersection_b_l)
#cv2.circle(rotated_image,intersection_b_l,16,(0,0,255),-1)
#
intersection_l_l=get_point_of_intersection(equation_left,rotated_equation_left)
print ("point where the rotated left line intersects the left margin is : ",intersection_l_l)
#cv2.circle(rotated_image,intersection_l_l,16,(0,0,255),-1)
#
intersection_l_t=get_point_of_intersection(equation_top,rotated_equation_left)
print ("point where the rotated left line intersects the top margin is : ",intersection_l_t)
#cv2.circle(rotated_image,intersection_l_t,16,(0,0,255),-1)
#

rotated_image =  get_intersection_crop(intersection_r_t, intersection_r_b,intersection_b_l,intersection_l_t, rotated_image)

cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
cv2.imshow('original', image)
cv2.imshow('rotated', rotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('rotated_positive.jpg',rotated_image)



