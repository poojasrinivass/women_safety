from ctypes import *
import math
import random
import cv2
import os
import numpy as np
import dlib
import scipy
import numpy as np
import scipy.misc
import sys
import os
import threading
import time
from scipy import spatial
import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = './trained_models/gender_models/simple_CNN.81-0.96.hdf5'

gender_labels = get_labels('imdb')

gender_offsets = (30, 60)
gender_offsets = (10, 10)

gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]


font = cv2.FONT_HERSHEY_SIMPLEX

def sample(probs):
	s = sum(probs)
	probs = [a/s for a in probs]
	r = random.uniform(0, 1)
	for i in range(len(probs)):
		r = r - probs[i]
		if r <= 0:
			return i
	return len(probs)-1

def c_array(ctype, values):
	arr = (ctype*len(values))()
	arr[:] = values
	return arr

class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float)]

class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int)]


class IMAGE(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_float))]

class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p))]

	

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
	out = predict_image(net, im)
	res = []
	for i in range(meta.classes):
		res.append((meta.names[i], out[i]))
	res = sorted(res, key=lambda x: -x[1])
	return res

def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
	im = load_image(image, 0, 0)
	num = c_int(0)
	pnum = pointer(num)
	predict_image(net, im)
	dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
	num = pnum[0]
	if (nms): do_nms_obj(dets, num, meta.classes, nms);

	res = []
	for j in range(num):
		for i in range(meta.classes):
			if dets[j].prob[i] > 0:
				b = dets[j].bbox
				res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
	res = sorted(res, key=lambda x: -x[1])
	free_image(im)
	free_detections(dets, num)
	return res
	
if __name__ == "__main__":
	#net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
	#im = load_image("data/wolf.jpg", 0, 0)
	#meta = load_meta("cfg/imagenet1k.data")
	#r = classify(net, meta, im)
	#print r[:10]
	predictor_path = '/home/iniesta/Desktop/Demo_face/shape_predictor_68_face_landmarks.dat'
	face_rec_model_path = '/home/iniesta/Desktop/Demo_face/dlib_face_recognition_resnet_model_v1.dat'
	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)
	facerec = dlib.face_recognition_model_v1(face_rec_model_path)
	face_conf={}
	face_gender={}
	faceTrackers = {}
	faceNames = {}
	face_des={}
	face_sim={}
	face_vis={}
	currentFaceID = 0
	car_count=0
	bike_count=0
	other_count=0
	net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
	meta = load_meta("cfg/coco.data")
	test_dir="data/data_check"
	first_flg=1


	file_count=len(os.listdir(test_dir))
	people_count=0
	for impt in range(file_count):
			# frame_name='/home/iniesta/Downloads/demo_face/demo'+str(i+1)+'.jpg'
			temp_people_count=0
			f=str(impt+36)+".jpg"
			img=scipy.misc.imread("./data/data_check/"+f, mode="RGB")
			dets, scores, idx = detector.run(img, 3, 0)

			r = detect(net, meta, "data/data_check/"+f)
			img = cv2.imread("data/data_check/"+f,3) 
			baseImage=img   
			fidskeep = []
			for i in r:

					name_veh=i[0]
					col = list(np.random.choice(range(256), size=3))
					x,y,w,h  = int(i[2][0]),int(i[2][1]),int(i[2][2]),int(i[2][3])  

					if(i[0]=="person"):
						temp_people_count=temp_people_count+1

						x_bar = x 
						y_bar = y 

						matchedFid = None
						for fid in faceTrackers.keys():
						
							tracked_position =  faceTrackers[fid].get_position()

							t_x = int(tracked_position.left())
							t_y = int(tracked_position.top())
							t_w = int(tracked_position.width())
							t_h = int(tracked_position.height())
							t_x_bar = t_x + (t_w)/2
							t_y_bar = t_y + (t_h)/2

							var1=100
							if(face_des[fid]==str(name_veh)):
								if (( t_x -var1<= x_bar   <= t_x+t_w+var1) and 
							 ( t_y -var1<= y_bar   <= (t_y+t_h +var1)) and 
							   ((x-(w//2))-var1  <= t_x_bar <= (  (x+(w//2)) +var1 )) and 
							 ( y-(h//2)-var1  <= t_y_bar <= ( y+(h//2)+var1 ))):
									matchedFid = fid
									fidskeep.append(matchedFid)
						if(matchedFid):
							tracker = dlib.correlation_tracker()
							tracker.start_track(baseImage,
												dlib.rectangle(  (x-(w//2)),
																	y-(h//2),
																	(x+(w//2)),
																	y+(h//2)))
							faceTrackers[matchedFid] = tracker
						if matchedFid is None:
								tracker = dlib.correlation_tracker()
								tracker.start_track(baseImage,
													dlib.rectangle(  (x-(w//2)),
																	y-(h//2),
																	(x+(w//2)),
																	y+(h//2)))
								fidskeep.append(currentFaceID)
								faceTrackers[currentFaceID] = tracker
								tracked_position =  faceTrackers[currentFaceID].get_position()
								x = int(tracked_position.left())
								y = int(tracked_position.top())
								w = int(tracked_position.width())
								h = int(tracked_position.height())
								face_des[currentFaceID] = str(name_veh)
								face_vis[currentFaceID]=0
								face_gender[currentFaceID]=-1
								face_conf[currentFaceID]=-1
								currentFaceID += 1

						cv2.rectangle(img, (x-(w//2),y-(h//2)),(x+(w//2),y+(h//2)), (0,255,0), 1)
						cv2.putText(img,i[0], (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.7,(0,0,0))
			if(temp_people_count< people_count-2):
				temp_people_count=people_count
			
			people_count=temp_people_count

			num1=faceTrackers.keys()
			num2=fidskeep
			del_list= list(set(num1).difference(set(num2)))
			for fid in del_list:
					if(face_des[fid]==str("person")):
						if(x < 150 or x > 960-150 or y< 150 or y> 720-150 ):
							car_count=car_count+1
							faceTrackers.pop(fid , None)
			men_count=0
			women_count=0
			for var_i, d in enumerate(dets):
				for fid in faceTrackers.keys():
					face_gender[fid]=-1
					tracked_position =  faceTrackers[fid].get_position()
					x = int(tracked_position.left())
					y = int(tracked_position.top())
					w = int(tracked_position.width())
					h = int(tracked_position.height())
					
				
					
					shape = sp(img, d)
					face_descriptor = facerec.compute_face_descriptor(img, shape)	
					face_x = int(d.left())
					face_y = int(d.top())
					face_w = int(d.right()-d.left())
					face_h = int(d.bottom()-d.top())
					crop_image=img[face_y:face_y+face_h,face_x:face_x+face_w]
					#calculate the centerpoint
						
					face_x_bar = face_x + 0.5 * face_w
					face_y_bar = face_y + 0.5 * face_h

				
					if(face_x_bar > x and face_x_bar < x+w and face_y_bar> y and face_y_bar < y+h):


						rgb_face = cv2.resize(crop_image, (gender_target_size))
		
						rgb_face = np.expand_dims(rgb_face, 0)
						gender_prediction = gender_classifier.predict(rgb_face)
						gender_label_arg = np.argmax(gender_prediction)
						gender_text = gender_labels[gender_label_arg]
						if(gender_text=="woman"):
							temp_p=0
						else:
							temp_p=1
						face_gender[fid]=temp_p

						# if(face_gender[fid]==-1 ) :
						#  #face_conf[fid]<scores[var_i]
						# 	# face_conf[fid]=scores[var_i]
						# 	face_gender[fid]=temp_p
						
						if(face_gender[fid]==0):
							gender_text="woman"

						else:
							gender_text="man"
						if(gender_text=="woman"):
							women_count+=1
						else:
							men_count+=1		

						cv2.putText(img,gender_text, (face_x, face_y), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,0))
						cv2.rectangle(img,(face_x,face_y),(face_x+face_w,face_y+face_h), (0,255,0), 1)
						break


			unvalidated_people=people_count-men_count-women_count
			if(women_count!=0):
				gender_ratio_known=men_count/women_count
			else:
				gender_ratio_known=0
			cv2.putText(img,str("people_count")+"  "+str(people_count), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,0,0))
			cv2.putText(img,str("women_count")+"  "+str(women_count), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,0,0))
			cv2.putText(img,str("men_count")+"  "+str(men_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,0.0))
			cv2.putText(img,str("unvalidated_people")+"  "+str(unvalidated_people), (20, 80), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0))
			cv2.putText(img,str("gender_ratio_known")+"  "+str(gender_ratio_known), (20, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,0,0))
			cv2.imwrite('camera/'+f,img)
			

