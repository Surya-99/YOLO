# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:07:57 2023

@author: HP
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#print(len(classes))
#print(classes)
def res(img):
      #  img=cv2.imread(img1)
        yolo = cv2.dnn.readNet("yolov3-spp.weights","weights1.cfg")
        classes = open("./coco_classes.txt", "r").read().splitlines()
        blob=cv2.dnn.blobFromImage(img,1/255,(320,320),swapRB=True,crop=False)
        j=blob[0].reshape(320,320,3)
        yolo.setInput(blob)
        output_layes_name=yolo.getUnconnectedOutLayersNames()
        layeroutput=yolo.forward(output_layes_name)
        height,width,_ =img.shape
        boxes=[]
        confidences=[]
        class_ids=[]
        for output in layeroutput:
            for detection in output:
                score=detection[5:] #first 4 boxes represents bx,by,bh,bw remaining boxes gives the probability that a
                             #particular object is present in the box
                class_id=np.argmax(score)
        #argmax returns index of max element in array. In this case max probability
                confidence=score[class_id]
        #confidence is the max probability
                if confidence>0.5:
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)
                    x=int(center_x-w/2)        #x and y values are for finding corners of boxes
                    y=int(center_y-h/2)
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        font=cv2.FONT_HERSHEY_PLAIN #Adding font
        colors=np.random.uniform(0,255,size=(len(classes),3))
        dic={}
        print("Labels and Confidences:")
        if len(indexes) > 0:
            for i in indexes.flatten():
                x,y,w,h=boxes[i]
                label=str(classes[class_ids[i]])  #For labeling objects
                confi=str(round(confidences[i],2)) #Confidences
                color=colors[i] 
                cv2.rectangle(img,(x,y),(x+w,y+h),color,3) #rectangular boxes
                cv2.putText(img,label+" "+confi,(x,y+20),font,2,(255,255,255),1)
                dic[label]=confi
        return img,dic