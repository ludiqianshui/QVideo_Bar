#!/usr/local/bin/python

import cv2
import unittest
import numpy as np
from numpy import binary_repr
# Load an color image in grayscale
from cv2 import *

class extract_img_open(object):

    def __init__(self):
        

        return
    def __del__(self):
        
        return
    
    def OverlayImage(self,src, overlay, posx, posy, S, D):
    
        for x in range(overlay.width):
            if x+posx < src.width:
                for y in range(overlay.height):
                    if y+posy < src.width:
                        source = cv.Get2D(src, y+posy, x+posx)
                        over   = cv.Get2D(overlay, y, x)
                        merger = [0, 0, 0, 0]
    
                        for i in range(3):
                            merger[i] = (S[i]*source[i]+D[i]*over[i])
                        merged = tuple(merger)
                        cv.Set2D(src, y+posy, x+posx, merged)
        return
    
    
    def extract_mp4(self, video_file, img_folder):
        cap = cv2.VideoCapture(video_file)
#        if cap.isOpened():
        if 1:
            self.rtn = 1
        i = 0
        while self.rtn:
            self.rtn, self.frame = cap.read()
            cv2.imwrite(img_folder+"frame%d.jpg" % i, self.frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()    
        return
    
    def build_barcode(self, frame_number, barcode_folder):
        self.img = np.zeros((10,100,3), np.uint8)
        frame_number_decimal = int(frame_number)
        frame_number_binary =  binary_repr(frame_number_decimal, width=10)
        print(frame_number_binary)
        digit_step = 0
        for frame_digit in frame_number_binary:
            if frame_digit == '1':
                cv2.rectangle(self.img,(0+digit_step*10 ,0),(digit_step*10+10,10),(255,255,255),-1)
            digit_step += 1
        cv2.imwrite(barcode_folder+str(frame_number)+'.jpg',self.img)
        return

    def overlay_imgs(self, barcode_folder, img_folder, output_folder):
        overlay = cv2.imread('bar_code/86.png')
        src = cv2.imread('frame67.jpg')
        S = (0.5, 0.5, 0.5, 0.5)   
        D = (0.5, 0.5, 0.5, 0.5)            
        posx = 200                  
        posy = 200                   
        self.OverlayImage(src, overlay, posx, posy, S, D)
        return
        
    def build_imgs(self, video_file, img_folder):
        cap = cv2.VideoCapture(video_file)
#        if cap.isOpened():
        if 1:
            self.rtn = 1
        frame_index = 0
        while self.rtn:
            self.rtn, self.frame = cap.read()
            cv2.rectangle(self.frame,(100,100),(300,100+10),(0,0,0),-1)
            frame_number_binary =  binary_repr(frame_index, width=20)
            digit_step = 0
            for frame_digit in frame_number_binary:
                if frame_digit == '1':
                    cv2.rectangle(self.frame,(100+digit_step*10 ,100),(100+digit_step*10+10,100+10),(255,255,255),-1)
                digit_step += 1
        
            cv2.imwrite(img_folder+"frame%d.jpg" % frame_index, self.frame)
            
            frame_index += 1
        cap.release()
        cv2.destroyAllWindows()    
        return
        
    def build_video_with_imgs(self):
        
        video=cv2.VideoWriter('video/video_with_bar.avi',
                              fourcc=cv.CV_FOURCC('M','P', '4', '2'),
                              30,
                              (1280,720))
        for img in range (0, 131):  
            video.write("output/frame"+str(img)+".jpg")
        
        cv2.destroyAllWindows()
        video.release()
        
class Test(unittest.TestCase):

    def test_username(self):
        mp4_extract = extract_img_open()
#        result = mp4_extract.build_imgs("SampleVideo.mp4", "output/")
        mp4_extract.build_video_with_imgs()
#        for digi in range (0,190):
#            mp4_extract.build_barcode(digi, "bar_code/")
#        mp4_extract.overlay_imgs("",'','')
#        print (result)
        return
