# -*- coding: utf-8 -*-
import unittest
import os
from skimage import io
import numpy as np
import time
import glob
import cv2


class TestMask(unittest.TestCase):
    def setUp(self):
        pass
    
    def get_file_path(self, relative_path):
        abs_path_module = os.path.realpath(__file__)
        module_dir, _ = os.path.split(abs_path_module)
        file_path = os.path.join(module_dir, relative_path)
        return file_path

    def load_image(self, path):
        file_path = self.get_file_path(path)
        image =  cv2.imread(file_path)
        return image

    def save_image(self, image, path):
        file_path = self.get_file_path(path)
        io.imsave(file_path, image)

    def concat_images(self, images):
        spacer = np.ones([128, 1, 3], dtype=np.uint8)
        images_with_spacers = []

        image_size = len(images)
  
        for i in range(image_size):
            images_with_spacers.append(images[i])
            if i != image_size-1:
                # 1ピクセルのスペースを空ける
                images_with_spacers.append(spacer)
        ret = np.hstack(images_with_spacers)
        return ret

    def is_white(self, path):
        if path[0]=='1' or path[0]=='2' or path[0]=='4' or path[0]=='5':
            return True
        if path[0]=='3' or path[0]=='6':
            return False
        else:
            raise ValueError("Undefined task number")


    def sub_test_attention(self, file_path):
        im_orig = self.load_image(file_path)
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        im_gray = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
        if self.is_white(file_base_name):
            _, im_cnt = cv2.threshold(im_gray, 245, 255, cv2.THRESH_BINARY)
        else:
            _, im_bin = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)
            im_inv = cv2.bitwise_not(im_bin) 
            kernel = np.ones((3,3),np.uint8)
            dilated = cv2.dilate(im_inv,kernel,iterations = 3)
            im_cnt = cv2.erode(dilated,kernel,iterations = 4)
            
        _, contours, _ = cv2.findContours(im_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im_color = im_orig.copy()
        for c in contours:
            if cv2.contourArea(c)<500:
                continue
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx)<4:
                continue
            cv2.drawContours(im_color, c, -1, (0, 0, 255), 3)
            cv2.drawContours(im_color, [approx], -1, (0, 255, 0), 3)
            break
        
        tmp = im_orig.copy()
        if self.is_white(file_base_name):
            mask = cv2.fillConvexPoly(tmp, approx.reshape(-1,2), [255,255,255] )
            mask_gray= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(mask_gray, 245, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_bin)
            mask_inv = np.tile(mask_inv, (3,1,1)).transpose(1,2,0)
            attention = (im_orig.astype('int32'))+(mask_inv.astype('int32'))
        else:
            mask = cv2.fillConvexPoly(tmp, approx.reshape(-1,2), [0,0,0] )
            mask_gray= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_bin)
            mask_inv = np.tile(mask_inv, (3,1,1)).transpose(1,2,0)
            attention = (im_orig.astype('int32'))*(mask_inv.astype('int32'))/255        
        attention = np.clip(attention, 0, 255)
        attention = attention.astype('uint8')

        #img = self.concat_images([im_orig, im_color, attention])
        self.save_image(attention, "test_results/out_{}.png".format(file_base_name))

    def test_attention(self):
        if not os.path.exists(self.get_file_path("test_results")):
            os.mkdir(self.get_file_path("test_results"))
        
        images_dir_path = self.get_file_path("images/task_images")
        
        file_path_list = glob.glob("{}/*.png".format(images_dir_path))
        for file_path in file_path_list:
            self.sub_test_attention(file_path)
            

if __name__ == '__main__':
    unittest.main()