import cv2
import math
import numpy as np

import brica

class Retina(object):
    """ Retina module.
    This retina module takes environemnt image and outputs processed image with 
    peripheral vision.
    
    Peripheral pixels are blurred and gray-scaled.
    """
    
    def __init__(self):
        self.timing = brica.Timing(1, 1, 0)
        
        width = 128
        
        self.blur_rates, self.inv_blur_rates = self._create_rate_datas(width)
        self.gray_rates, self.inv_gray_rates = self._create_rate_datas(width, gain=0.5)

        self.last_retina_image = None

    def __call__(self, inputs):
        if 'from_environment' not in inputs:
            raise Exception('Retina did not recieve from Retina')
        
        image, angle = inputs['from_environment']
        retina_image = self._create_retina_image(image)
        is_white = self._is_white(image)

        # Store retina image for debug visualizer
        self.last_retina_image = retina_image

        return dict(to_lip=retina_image,
                    to_vc=(retina_image, is_white),
                    to_hp=(retina_image, angle))

    def _gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))

    def _create_rate_datas(self, width, sigma=0.32, clipping_gain=1.2, gain=1.0):
        """ Create mixing rate.
        Arguments:
            width: (int) width of the target image.
            sigma: (float) standard deviation of the gaussian.
            clipping_gain: (float) To make the top of the curve flat, apply gain > 1.0
            gain: (float) Final gain for the mixing rate. 
                          e.g.) if gain=0.8, mixing rates => 0.2~1.0
        Returns:
            Float ndarray (128, 128, 1): Mixing rates and inverted mixing rates. 
        """
        rates = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                rates[j*width + i] = self._gauss(r, sigma=sigma)
        rates = np.array(rates)
        # Normalize
        rates = rates / np.max(rates)
        
        # Make top flat by multipying and clipping 
        rates = rates * clipping_gain
        rates = np.clip(rates, 0.0, 1.0)

        # Apply final gain
        if gain != 1.0:
            rates = rates * gain + (1-gain)
        rates = rates.reshape([width, width, 1])
        inv_rates = 1.0 - rates
        return rates, inv_rates

    def _is_white(self, image):
        return np.sum(np.sum(image, axis=2)>764)>100

    def _create_attention(self, image):
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._is_white(image):
            _, im_cnt = cv2.threshold(im_gray, 245, 255, cv2.THRESH_BINARY)
        else:
            _, im_bin = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)
            im_inv = cv2.bitwise_not(im_bin) 
            kernel = np.ones((3,3),np.uint8)
            dilated = cv2.dilate(im_inv,kernel,iterations = 3)
            im_cnt = cv2.erode(dilated,kernel,iterations = 4)
            
        _, contours, _ = cv2.findContours(im_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx = np.array([[0,0], [0,127], [127,0], [127,127]]) # initialize
        for c in contours:
            if cv2.contourArea(c)<500:
                continue
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx)<4:
                continue
            else:
                break
        
        tmp = image.copy()
        if self._is_white(image):
            mask = cv2.fillConvexPoly(tmp, approx.reshape(-1,2), [255,255,255] )
            mask_gray= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(mask_gray, 245, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_bin)
            mask_inv = np.tile(mask_inv, (3,1,1)).transpose(1,2,0)
            attention = (image.astype('int32'))+(mask_inv.astype('int32'))
        else:
            mask = cv2.fillConvexPoly(tmp, approx.reshape(-1,2), [0,0,0] )
            mask_gray= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_bin)
            mask_inv = np.tile(mask_inv, (3,1,1)).transpose(1,2,0)
            attention = (image.astype('int32'))*(mask_inv.astype('int32'))/255        
        attention = np.clip(attention, 0, 255)
        return attention.astype('uint8')

    def _create_blur_image(self, image):
        h = image.shape[0]
        w = image.shape[1]

        # Resizeing to 1/2 size
        resized_image0 = cv2.resize(image,
                                  dsize=(h//2, w//2),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/4 size
        resized_image1 = cv2.resize(resized_image0,
                                  dsize=(h//4, w//4),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/8 size
        resized_image2 = cv2.resize(resized_image1,
                                  dsize=(h//8, w//8),
                                  interpolation=cv2.INTER_LINEAR)
        
        # Resizing to original size
        blur_image = cv2.resize(resized_image2,
                                dsize=(h, w),
                                interpolation=cv2.INTER_LINEAR)

        # Conver to Grayscale
        gray_blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        gray_blur_image = np.reshape(gray_blur_image,
                                     [gray_blur_image.shape[0],
                                      gray_blur_image.shape[0], 1])
        gray_blur_image = np.tile(gray_blur_image, 3)
        return blur_image, gray_blur_image

    def _create_retina_image(self, image):
        processed = self._create_attention(image)
        blur_image, gray_blur_image = self._create_blur_image(processed)
        # Mix original and blur image
        blur_mix_image = processed * self.blur_rates + blur_image * self.inv_blur_rates
        # Mix blur mixed image and gray blur image.
        gray_mix_image = blur_mix_image * self.gray_rates + gray_blur_image * self.inv_gray_rates
        return processed
#        return gray_mix_image.astype(np.uint8)