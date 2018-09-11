import brica
import numpy as np
import cv2

class VC(object):
    """ Visual Cortex module.
    
    You can add feature extraction code as like if needed.
    """
    
    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)
        self.thres = 0.9

    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('VC did not recieve from Retina')
        penalty = 0
        retina_image = inputs['from_retina']
        retina_hsv = retina_image.copy()
        retina_hsv = cv2.cvtColor(retina_hsv.astype(np.uint8), cv2.COLOR_RGB2HSV)
        lightblue = np.array([115,209,255], np.uint8)
        lightblue =  np.tile(lightblue,(128,128,1))
        lightblue = cv2.cvtColor(lightblue, cv2.COLOR_RGB2HSV)
        cos_sim = _cosine_similarity(lightblue[32:96,32:96,:], retina_hsv[32:96,32:96,:])

        penalty = min(0, (self.thres-cos_sim)*20)
        #print(penalty)
        

        # Current implementation just passes through input retina image to FEF and PFC.
        
        return dict(to_fef=retina_image,
                    to_pfc=retina_image,
                    to_bg=penalty)

def _cosine_similarity(x, y):
    X = x+0.1 # Avoid zero devision
    Y = y+0.1
    dot = np.sum(X*Y, axis=2)
    X_norm = np.linalg.norm(X, axis=2)
    Y_norm = np.linalg.norm(Y, axis=2)
    return np.mean(dot/X_norm/Y_norm)