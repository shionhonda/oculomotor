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
        retina_image = inputs['from_retina'][0]
        is_white = inputs['from_retina'][1]
        if is_white:
            cnt = np.sum(np.sum(retina_image, axis=2)>255*3-1)
        else:
            cnt = np.sum(np.sum(retina_image, axis=2)<1)
            penalty = min(0, (cnt-10000)/500)
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