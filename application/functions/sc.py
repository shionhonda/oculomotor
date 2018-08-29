import numpy as np
import brica


class SC(object):
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

        self.last_fef_data = None
        self.last_sc_data = None

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')
        
        fef_data = np.array(inputs['from_fef'])
        bg_data =inputs['from_bg']

        action, self.last_sc_data = self._decide_action(fef_data, bg_data)
        
        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data
        
        return dict(to_environment=action)

    def _decide_action(self, fef_data, bg_data):
        '''
          Function: Choose the direction with maximum likelihood subtracted by threshold
          Inputs: 
            bg_data: 0-63 are saliency thresholds; 64-127 are cursor thresholds; 128 is lambda
        '''

        diff = fef_data[:,0]-bg_data[:-1]
        lamda = bg_data[-1]
        print("IIIIIII")
        last_sc_data = lamda*diff[:64] + (1-lamda)*diff[64:]
        max_idx = np.argmax(last_sc_data)
        action = 0.01 * fef_data[max_idx, 1:]
        return action, last_sc_data
