import cv2
import numpy as np
import math
import itertools
import brica
from .utils import Util


GAUSSIAN_KERNEL_SIZE = (5,5)


class OpticalFlow(object):
    def __init__(self):
        """ Calculating optical flow.
        Input image can be retina image or saliency map. 
        """
        self.last_gray_image = None
        self.hist_32 = np.zeros((128, 128), np.float32)
        
        self.inst = cv2.optflow.createOptFlow_DIS(
            cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.inst.setUseSpatialPropagation(False)
        self.flow = None
        
    def _warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
        
    def process(self, image, is_saliency_map=False):
        if image is None:
            return

        if not is_saliency_map:
            # Input is retina image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Input is saliency map
            gray_image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            
        if self.last_gray_image is not None:
            if self.flow is not None:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           self._warp_flow(self.flow, self.flow))
            else:
                self.flow = self.inst.calc(self.last_gray_image,
                                           gray_image,
                                           None)
            # (128, 128, 2)
        self.last_gray_image = gray_image
        return self.flow

def normalize_range(src, begin=0, end=255):
    dst = np.zeros((len(src), len(src[0])))
    amin, amax = np.amin(src), np.amax(src)
    for y, x in itertools.product(range(len(src[1])), range(len(src[0]))):
        if amin != amax:
            dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
        else:
            dst[y][x] = (end + begin) / 2

    if end == 255:
        dst = np.uint8(dst)
    return dst


def normalize(src):
    src = normalize_range(src, 0., 1.)
    amax = np.amax(src)
    maxs = []

    for y in range(1, len(src[1]) - 1):
        for x in range(1, len(src[0]) - 1):
            val = src[y][x]
            if val == amax:
                continue
            is_max = val > src[y - 1][x] and val > src[y + 1][x] and val > \
                src[y][x - 1] and val > src[y][x + 1]
            if is_max:
                maxs.append(val)

    if len(maxs) != 0:
        src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)

    return src


class GaussianPyramid:
    def __init__(self, src):
        self.maps = self.__make_gaussian_pyramid(src)

    def __make_gaussian_pyramid(self, src):
        # gaussian pyramid | 0 ~ 8(1/256) . not use 0 and 1.
        maps = {'intensity': [],
                'colors': {'b': [],
                           'g': [],
                           'r': [],
                           'y': []},
                'orientations': {'0': [],
                                 '45': [],
                                 '90': [],
                                 '135': []}}
        amax = np.amax(src)
        # RGBにそれぞれチャンネルを分割
        b, g, r = cv2.split(src)
        
        for x in range(1, 9):
            # 画像ピラミッドの作成
            b, g, r = list(map(cv2.pyrDown, [b, g, r]))
            if x < 2:
                # 1は利用しない
                continue
            buf_its = np.zeros(b.shape)
            buf_colors = list(map(lambda _: np.zeros(b.shape), range(4)))  # b, g, r, y
            
            for y, x in itertools.product(range(len(b)), range(len(b[0]))):
                buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])
                buf_colors[0][y][x], buf_colors[1][y][x], buf_colors[2][y][x], buf_colors[3][y][x] = self.__get_bgry_colors(b[y][x], g[y][x], r[y][x], buf_its[y][x], amax)
                
            maps['intensity'].append(buf_its)
            for (color, index) in zip(sorted(maps['colors'].keys()), range(4)):
                maps['colors'][color].append(buf_colors[index])
            for (orientation, index) in zip(sorted(maps['orientations'].keys()), range(4)):
                # ガボールフィルタをかける
                maps['orientations'][orientation].append(self.__conv_gabor(buf_its,
                                                                           np.pi * index / 4))
        return maps

    def __get_intensity(self, b, g, r):
        # rgbからintensityを計算
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

    def __get_bgry_colors(self, b, g, r, i, amax):
        # 
        b, g, r = list(map(lambda x: np.float64(x) if (x > 0.1 * amax) else 0., [b, g, r]))
        nb, ng, nr = list(map(lambda x, y, z: max(x - (y + z) / 2., 0.),
                              [b, g, r],
                              [r, r, g],
                              [g, b, b]))
        ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)

        if i != 0.0:
            return list(map(lambda x: x / np.float64(i), [nb, ng, nr, ny]))
        else:
            return nb, ng, nr, ny

    def __conv_gabor(self, src, theta):
        kernel = cv2.getGaborKernel((3, 3), 1, theta, 3, 1)
        return cv2.filter2D(src, -1, kernel)


class FeatureMap:
    def __init__(self, srcs):
        self.maps = self.__make_feature_map(srcs)

    def __make_feature_map(self, srcs):
        # scale index for center-surround calculation | (center, surround)
        # index of 0 ~ 6 is meaned 2 ~ 8 in thesis (Ich)
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6))
        maps = {
            'intensity': [],
            'colors': {'bg': [],
                       'ry': []},
            'orientations': {'0': [],
                             '45': [],
                             '90': [],
                             '135': []}}

        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(srcs['intensity'][c],
                                                       srcs['intensity'][s]))
            for key in maps['orientations'].keys():
                maps['orientations'][key].append(self.__scale_diff(srcs['orientations'][key][c],
                                                                   srcs['orientations'][key][s]))
            for key in maps['colors'].keys():
                maps['colors'][key].append(self.__scale_color_diff(
                    srcs['colors'][key[0]][c], srcs['colors'][key[0]][s],
                    srcs['colors'][key[1]][c], srcs['colors'][key[1]][s]
                ))
        return maps

    def __scale_diff(self, c, s):
        c_size = tuple(reversed(c.shape))
        return cv2.absdiff(c, cv2.resize(s, c_size, None, 0, 0, cv2.INTER_NEAREST))

    def __scale_color_diff(self, c1, s1, c2, s2):
        c_size = tuple(reversed(c1.shape))
        return cv2.absdiff(c1 - c2, cv2.resize(s2 - s1, c_size, None, 0, 0, cv2.INTER_NEAREST))


class ConspicuityMap:
    def __init__(self, srcs):
        self.maps = self.__make_conspicuity_map(srcs)

    def __make_conspicuity_map(self, srcs):
        normalized_intensity = list(map(normalize, srcs['intensity']))
        intensity = self.__scale_add(normalized_intensity)
        
        for key in srcs['colors'].keys():
            srcs['colors'][key] = list(map(normalize, srcs['colors'][key]))
            
        color = self.__scale_add([srcs['colors']['bg'][x] + \
                                  srcs['colors']['ry'][x] \
                                  for x in range(len(srcs['colors']['bg']))])
        
        orientation = np.zeros(intensity.shape)
        for key in srcs['orientations'].keys():
            orientation += self.__scale_add(list(map(normalize, srcs['orientations'][key])))
        return {'intensity': intensity,
                'color': color,
                'orientation': orientation}

    def __scale_add(self, srcs):
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv2.resize(x, tuple(reversed(buf.shape)))
        return buf


class SaliencyMap:
    def __init__(self, src):
        self.gp = GaussianPyramid(src)
        self.fm = FeatureMap(self.gp.maps)
        self.cm = ConspicuityMap(self.fm.maps)
        self.map = cv2.resize(self.__make_saliency_map(self.cm.maps),
                              tuple(reversed(src.shape[0:2])))

    def __make_saliency_map(self, srcs):
        srcs = list(map(normalize, [srcs[key] for key in srcs.keys()]))
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.

class LIP(object):
    """ Retina module.
    This LIP module calculates saliency cv2. and optical flow from retina image.
    """
    
    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)

        self.optical_flow = OpticalFlow()

        self.last_saliency_map = None
        self.last_optical_flow = None

    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('LIP did not recieve from Retina')

        retina_image = inputs['from_retina'] # (128, 128, 3)
        saliency_map = SaliencyMap(retina_image).map # (128, 128)
        saliency_map /= np.max(saliency_map)
        #saliency_map = self._get_saliency_map(retina_image)
        use_saliency_flow = False

        if not use_saliency_flow:
            # Calculate optical flow with retina image
            optical_flow = self.optical_flow.process(retina_image,
                                                     is_saliency_map=False)
        else:
            # Calculate optical flow with saliency map
            optical_flow = self.optical_flow.process(saliency_map,
                                                     is_saliency_map=True)
        
        # Store saliency map for debug visualizer
        self.last_saliency_map = saliency_map
        
        self.last_optical_flow = optical_flow
        
        return dict(to_fef=(saliency_map, optical_flow))

    def _get_saliency_magnitude(self, image):
        # Calculate FFT
        dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

        log_magnitude = np.log10(magnitude.clip(min=1e-10))

        # Apply box filter
        log_magnitude_filtered = cv2.blur(log_magnitude, ksize=(3, 3))

        # Calculate residual
        magnitude_residual = np.exp(log_magnitude - log_magnitude_filtered)

        # Apply residual magnitude back to frequency domain
        dft[:, :, 0], dft[:, :, 1] = cv2.polarToCart(magnitude_residual, angle)
    
        # Calculate Inverse FFT
        image_processed = cv2.idft(dft)
        magnitude, _ = cv2.cartToPolar(image_processed[:, :, 0],
                                       image_processed[:, :, 1])
        return magnitude

    def _get_saliency_map(self, image):
        resize_shape = (64, 64) # (h,w)

        # Size argument of resize() is (w,h) while image shape is (h,w,c)
        image_resized = cv2.resize(image, resize_shape[1::-1])
        # (64,64,3)

        saliency = np.zeros_like(image_resized, dtype=np.float32)
        # (64,64,3)
    
        channel_size = image_resized.shape[2]
    
        for ch in range(channel_size):
            ch_image = image_resized[:, :, ch]
            saliency[:, :, ch] = self._get_saliency_magnitude(ch_image)

        # Calclate max over channels
        saliency = np.max(saliency, axis=2)
        # (64,64)

        saliency = cv2.GaussianBlur(saliency, GAUSSIAN_KERNEL_SIZE, sigmaX=8, sigmaY=0)

        SALIENCY_ENHANCE_COEFF = 4 # Strong saliency contrast
        #SALIENCY_ENHANCE_COEFF = 1 # Low saliency contrast, but sensible for weak saliency

        # Emphasize saliency
        saliency = (saliency ** SALIENCY_ENHANCE_COEFF)

        # Normalize to 0.0~1.0
        saliency = (saliency-np.min(saliency)) / (np.max(saliency)-np.min(saliency))
    
        # Resize to original size
        saliency = cv2.resize(saliency, image.shape[1::-1])
        return saliency