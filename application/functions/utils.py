import os
import cv2
import sys
import math
import itertools
import numpy as np

def load_image(file_path):
    module_dir, _ = os.path.split(os.path.realpath(__file__))
    absolute_path = os.path.join(module_dir, file_path)
    image = cv2.imread(absolute_path)
    # (h, w, c), uint8
    # Change BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image, file_path):
    module_dir, _ = os.path.split(os.path.realpath(__file__))
    absolute_path = os.path.join(module_dir + "/../..", file_path)

    # Change RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(absolute_path, image)

class OpencvIo:
    def __init__(self):
        self.__util = Util()

    def imread(self, path, option=1):
        try:
            if not os.path.isfile(os.path.join(os.getcwd(), path)):
                raise IOError('File is not exist')
            src = cv2.imread(path, option)
        except IOError:
            raise
        except:
            print('Arugment Error : Something wrong')
            sys.exit()
        return src

    def imshow(self, src, name='a image'):
        cv2.imshow(name, src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imshow_array(self, images):
        name = 0
        for x in images:
            cv2.imshow(str(name), np.uint8(self.__util.normalize_range(x)))
            name = name + 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Util:
    def normalize_range(self, src, begin=0, end=255):
        dst = np.zeros((len(src), len(src[0])))
        amin, amax = np.amin(src), np.amax(src)
        for y, x in itertools.product(range(len(src)), range(len(src[0]))):
            if amin != amax:
                dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
            else:
                dst[y][x] = (end + begin) / 2
        return dst

    def normalize(self, src):
        src = self.normalize_range(src, 0., 1.)
        amax = np.amax(src)
        maxs = []

        for y in range(1, len(src) - 1):
            for x in range(1, len(src[0]) - 1):
                val = src[y][x]
                if val == amax:
                    continue
                if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
                    maxs.append(val)

        if len(maxs) != 0:
            src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)

        return src