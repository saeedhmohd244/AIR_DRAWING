import os
import cv2
import sys
import numpy as np
from scipy import ndimage


class Segmenting:
    def openImage(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def find_connected(img, label, id, i, j, threshold, xmin, ymin, xmax, ymax):
        h, w = img.shape
        if img[i, j] < threshold or i >= h - 1 or j >= w - 1:
            return label, xmin, ymin, xmax, ymax
        else:
            if label[i, j] == 0:
                if j < xmin:
                    xmin = j
                if j > xmax:
                    xmax = j
                if i < ymin:
                    ymin = i
                if i > ymax:
                    ymax = i

                label[i, j] = id

                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i + 1, j, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i + 1, j + 1, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i, j + 1, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i - 1, j + 1, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i - 1, j, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i - 1, j - 1, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i, j - 1, threshold, xmin, ymin, xmax, ymax)
                label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i + 1, j - 1, threshold, xmin, ymin, xmax, ymax)
        return label, xmin, ymin, xmax, ymax

    def bounding():
        sys.setrecursionlimit(250000000)
        img_path = 'new.jpg'
        img = Segmenting.openImage(img_path)
        print('--> Input Image')

        ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
        print('--> Inverse Binary Image')

        img = ndimage.median_filter(img, 5)
        print('--> Applying Median Filter')
    
        height, width = img.shape
        threshold = 127
        id = 0
        label = np.zeros((height, width))
        characters_folder = "characters/0"
        if not os.path.exists(characters_folder):
            os.makedirs(characters_folder)

        for i in range(height):
            for j in range(width):
                if img[i, j] >= threshold and label[i, j] == 0:
                    id = id + 1
                    label, xmin, ymin, xmax, ymax = Segmenting.find_connected(img, label, id, i, j, threshold, width, height, -1, -1)
                    crop_img = img[ymin:ymax, xmin:xmax]

                    cv2.imwrite(f"{characters_folder}/character_{id}.png", crop_img)

                    img[ymin - 2:ymin - 1, xmin - 1:xmax + 1] = 100
                    img[ymax + 1:ymax + 2, xmin - 1:xmax + 1] = 100
                    img[ymin - 1:ymax + 1, xmin - 2:xmin - 1] = 100
                    img[ymin - 1:ymax + 1, xmax + 1:xmax + 2] = 100

        print('--> segmented images: ')
        
