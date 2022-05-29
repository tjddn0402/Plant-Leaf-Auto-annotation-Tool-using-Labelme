import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from morph import *

def histogram_hue(img_path):
    """
    한 image file의 hue histogram을 plot하는 함수
    """
    img = cv2.imread(img_path)
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist(hsi,[0],None,[180],[0,180])
    return hue_hist.astype(np.uint64)

def histogram_intensity(img_path):
    """
    한 image file의 intensity histogram을 plot하는 함수
    """
    img = cv2.imread(img_path)
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    intensity_hist = cv2.calcHist(hsi,[2],None,[256],[0,256])
    return intensity_hist.astype(np.uint64)

def AND(img,mask):
    H, W, C = img.shape
    masked = img.copy()
    for h in range(H):
        for w in range(W):
            if mask[h][w]==0:
                masked[h][w]=0
    return masked


def masking(path,low=30,high=67,plot_hue=True,plot_intensity=True):
    # path 내부의 모든 이미지 segmentation, masking
    # https://codechacha.com/ko/python-list-all-files-in-dir/
    files = os.listdir(path)

    if plot_hue:
        hue_hist = np.zeros((180,1),dtype=np.uint64)
        for file in files:
            file_path = path +'/'+ file
            hue_hist += histogram_hue(file_path)
        plt.plot(hue_hist),plt.show()

    if plot_intensity:
        intensity_hist = np.zeros((256,1),dtype=np.uint64)
        for file in files:
            file_path = path +'/'+ file
            intensity_hist += histogram_intensity(file_path)
        plt.plot(intensity_hist),plt.show()

    if not os.path.exists(path+'_mask'):
        os.mkdir(path+'_mask')
    if not os.path.exists(path+'_masked'): # 제대로 마스킹 되었는지 확인
        os.mkdir(path+'_masked')
        
    for file in files:
        img_path = path + '/' + file
        img = cv2.imread(img_path)

        hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,i = cv2.split(hsi)
        mask = np.where((low < h)&(h < high), 255, 0).astype(np.uint8)
        cv2.imwrite(path+'_mask/'+file, mask)

        masked = AND(img,mask)
        merged = cv2.hconcat([img,masked])
        cv2.imwrite(path+'_masked/'+file, merged)

if __name__ == "__main__":
    # test case
    masking('./BIBICHU',plot_hist=False)