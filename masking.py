import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def calcHist_hsi(img_path):
    img = cv2.imread(img_path)
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist(hsi,[0],None,[180],[0,180])
    saturation_hist = cv2.calcHist(hsi,[1],None,[256],[0,256])
    intensity_hist = cv2.calcHist(hsi,[2],None,[256],[0,256])
    return [hue_hist,saturation_hist,intensity_hist]

def histogram_hsi(dir_path):
    """
    directory 내부의 h,s,i plot을 모두 더하여 한꺼번에 본다.
    """
    files = os.listdir(dir_path)
    hue_hist = np.zeros((180,1),dtype=np.uint64)
    saturation_hist = np.zeros((180,1),dtype=np.uint64)
    intensity_hist = np.zeros((180,1),dtype=np.uint64)

    for file in files:
        file_path = dir_path +'/'+ file
        hsi_hist = calcHist_hsi(file_path)
        hue_hist += hsi_hist[0]
        saturation_hist += hsi_hist[0]
        intensity_hist += hsi_hist[0]
        
    plt.plot(hue_hist),plt.show()
    plt.plot(saturation_hist),plt.show()
    plt.plot(intensity_hist),plt.show()




def AND(img,mask):
    H, W, C = img.shape
    masked = img.copy()
    for h in range(H):
        for w in range(W):
            if mask[h][w]==0:
                masked[h][w]=0
    return masked




def masking(path,low=30,high=67,save_merged_file=False ,verbose=False):
    # path 내부의 모든 이미지 segmentation, masking
    # https://codechacha.com/ko/python-list-all-files-in-dir/
    files = os.listdir(path)

    if not os.path.exists(path+'_mask'):
        os.mkdir(path+'_mask')
    if save_merged_file & (not os.path.exists(path+'_masked')): # 제대로 마스킹 되었는지 확인
        os.mkdir(path+'_masked')
        
    for file in files:
        img_path = path + '/' + file
        img = cv2.imread(img_path)

        hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,i = cv2.split(hsi)
        mask = np.where((low < h)&(h < high), 255, 0).astype(np.uint8)
        cv2.imwrite(path+'_mask/'+file, mask)
        if verbose:
            print(path+'_mask/'+file+" generated.")
        if save_merged_file:
            masked = AND(img,mask)
            merged = cv2.hconcat([img,masked])
            cv2.imwrite(path+'_masked/'+file, merged)


def edge_detect(mask):
    canny = cv2.Canny(mask,50,180)
    edge = np.zeros(mask.shape)
    edge[canny < 128] = 255
    mask = AND(edge,mask=mask)
    return mask


if __name__ == "__main__":
    # plot histogram
    # test case
    masking('./BIBICHU',plot_hist=False)