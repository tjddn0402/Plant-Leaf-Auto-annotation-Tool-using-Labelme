import os, glob
from pathlib import Path
import cv2
from imageio import save
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter
from BinaryMorph import MorphChain, Dilation

####################################################################
# edge를 어떻게 처리할 것인지 정한다.
edge_dilation = MorphChain({'dil_1':Dilation(), 'dil_2':Dilation()})
####################################################################


root = tkinter.Tk()
root.wm_withdraw()

PATH = filedialog.askdirectory(initialdir="C:\\", \
    title="선택한 디렉터리의 모든 이미지에 대한 마스크를 생성합니다.")

def AND_(img,mask):
    H, W, C = img.shape
    masked = img.copy()
    for h in range(H):
        for w in range(W):
            if mask[h][w]==0:
                masked[h][w]=0
    return masked


def get_edge(img, show=False):
    gaussian_kernel = np.array([[1,4,6,4,1],
                                [4,16,24,16,4],
                                [6,24,36,24,6],
                                [4,16,24,16,4],
                                [1,4,6,4,1]],dtype=np.float64)/256
    filtered = cv2.filter2D(img, -1, gaussian_kernel)
    canny = cv2.Canny(filtered,50,180)
    if show:
        plt.hist(canny.ravel(),bins=256,range=[0,256]),plt.show()
    edge = np.where((canny>128), 0, 255)
    edge = edge_dilation.operate(edge)
    return edge


def masking(img,hue_low=40,hue_high=90, draw_edge=False):
    """
    입력받은 한장의 이미지를 hue의 값의 범위에 따라 초록색만 masking.
    필요할 경우 draw_edge = True로 하여 edge로 masking된 부분 구별
    """
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,_,i = cv2.split(hsi)

    green = np.where((hue_low < h)&(h < hue_high), 255, 0).astype(np.uint8)

    if draw_edge:
        # edge부분을 0으로 함.
        edge = get_edge(i,show=False)
        green = np.where((green&edge),255,0)

    return green.astype(np.uint8)


def masking_all(path, hue_low=40, hue_high=90, edge=False, save_merged_file=False ,verbose=False):
    """
    path 내부의 모든 jpg 이미지 segmentation, masking
        path - 
        hue_low - 마스킹할 hue range의 최솟값
        hue_high - 마스킹할 hue range의 최댓값
        save_merged_file - 마스킹된 이미지와 이미지를 concatenate 하여 저장합니다.
        verbose - 진행상황을 보여줍니다.
    """
    # https://codechacha.com/ko/python-list-all-files-in-dir/
    MASK_PATH = path+'_mask'
    if not os.path.exists(MASK_PATH):
        os.mkdir(MASK_PATH)
    if save_merged_file & (not os.path.exists(path+'_masked')):
        # 마스킹된 이미지와 이미지를 concatenate 하여 저장합니다.
        os.mkdir(path+'_masked')
        
    files = glob.iglob(os.path.join(path,"*.jpg"))
    files = [Path(file).stem for file in files]
    # print(files)
    for file in files:
        img = cv2.imread(os.path.join(path,file+'.jpg'))
        # assert img
        mask = masking(img, hue_low, hue_high, edge)
        # edge = get_edge(img)
        cv2.imwrite(os.path.join(MASK_PATH,file+'.jpg'), mask)

        if verbose:
            print(os.path.join(MASK_PATH,file+".jpg")+" generated.")
            
        if save_merged_file:
            masked = AND_(img,mask)
            merged = cv2.hconcat([img,masked])
            cv2.imwrite(path+'_masked/'+file+'.jpg', merged)


masking_all(PATH, hue_low=30, hue_high=90, edge=False, save_merged_file=True, verbose=True)
root.destroy()