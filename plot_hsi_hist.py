import numpy as np
import cv2
import glob, os
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog
import tkinter


"""
주의사항.
첫번째 hue histogram 창이 뜬 후, tkinter window를 닫지 않으면
두, 세번째 histogram을 볼 수 없다.
(원인파악 못함)
"""

# root = tkinter.Tk()
# root.wm_withdraw()

PATH = filedialog.askdirectory(initialdir="C:\\", \
    title="선택한 디렉터리의 모든 이미지의 hsi 채널에 대한 대한 histogram을 합쳐서 보여줍니다.")

def calcHist_hsi(img_path):
    """
    한 이미지에 대한 h,s,i 채널의 histogram return
    """
    img = cv2.imread(img_path)
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSI 색상계로 변환
    hue_hist = cv2.calcHist([hsi],[0],None,[180],[0,180]) # hue의 level은 180단계로 나눠짐.
    saturation_hist = cv2.calcHist([hsi],[1],None,[256],[0,256])
    intensity_hist = cv2.calcHist([hsi],[2],None,[256],[0,256])
    return [hue_hist, saturation_hist, intensity_hist]

def histogram_hsi(dir_path):
    """
    directory 내부의 모든 이미지에 대한
    h,s,i plot을 모두 더하여 한꺼번에 본다.
    """
    hue_hist = np.zeros((180,1))
    saturation_hist = np.zeros((256,1))
    intensity_hist = np.zeros((256,1))

    files = glob.iglob(os.path.join(dir_path,"*.jpg")) # 디렉터리 내 모든 jpg이미지 불러오기
    for file in files:
        hsi_hist = calcHist_hsi(file)
        hue_hist += hsi_hist[0]
        saturation_hist += hsi_hist[1]
        intensity_hist += hsi_hist[2]
        
    plt.plot(hue_hist),plt.title(f'{Path(dir_path).stem} hue'),plt.show()
    plt.plot(saturation_hist),plt.title(f'{Path(dir_path).stem} saturation'),plt.show()
    plt.plot(intensity_hist),plt.title(f'{Path(dir_path).stem} intensity'),plt.show()

histogram_hsi(PATH)

# root.destroy()