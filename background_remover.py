import cv2
import tkinter
from tkinter import messagebox, filedialog
from pathlib import Path, PurePath
import numpy as np
import os
from BinaryMorph import MorphChain, Erosion, Dilation
from mask2polygon import mask2json
import json

######################### 적용하고싶은 연산 정의 ###############################################
defined_morph = MorphChain({'e1':Erosion(),'e2':Erosion(),'d1':Dilation(),'d2':Dilation()})
######################### 제거할 주변값 정의 ###################################################
H_RANGE = 10
S_RANGE = 10
# I_RANGE = 3
###############################################################################################

root = tkinter.Tk()
root.wm_withdraw()

keep_go = True
while keep_go:
    # 1. 편집할 파일 선택
    file_full_path = filedialog.askopenfilename(initialdir="./dataset",\
                                        title = "이미지 파일을 선택 해 주세요",\
                                        filetypes = [("Image", "*.jpg *.jpeg")])
    file_name = Path(file_full_path).stem
    class_name = PurePath(file_full_path).parent.name
    img_path = os.path.abspath(os.path.join(file_full_path, os.pardir))
    parent_path = os.path.abspath(os.path.join(img_path, os.pardir))
    mask_path = os.path.join(parent_path, class_name+"_mask")
    mask_path = os.path.join(mask_path, file_name+".jpg")

    # print(file_full_path)

    # 2. 클릭해서 좌표 얻기
    image = cv2.imread(file_full_path)
    hsi_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_chan,s_chan,i_chan = cv2.split(hsi_img)

    coordinates = []
    def onmouse(event, x,y,flags, param):
        """
        좌클릭시 coordinates에 좌표 추가
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append([x,y])
            print(f"selected coordinate: {x}, {y}")
            h = h_chan[y,x]
            s = s_chan[y,x]
            i = i_chan[y,x]
            print(f"hue: {h}, saturation: {s}, intensity:{i}\n")

    cv2.imshow(file_name, image)
    cv2.setMouseCallback(file_name, on_mouse=onmouse)
    cv2.waitKey(),cv2.destroyAllWindows()



    # 3. 선택한 좌표에서 hsi값 얻고 마스크 재생성
    mask = cv2.imread(mask_path, 0)

    for coord in coordinates:
        x, y = coord
        h = h_chan[y,x]
        s = s_chan[y,x]
        i = i_chan[y,x]
        # print(f"hue: {h}, saturation: {s}, intensity:{i}")
        kill = np.where((h-H_RANGE<h_chan)&(h_chan<h+H_RANGE)&(s-S_RANGE<s_chan)&(s_chan<s+S_RANGE), 0, 255)
        mask = np.where((mask>0)&(kill>0), 255, 0)

    mask = defined_morph.operate(mask=mask)
    cv2.imwrite(mask_path, mask)

    # 4. 잡음 제거된 마스크 이미지로 annotation 생성
    json_file = mask2json(mask, img_path, file_name+".jpg", class_name)

    with open(os.path.join(img_path, file_name+".json"), 'w') as f:
        json.dump(json_file, f, indent=4)

    keep_go = messagebox.askyesno(title="알림" ,message="다른 이미지로도 계속 진행하시겠습니까?")