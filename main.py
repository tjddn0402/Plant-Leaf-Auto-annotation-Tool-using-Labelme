import sys, os
import glob
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tkinter import filedialog

from mask import histogram_hsi, masking_all
import BinaryMorph as bm
from mask2polygon import mask2json


PATH = filedialog.askdirectory(initialdir="C:\\", \
    title="디렉터리를 선택하십쇼.")



def apply_morph(morph_sequence, mask_path, verbose = False):
    """
    정의한 Morphologic 연산들을
    폴더 내 모든 파일에 적용 후 저장
    parameters:
        morph_sequence - 정의한 MorphChain 객체
        mask_path - 처리한 binary 마스크를 저장할 directory
        verbose - 진행상황 관찰하고 싶을 때 True 로 하면 한장씩 처리될때마다 메시지 띄워줌
    """
    mask_files = os.listdir(mask_path)
    for mask_file in mask_files:
        mask = cv2.imread(mask_path+'/'+mask_file, 0)
        mask = morph_sequence.operate(mask) # 정의한 연산 적용
        cv2.imwrite(mask_path+'/'+mask_file, mask)
        if verbose:
            print("Filtered mask for " + mask_file + " is generated.")

def gen_labels(verbose = False):
    mask_files = os.listdir(PATH+"_mask")
    print(mask_files)
    for file_name in mask_files:
        print(file_name)
        mask = cv2.imread(os.path.join(PATH,file_name), 0)
        JSON = mask2json(mask, PATH, file_name, PATH)
        # print(JSON)
        if JSON is not None:
            with open(os.path.join(PATH, Path(file_name).stem+".json"), 'w') as f:
                json.dump(JSON, f, indent=4)
                if verbose:
                    print("Label for " + Path(file_name).stem + " is created.")


# masking -> morph -> mask2polygon -> .json(원래 이미지 있는 폴더에 저장)
# masking_all(PATH, verbose=True)

# defined_morphologic = bm.MorphChain({'erosion_1':bm.Erosion(),
#                                     'erosion_2':bm.Erosion(),
#                                     'dilation_1':bm.Dilation(),
#                                     'dilation_2':bm.Dilation()
#                                     })
# defined_morphologic.summary()

# apply_morph(defined_morphologic, PATH+"_mask", verbose=True)

gen_labels(verbose=True)
