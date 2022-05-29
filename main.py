import sys, os
import glob
import json
import numpy as np
import cv2
from pathlib import Path

from masking import masking
import morph as mp
from mask2polygon import mask2json

"""
bash 에서 다음과 같이 실행

python main.py SANMANEUL
"""
NAME = sys.argv[1]
# SANMANEUL
# BIBICHU
# EUNBANGUL
PATH = './'+NAME


def define_morph(mask):
    """
    이미지 마스크에 적용할 Morphological연산을 사용자가 정의
    """
    mask = mp.repErosionDilation(mask,rep=6)
    mask = mp.holeFilling(mask)
    return mask

def apply_morph(mask_path):
    mask_files = os.listdir(mask_path)
    for mask_file in mask_files:
        mask = cv2.imread(mask_path+'/'+mask_file, 0)
        mask = define_morph(mask)
        cv2.imwrite(mask_path+'/'+mask_file, mask)

def gen_labels():
    mask_files = os.listdir(PATH+"_mask")
    for file_name in mask_files:
        mask = cv2.imread(PATH+'_mask/'+file_name, 0)
        JSON = mask2json(mask, PATH, file_name, PATH)
        with open(PATH + Path(file_name).stem + ".json", 'w') as f:
            json.dump(JSON, f, indent=4)


if __name__ == "__main__":
    # masking -> morph -> mask2polygon -> .json(원래 이미지 있는 폴더에 저장)
    masking(PATH,plot_hist=False)
    apply_morph(PATH+"_mask")