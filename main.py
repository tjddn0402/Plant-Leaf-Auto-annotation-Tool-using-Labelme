import sys, os
import glob
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

from masking import masking
import morphological as mp
from mask2polygon import mask2json

"""
bash 에서 다음과 같이 실행

python main.py BIBICHU
python main.py SANMANEUL
python main.py EUNBANGUL
"""
# NAME = sys.argv[1]
# NAME = "SANMANEUL"
NAME = "BIBICHU"
# NAME = "EUNBANGUL"

PATH = './'+NAME


def apply_morph(morph_sequence, mask_path, verbose = False):
    mask_files = os.listdir(mask_path)
    for mask_file in mask_files:
        mask = cv2.imread(mask_path+'/'+mask_file, 0)
        mask = morph_sequence.operate(mask)
        cv2.imwrite(mask_path+'/'+mask_file, mask)
        if verbose:
            print("Filtered mask for " + mask_file + " is generated.")

def gen_labels(verbose = False):
    mask_files = os.listdir(PATH+"_mask")
    for file_name in mask_files:
        mask = cv2.imread(PATH+'_mask/'+file_name, 0)
        JSON = mask2json(mask, PATH, file_name, PATH)
        with open(PATH + Path(file_name).stem + ".json", 'w') as f:
            json.dump(JSON, f, indent=4)
            if verbose:
                print("Label for " + Path(file_name).stem + "is created.")


if __name__ == "__main__":
    # masking -> morph -> mask2polygon -> .json(원래 이미지 있는 폴더에 저장)
    # masking(PATH,plot_hue=False,plot_intensity=False, verbose=True)
    defined_morphologic = mp.MorphSequence({'erosion_1':mp.Erosion(),
                                            'dilation_1':mp.Dilation()
                                            })
    defined_morphologic.summary()
    apply_morph(defined_morphologic, PATH+"_mask", verbose=True)
    gen_labels(verbose=True)