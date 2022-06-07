import os, glob
import json
import numpy as np
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import labelme
import base64
import cv2
from pathlib import Path, PurePath
import tkinter
from tkinter import filedialog

# Reference
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_polygon.html
# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
# labelme imageData 필드란 무엇인가
# https://stackoverflow.com/questions/57004792/what-is-imagedata-in-json-file-which-comes-from-labelme-tool
# https://github.com/cocodataset/cocoapi/issues/131
# https://github.com/wkentaro/labelme/issues/376


def mask2poly(mask, label=None, shape_type="polygon"):
    """
    label - labeling 이미지가 가지는 클래스
    """
    H,W = mask.shape
    PAD_LENGTH = 50
    mask_pad = np.pad(mask,PAD_LENGTH)
    contours = find_contours(mask_pad, 0)
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
    # contours = find_contours(mask, fully_connected='high')
    if contours is None:
        # print("failed to draw contours")
        return None
    coords = []
    for contour in contours:
        coord = approximate_polygon(contour, tolerance=0.02)
        coord = np.flip(coord,1) - PAD_LENGTH
        coord_x = np.clip(coord[:,0],0,W-1)
        coord_y = np.clip(coord[:,1],0,H-1)
        coord = np.dstack((coord_x, coord_y))
        coord = coord[0]
        coords.append(coord[::15]) # step size설정하지 않으면 너무 촘촘히 찍힘.
    
    annotations = []
    for coord in coords:
        if len(coord) > 20: # 점이 일정갯수 이상 찍혀야 인정.
            ann = {"label":label,
                "points":coord.tolist(),
                "group_id":None,
                "shape_type":shape_type,
                "flags":{}
            }
            annotations.append(ann)
    return annotations

def mask2json(mask, dir, file_name, label=None):
    """
    mask: annotation을 얻고자 하는 이미지의 mask (cv2.imread로 미리 load해놔야 함)
    dir: 원본 파일이 포함되어 있는 directory
    file_name: 원본 파일의 이름 ex)OOOO.jpg (확장자 포함)
    label: 이미지의 label
    """
    H,W = mask.shape

    try:
        data = labelme.LabelFile.load_image_file(os.path.join(dir,file_name))
    except:
        print(file_name+' cannot processed. Check if you input correct dir or file_name')
        return

    image_data = base64.b64encode(data).decode('utf-8')

    coords = mask2poly(mask,label)
    if coords is not None:
        JSON = {"version":labelme.__version__,
                "flags":{},
                "shapes":mask2poly(mask, label),
                "imagePath":file_name,
                "imageData":image_data,
                "imageHeight":H,
                "imageWidth":W
        }
        return JSON 
    else:
        print("failed to draw contour for {}".format(file_name))
        return None

def gen_labels(verbose = True):
    for file_path in MASK_FILES:
        # print(file_path)
        file_name = Path(file_path).stem
        mask = cv2.imread(file_path, 0)
        JSON = mask2json(mask, PATH, file_name+'.jpg', LABEL)
        # print(JSON)
        if JSON is not None:
            with open(os.path.join(PATH, file_name+".json"), 'w') as f:
                json.dump(JSON, f, indent=4)

                if verbose: # 하나의 json 생성할때마다 출력
                    print("Label for " + file_name + " is created.")

if __name__=="__main__":       
    root = tkinter.Tk()
    root.wm_withdraw()
    PATH = filedialog.askdirectory(initialdir="C:\\", \
    title="레이블을 생성할 이미지가 들어있는 디렉터리를 선택하십쇼.")
    LABEL = PurePath(PATH).name
    MASK_PATH = os.path.join(PurePath(PATH).parent, LABEL+"_mask")
    MASK_FILES = list(glob.iglob(os.path.join(MASK_PATH,"*.jpg")))
    # MASK_FILES = MASK_FILES[104:] # 데이터셋의 일부만 진행.

    gen_labels(verbose=True)


# if __name__=="__main__":
#     # test case
#     # import cv2
#     # file_name = './dataset/BIBICHU_mask/0128.jpg'
#     # sample_mask = cv2.imread(file_name, 0)
#     # sample_json = mask2json(sample_mask, './dataset/BIBICHU', '0128.jpg', label="BIBICHU")
#     # with open('./dataset/BIBICHU/0128.json', 'w') as f:
#     #     json.dump(sample_json, f, indent=4)
#     directory_path = tkinter.filedialog()