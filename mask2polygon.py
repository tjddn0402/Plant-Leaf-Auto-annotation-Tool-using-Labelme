import os
import json
import numpy as np
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import labelme
from labelme.utils import img_b64_to_arr, img_arr_to_b64
import base64

# Reference
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_polygon.html
# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
# labelme imageData 필드란 무엇인가
# https://stackoverflow.com/questions/57004792/what-is-imagedata-in-json-file-which-comes-from-labelme-tool
# https://github.com/cocodataset/cocoapi/issues/131
# https://github.com/wkentaro/labelme/issues/376

def mask2poly(mask, label=None, shape_type="polygon"):
    """
    iscrowd: 
    """
    coords = []
    for contour in find_contours(mask, 0):
        coord = approximate_polygon(contour, tolerance=3)
        coord = np.flip(coord,1)
        coords.append(coord)
    

    annotations = []
    for coord in coords:
        if len(coord) > 5:
            ann = {"label":label,
                "points":coord.tolist(),
                "group_id":None,
                "shape_type":shape_type,
                "flags":{}
            }
            annotations.append(ann)
    return annotations

def mask2json(mask, dir, file_name, label):
    """
    mask: annotation을 얻고자 하는 이미지의 mask (cv2.imread로 미리 load해놔야 함)
    dir: 원본 파일이 포함되어 있는 directory
    file_name: 원본 파일의 이름 ex)OOOO.jpg
    label: 이미지의 label
    """
    H,W = mask.shape
    data = labelme.LabelFile.load_image_file(os.path.join(dir,file_name))
    image_data = base64.b64encode(data).decode('utf-8')

    JSON = {"version":labelme.__version__,
            "flags":{},
            "shapes":mask2poly(mask, label),
            "imagePath":file_name,
            "imageData":image_data,
            "imageHeight":H,
            "imageWidth":W
    }
    return JSON
    

if __name__=="__main__":
    # test case
    import cv2
    dir = './'
    file_name = 'result_6.jpg'
    sample_mask = cv2.imread(os.path.join(dir,file_name), 0)
    sample_json = mask2json(sample_mask, dir, file_name, label="SANMANEUL")
    with open('./result_6.json', 'w') as f:
        json.dump(sample_json, f, indent=4)