import cv2
import numpy as np


def erosion(mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3,3))

    H,W = mask.shape
    KH,KW = kernel.shape
    PH,PW = KH//2,KW//2
    mask_pad = np.pad(mask,((PH,PH),(PW,PW)), mode='edge') # replicative padding

    result = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            ij = np.multiply(mask_pad[i:i+KH,j:j+KW], kernel)
            if np.any(ij == 0):
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result


def dilation(mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3,3))

    H,W = mask.shape
    KH,KW = kernel.shape
    PH,PW = KH//2,KW//2
    mask_pad = np.pad(mask,((PH,PH),(PW,PW)), mode='edge') # replicative padding

    result = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            ij = np.multiply(mask_pad[i:i+KH,j:j+KW], kernel)
            if np.all(ij == 0): 
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result


def opening(mask, kernel):
    if kernel is None:
        kernel = np.ones((3,3))

    eroded = erosion(mask)
    opened = dilation(eroded)
    return opened

def closing(mask, kernel):
    if kernel is None:
        kernel = np.onesd((3,3))

    dilated = dilation(mask)
    closed = erosion(dilated)
    return closed

def repErosionDilation(mask, rep = 2, kernel=None):
    """
    Erosion을 rep번, Dilation을 rep번 한다.
    """
    if kernel is None:
        kernel = np.ones((3,3))

    for _ in range(rep):
        mask = erosion(mask)
    for _ in range(rep):
        mask = dilation(mask)
    return mask

def repDilationErosion(mask, rep = 2, kernel=None):
    """
    Dilation을 rep번, Erosion을 rep번 한다.
    """
    if kernel is None:
        kernel = np.ones((3,3))

    for _ in range(rep):
        mask = dilation(mask)
    for _ in range(rep):
        mask = erosion(mask)
    return mask
    
def holeFilling(mask, kernel=None):
    if kernel is None:
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    return mask
    

if __name__ == "__main__":
    # test cases
    test_mask = cv2.imread('./BIBICHU_mask/0745.jpg',0)
    cv2.imwrite(f'result_6.jpg',repErosionDilation(test_mask,6))