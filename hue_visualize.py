import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize():
    sh = np.zeros((256,180,3))
    sh.fill(200)
    S,H,C = sh.shape

    for h in range(H):
        for s in range(S):
                sh[s,h,0]=h
                sh[s,h,1]=s

    sh = cv2.cvtColor(sh.astype(np.uint8),cv2.COLOR_HSV2BGR)
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(sh),plt.show()

# brown = cv2.cvtColor(np.array([[[165,42,42]]],dtype=np.uint8),cv2.COLOR_BGR2HSV)
# print(brown)

holes = []
def onmouse(event, x,y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        holes.append([x,y])
    if event == cv2.EVENT_RBUTTONDBLCLK:
        keep_click=False
    

if __name__=="__main__":
    visualize()
    img_path = sys.argv[1]
    img = cv2.imread(img_path)

    # from mask import calcHist_hsi
    # h_hist,s_hist,i_hist = calcHist_hsi(img_path)
    # plt.plot(h_hist),plt.show()

    keep_click = True
    while keep_click:
        cv2.setMouseCallback('select hole(left click), stop selecting hole(right click)', onMouse=onmouse)

