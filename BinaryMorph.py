from collections import OrderedDict
import cv2
import numpy as np



"""
Morphological Operations of Binary Image
"""

class Operator():
    """
    모든 연산이 공통적으로 상속받는 연산 기본클래스로 정의
    attributes:
        kernel
        KH, KW
        PH, PW
    method:
        set_kernel
    """
    def __init__(self, kernel=None):
        if kernel is None:
            # 커널을 따로 지정하지 않으면 3x3 의 1이 채워진 행렬
            self.kernel = np.ones((3,3), dtype=np.uint8)
        else:
            self.kernel = kernel

        self.KH, self.KW = self.kernel.shape
        self.PH, self.PW = self.KH//2, self.KW//2
    
    def set_kernel(self, kernel):
        """
        연산의 커널을 바꾸고싶을 때 사용
        """
        self.kernel = kernel

class Erosion(Operator):
    def __init__(self, kernel=None):
        super().__init__(kernel)

    def execute(self, mask):
        H,W = mask.shape
        mask_pad = np.pad(mask,((self.PH,self.PH),(self.PW,self.PW)), mode='edge') # replicative padding

        result = np.zeros((H,W))
        for i in range(H):
            for j in range(W):
                ij = np.multiply(mask_pad[i:i+self.KH,j:j+self.KW], self.kernel)
                if np.any(ij == 0):
                    result[i][j] = 0
                else:
                    result[i][j] = 255
        return result.astype(np.uint8)

class Dilation(Operator):
    def __init__(self, kernel=None):
        super().__init__(kernel)

    def execute(self, mask):
        H,W = mask.shape
        mask_pad = np.pad(mask,((self.PH,self.PH),(self.PW,self.PW)), mode='edge') # replicative padding

        result = np.zeros((H,W))
        for i in range(H):
            for j in range(W):
                ij = np.multiply(mask_pad[i:i+self.KH,j:j+self.KW], self.kernel)
                if np.all(ij == 0): 
                    result[i][j] = 0
                else:
                    result[i][j] = 255
        return result.astype(np.uint8)

class Opening(Operator):
    def __init__(self, kernel=None):
        super().__init__(kernel)
        self.erosion = Erosion(kernel)
        self.dilation = Dilation(kernel)

    def execute(self, mask):
        eroded = self.erosion.execute(mask)
        opened = self.dilation.execute(eroded)
        return opened.astype(np.uint8)

class Closing(Operator):
    def __init__(self, kernel=None):
        super().__init__(kernel)
        self.dilation = Dilation(kernel)
        self.erosion = Erosion(kernel)

    def execute(self, mask):
        dilated = self.dilation.execute(mask)
        closed = self.erosion.execute(dilated)
        return closed.astype(np.uint8)

    

class HitMiss():
    def __init__(self, B1=None, B2=None):
        if B1 is None:
            # 커널을 따로 지정하지 않으면 3x3 의 1이 채워진 행렬
            self.B1 = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
        else:
            self.B1 = B1
        if B2 is None:
            self.B2 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.uint8)
        else:
            self.B2 = B2

    def execute(self, mask):
        mask_c = np.where((mask>0), 0, 255)
        erosion1 = Erosion(self.B1)
        erosion2 = Erosion(self.B2)
        return np.where((erosion1(mask) & erosion2(mask_c)), 255, 0).astype(np.uint8)

class HoleFilling(Operator):
    def __init__(self, kernel=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)):
        super().__init__(kernel)
        self.dilation = Dilation()
    
    def execute(self, mask, init_point):
        x,y = init_point
        hole = np.zeros_like(mask)
        hole[y,x] = 255
        mask_c = np.full_like(mask,255)
        mask_c = mask_c - mask

        before = hole
        while True:
            dilated = self.dilation.execute(before)
            dilated[dilated != mask_c] = 0
            after = dilated
            if (before == after).all():
                break            
            before = after

        result = after + mask
        return result.astype(np.uint8)

class ConvexHull(Operator):
    def __init__(self, kernel=None):
        super().__init__(kernel)

    def execute(self, mask):
        raise NotImplementedError



class MorphChain():
    """
    attributes:
        operations - OrderedDict 객체. Mask image에 순서대로 적용할 연산의 이름 - 연산자를 key-value 형태로 저장
    methods:
        add - 적용할 연산자를 추가
        sub - key 값으로 연산자 제거
        summary - 포함된 Operator 순서대로 출력
        operate - binary mask를 입력받아 
    """
    def __init__(self, operations=None):
        if operations is None:
            self.operations = OrderedDict()
        self.operations = OrderedDict(operations)

    def add(self, opname, operation):
        self.operations[opname] = operation

    def sub(self, opname):
        self.operations.pop(opname)

    def summary(self):
        for (opname, op) in self.operations.items():
            print(opname, op)

    def operate(self, mask, save_path = None, show_result = False, verbose=False):
        for opname, operation in self.operations.items():
            mask = operation.execute(mask)
            if verbose:
                print(f'{opname} executed')

        if save_path is not None:
            cv2.imwrite(save_path, mask)

        if show_result:
            cv2.imshow("result", mask)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return mask
    

if __name__ == "__main__":
    # test cases
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.wm_withdraw()

    PATH = filedialog.askdirectory(initialdir="C:\\", \
        title="선택한 디렉터리의 모든 마스크 이미지를 변환합니다.")

    defined_morphologic = MorphChain({'erosion_1':Erosion(),
                                        'erosion_2':Erosion(),
                                        'erosion_3':Erosion(),
                                        'erosion_4':Erosion(),
                                        'dilation_1':Dilation(),
                                        'dilation_2':Dilation(),
                                        'dilation_3':Dilation(),
                                        'dilation_4':Dilation()})
    defined_morphologic.summary()

    import glob, os
    mask_image_files = glob.iglob(os.path.join(PATH,"*.jpg"))

    for mask_file in mask_image_files:
        mask = cv2.imread(mask_file,0) # 이미지 경로 입력
        defined_morphologic.operate(mask, save_path=mask_file, show_result=False, verbose=False)
        print(mask_file+" retouched.")
