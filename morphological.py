from collections import OrderedDict
from sys import implementation
import cv2
import numpy as np


class Operation():
    """
    모든 연산에 공통적으로 적용되는 연산 기본클래스로 정의
    """
    def __init__(self, kernel=None):
        if kernel is None:
            self.kernel = np.ones((3,3), dtype=np.uint8)
        else:
            self.kernel = kernel

        self.KH, self.KW = self.kernel.shape
        self.PH, self.PW = self.KH//2, self.KW//2
    
    def set_kernel(self, kernel):
        if not kernel:  
            raise AssertionError
        self.kernel = kernel

class Erosion(Operation):
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

class Dilation(Operation):
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

class Opening(Operation):
    def __init__(self, kernel=None):
        super().__init__(kernel)
        self.erosion = Erosion(kernel)
        self.dilation = Dilation(kernel)

    def execute(self, mask):
        eroded = self.erosion(mask)
        opened = self.dilation(eroded)
        return opened.astype(np.uint8)

class Closing(Operation):
    def __init__(self, kernel=None):
        super().__init__(kernel)
        self.dilation = Dilation(kernel)
        self.erosion = Erosion(kernel)

    def execute(self, mask):
        dilated = self.dilation(mask)
        closed = self.erosion(dilated)
        return closed.astype(np.uint8)

class HoleFilling(Operation):
    def __init__(self, kernel=None):
        super().__init__(kernel)
    
    def execute(self, mask):
        raise NotImplementedError

class MorphSequence():
    def __init__(self, operations=None):
        if operations is None:
            self.operations = OrderedDict()
        self.operations = OrderedDict(operations)

    def add(self, opname, operation):
        if (opname|operation) is None:
            raise AssertionError
        self.operations[opname] = operation

    def sub(self, opname):
        self.operations.pop(opname)

    def summary(self):
        for (opname, op) in self.operations.items():
            print(opname, op)

    def operate(self, mask, save_path = None, show_result = False):
        for _, operation in self.operations.items():
            mask = operation.execute(mask)

        if save_path is not None:
            cv2.imwrite(save_path, mask)

        if show_result:
            cv2.imshow("result", mask)
        
        return mask
    

if __name__ == "__main__":
    # test cases
    test_mask = cv2.imread('./BIBICHU_mask/0745.jpg',0)
    defined_morphologic = MorphSequence({'erosion_1':Erosion(),
                                        'erosion_2':Erosion(),
                                        'erosion_3':Erosion(),
                                        'erosion_4':Erosion(),
                                        'erosion_5':Erosion(),
                                        'erosion_6':Erosion(),
                                        'dilation_1':Dilation(),
                                        'dilation_2':Dilation(),
                                        'dilation_3':Dilation(),
                                        'dilation_4':Dilation(),
                                        'dilation_5':Dilation(),
                                        'dilation_6':Dilation()
                                        })
    defined_morphologic.summary()
    test_mask = defined_morphologic.operate(test_mask, save_path="result.jpg")
