"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x1_paths = paths
        self.x2_paths = list(map(lambda x : x.replace('x1', 'x2'), self.x1_paths))
        self.y_paths = list(map(lambda x : x.replace('x1', 'y'), self.x1_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode


    def __len__(self):
        return len(self.x1_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x1_paths[id_]) # Get filename for logging
        x1 = cv2.imread(self.x1_paths[id_], cv2.IMREAD_COLOR)
        x2 = cv2.imread(self.x2_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x1.shape

        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB)
        x1 = cv2.resize(x1, self.input_size)
        x1 = self.scaler(x1)
        x1 = np.transpose(x1, (2, 0, 1))

        x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)
        x2 = cv2.resize(x2, self.input_size)
        x2 = self.scaler(x2)
        x2 = np.transpose(x2, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)

            image = y
            for i in range(len(image)):
                for j in range(len(image)):
                    if image[i][j] == 1:
                        image[i][j] = 100
                        print(image[i][j])
                    elif image[i][j] == 2:
                        image[i][j] = 180
                        print(image[i][j])
                    elif image[i][j] == 3:
                        image[i][j] = 255

            import matplotlib.pylab as plt
            plt.imshow(image)

            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)
            return x1, x2, y, filename

        elif self.mode in ['test']:
            return x1,x2, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"


