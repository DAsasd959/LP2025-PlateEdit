import numpy as np
import cv2
import torch
from torchmetrics.classification import MulticlassJaccardIndex
import os
from tqdm import tqdm

a = []
filepath = '/media/avlab/disk2/Alex3/chinese_test/i_s_bbox/'
file_list = os.listdir(filepath)
imgpath = '/media/avlab/disk2/Alex3/chinese_test/i_s'


# for file in filepath:
with open('/media/avlab/disk2/Alex3/chinese_test/i_s_bbox/0.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        # print(line)
        s = line.strip('\n').split(' ')
        a.append(s[1:])
        print(s)

x = int(a[0][7])
text_len = len(a)

# for img in imgpath:
img = cv2.imread(r'/media/avlab/disk2/Alex3/chinese_test/i_s/0.png')
# img = np.ascontiguousarray(img)
color = (0, 0, 255)
for num in range(text_len):
    cv2.line(img, (int(a[num][0]), int(a[num][1])), (int(a[num][2]), int(a[num][3])), (0, 0, 255), 3)
    cv2.line(img, (int(a[num][2]), int(a[num][3])), (int(a[num][4]), int(a[num][5])), (0, 0, 255), 3)
    cv2.line(img, (int(a[num][4]), int(a[num][5])), (int(a[num][6]), int(a[num][7])), (0, 0, 255), 3)
    cv2.line(img, (int(a[num][6]), int(a[num][7])), (int(a[num][0]), int(a[num][1])), (0, 0, 255), 3)

cv2.imwrite(os.path.join('/media/avlab/disk2/Alex3/chinese_test/test3.png'),img)
        

