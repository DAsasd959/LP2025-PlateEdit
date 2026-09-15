import os 
import shutil

path = '/home/avlab/下載/fonts-main/apache'
font_path = '/home/avlab/scenetext/SRNet-Datagen_CCPD/datasets/fonts/english_ttf'


for root,folders,_ in os.walk(path):
    for folder in folders:
        folder_path = os.path.join(root,folder)
        for file in os.listdir(folder_path):
            if '.ttf' in file:
                old_path = os.path.join(folder_path,file)
                new_path = os.path.join(font_path,file)
                shutil.copy(old_path,new_path)