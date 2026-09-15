# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def draw_bbox(img, pts, color=(0,255,0), thickness=2):
    pts = np.array(pts, dtype=np.int32).reshape(4,2)
    cv2.polylines(img, [pts], True, color, thickness)
    return img

def parse_bbox_file(bbox_path):
    data = []
    with open(bbox_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split()
            char = parts[0]
            coords = list(map(float, parts[1:]))
            pts = [(coords[i], coords[i+1]) for i in range(0,8,2)]
            data.append((char, pts))
    return data


def visualize():

    image_dir = '20260423_test500/i_s'
    bbox_dir = '20260423_test500/i_s_bbox'

    save_dir = '20260423_test500/vis'
    os.makedirs(save_dir, exist_ok=True)

    image_list = sorted(os.listdir(image_dir))

    for img_name in image_list:

        img_path = os.path.join(image_dir, img_name)
        bbox_path = os.path.join(bbox_dir, img_name.replace(".png",".txt"))

        if not os.path.exists(bbox_path):
            continue

        img = cv2.imread(img_path)

        bbox_data = parse_bbox_file(bbox_path)

        for i,(char,pts) in enumerate(bbox_data):

            if i == 0:
                color = (0,0,255)  # text bbox 紅色
            else:
                color = (0,255,0)  # char bbox 綠色

            img = draw_bbox(img, pts, color)

            x,y = int(pts[0][0]), int(pts[0][1])
            cv2.putText(img,char,(x,y-3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,color,1,cv2.LINE_AA)

        save_path = os.path.join(save_dir,img_name)
        cv2.imwrite(save_path,img)

    print("Visualization finished!")


if __name__ == "__main__":
    visualize()