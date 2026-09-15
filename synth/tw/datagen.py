# -*- coding: utf-8 -*-
import os
import cv2
import cfg
from Synthtext.gen import datagen
from tqdm import tqdm
import time

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(data_dir: str, sample_num: int) -> None:
    
    i_s_dir = os.path.join(data_dir, cfg.i_s_dir)
    i_s_bbox_dir = os.path.join(cfg.data_dir, cfg.i_s_bbox_dir)
    t_b_dir = os.path.join(data_dir, cfg.t_b_dir)
    mask_s_dir = os.path.join(data_dir, cfg.mask_s_dir)

    makedirs(i_s_dir)
    makedirs(t_b_dir)
    makedirs(mask_s_dir)
    makedirs(i_s_bbox_dir)

    gen = datagen()
    digit_num = len(str(sample_num)) - 1

    i_s_text_dir = os.path.join(data_dir, 'i_s.txt')
    f1 = open(i_s_text_dir, 'w+', encoding='utf-8')
    f3 = open(os.path.join(data_dir, 'angle_stat.txt'), 'w+', encoding='utf-8')

    # 用 tqdm 包裝迴圈
    for idx in tqdm(range(sample_num), desc="Generating plates"):
        try:
            # 正確解包 9 個返回值
            i_s, t_sk, surf, t_b, fg_col, char_bbox, text_bbox, angle, font_name, text1 = gen.gen_srnet_data_with_background()

            # 圖片路徑
            i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
            t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
            mask_s_path = os.path.join(mask_s_dir, str(idx).zfill(digit_num) + '.png')
            i_s_bbox_path = os.path.join(i_s_bbox_dir, str(idx).zfill(digit_num) + '.txt')
            # 寫入文字 bbox
            with open(i_s_bbox_path, 'w', encoding='utf-8') as bbox_fw:

                # text bbox
                text_flat = [str(p) for pt in text_bbox for p in pt]
                bbox_fw.write(text1 + ' ' + ' '.join(text_flat) + '\n')

                # char bbox
                for char, char_b in zip(text1, char_bbox):
                    char_flat = [str(p) for pt in char_b for p in pt]
                    bbox_fw.write(char + ' ' + ' '.join(char_flat) + '\n')

            # 寫圖片
            cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(mask_s_path, surf, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # surf 當作 mask

            # 寫文字檔
            f1.writelines(f"{str(idx).zfill(digit_num)}.png {text1}\n")

            # 寫角度資訊
            f3.writelines(f"{str(idx).zfill(digit_num)}.png {' '.join(map(str, angle))}\n")

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    f1.close()
    f3.close()

if __name__ == '__main__':
    main(data_dir=cfg.data_dir, sample_num=cfg.sample_num)