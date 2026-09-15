# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""

import os
import time
import cv2
import math
import numpy as np
import pygame
from pygame import freetype
import random
import multiprocessing
import queue
import Augmentor


from . import render_text_by_fontsize as render
from . import colorize
from . import skeletonization
from . import render_standard_text
from . import data_cfg
import pickle as cp
       
class Sampler:
    @staticmethod
    def normal(mean, std, min_val=None, max_val=None):
        val = np.random.normal(mean, std)
        if min_val is not None: val = max(val, min_val)
        if max_val is not None: val = min(val, max_val)
        return val

    @staticmethod
    def gamma(shape, scale, offset=0, min_val=None, max_val=None):
        val = np.random.gamma(shape, scale) + offset
        if min_val is not None: val = max(val, min_val)
        if max_val is not None: val = min(val, max_val)
        return int(val)
    
    @staticmethod
    def log_normal(mean_log, std_log, offset=0, min_val=None, max_val=None):
            """
            生成對數常態分佈 (Log-Normal Distribution)
            非常適合模擬自然場景中，物件因距離變化產生的像素大小分佈
            """
            val = np.random.lognormal(mean_log, std_log) + offset
            
            # 截斷機制 (Truncation): 避免過小或過大的極端值
            if min_val is not None: 
                val = max(val, min_val)
            if max_val is not None: 
                val = min(val, max_val)
                
            return int(val)

def select_texts(texts_list):
    valid = [t for t in texts_list if 6 <= len(t) <= 10]
    return random.choice(valid)

class datagen():

    def __init__(self):
        freetype.init()
        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        root_file_path = os.path.dirname(cur_file_path)

        # 1. 字體路徑（使用絕對路徑）
        font_dir = os.path.join(root_file_path, data_cfg.font_dir)
        self.font_list = [os.path.join(font_dir, f) for f in os.listdir(font_dir)]

        # 2. 文字列表
        text_file_path = os.path.join(cur_file_path, data_cfg.text_filepath)
        with open(text_file_path, 'r') as text_file:
            self.texts_list = [text.strip('\n') for text in text_file.readlines()]

        # 3. 顏色矩陣
        color_file_path = os.path.join(cur_file_path, data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_file_path)

        # 4. 背景列表
        bg_file_path = os.path.join(root_file_path, data_cfg.bg_filepath)
        with open(bg_file_path, 'rb') as f:
            self.bg_list = list(cp.load(f))

        self.bg_folder = os.path.join(root_file_path, data_cfg.temp_bg_path)

        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(probability=data_cfg.brightness_rate,
                                            min_factor=data_cfg.brightness_min,
                                            max_factor=data_cfg.brightness_max)
        self.bg_augmentor.random_color(probability=data_cfg.color_rate,
                                    min_factor=data_cfg.color_min,
                                    max_factor=data_cfg.color_max)
        self.bg_augmentor.random_contrast(probability=data_cfg.contrast_rate,
                                        min_factor=data_cfg.contrast_min,
                                        max_factor=data_cfg.contrast_max)

    def gen_srnet_data_with_background(self):
        while True:
            try:
                # --------------------------------------
                # A. 選字與字體
                # --------------------------------------
                font_path = np.random.choice(self.font_list)
                font_name = os.path.basename(font_path)
                font = freetype.Font(font_path)
                font.antialiased = True

                text1 = select_texts(self.texts_list)

                # --------------------------------------
                # B. 字體大小（統計分佈，直接決定渲染尺寸）
                #    canvas 大小由文字內容自然決定，不再限制寬高比
                # --------------------------------------
                font_size = Sampler.log_normal(**data_cfg.font_size_params)

                # --------------------------------------
                # C. 渲染 Mask
                # --------------------------------------
                surf1, text_bbox1, char_bbox1 = render.render_text_by_fontsize(
                    font, text1, font_size, 
                    text_height_ratio = np.random.uniform(data_cfg.text_height_ratio[0], data_cfg.text_height_ratio[1]), 
                    padding_x_ratio = np.random.uniform(data_cfg.padding_x_ratio[0], data_cfg.padding_x_ratio[1])
                )

                # --------------------------------------
                # D. 透視變換
                # --------------------------------------
                pitch = np.random.uniform(data_cfg.pitch_param[0], data_cfg.pitch_param[1])
                yaw = np.random.uniform(data_cfg.yaw_param[0], data_cfg.yaw_param[1])
                roll = np.random.uniform(data_cfg.roll_param[0], data_cfg.roll_param[1])
                focal = np.random.randint(data_cfg.focal_param[0], data_cfg.focal_param[1])
                angle = [pitch, yaw, roll, focal]

                # -----------------------------
                # E. 背景裁切 
                # -----------------------------
                bg_name = random.choice(self.bg_list).strip()
                bg_full_path = os.path.join(self.bg_folder, bg_name)
                if not os.path.exists(bg_full_path): 
                    continue
                    
                text_h, text_w = surf1.shape[:2]
                t_b = cv2.imread(bg_full_path)
                # temp = t_b
                if t_b is None:
                    continue
                
                
                bg_h, bg_w = t_b.shape[:2]
                patch_h = max(15, text_h // np.random.randint(4, 7))
                patch_w = max(30, text_w // np.random.randint(4, 7))
                patch_h = min(patch_h, bg_h - 1)
                patch_w = min(patch_w, bg_w - 1)
                
                # 隨機裁切微小 Patch
                y = np.random.randint(0, bg_h - patch_h)
                x = np.random.randint(0, bg_w - patch_w)
                t_b_patch = t_b[y:y+patch_h, x:x+patch_w]

                # 把這個小 Patch "暴力放大"
                # 使用 INTER_LINEAR 或 INTER_CUBIC 會產生平滑的漸層模糊，完美模擬「不均勻的光照」！
                t_b = cv2.resize(t_b_patch, (text_w, text_h), interpolation=cv2.INTER_CUBIC)

                surf1, t_b, text_bbox1, char_bbox1 = render.perspective(
                    surf1, t_b, angle, text_bbox1, char_bbox1
                )

                # --------------------------------------
                # F. 亮度調整
                # --------------------------------------
                bgs = [[t_b]]  # Augmentor 需要 list of list
                self.bg_augmentor.augmentor_images = bgs
                t_b = self.bg_augmentor.sample(1)[0][0]

                # --------------------------------------
                # G. 顏色控制
                # --------------------------------------
                if np.random.rand() < data_cfg.use_random_color_rate:
                    fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)
                else:
                    fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)


                param = {
                    'is_border': np.random.rand() < data_cfg.is_border_rate,
                    'bordar_color': tuple(np.random.randint(0, 256, 3)),
                    'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
                    'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                                    + data_cfg.shadow_angle_param[0] * np.random.randn(),
                    'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                                    + data_cfg.shadow_shift_param[1, :],
                    'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn()
                                        + data_cfg.shadow_opacity_param[1]
                }
                # --------------------------------------
                # H. 合成
                # --------------------------------------
                _, i_s = colorize.colorize(surf1, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, text_h, param)


                # -----------------------------
                # I. Moderate single-stage degradation
                # -----------------------------
                # 先採樣一個統一的降質強度
                degradation_level = np.random.beta(2, 3)  
                # Beta(2,3): 偏向低降質，均值=0.4，模擬大多數車牌品質尚可

                # Blur：連動 degradation_level
                if np.random.rand() < data_cfg.blur_prob:  # 真實資料約 55% 有明顯模糊
                    sigma = 0.6 + degradation_level * 1.5   # 範圍 0.6~2.1，對應 p10~p90
                    k_w = int(2 * np.ceil(3*sigma) + 1)
                    k_w = max(3, k_w | 1)  # 確保奇數
                    i_s = cv2.GaussianBlur(i_s, (k_w, k_w), sigma)

                # Noise：連動同一個 degradation_level
                if np.random.rand() < data_cfg.noise_prob:  # 噪聲比模糊更普遍
                    noise_std = 2.0 + degradation_level * 12.0  # 範圍 2~14，對應 p10~p90
                    noise = np.random.normal(0, noise_std, i_s.shape).astype(np.float32)
                    i_s = np.clip(i_s.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
                # --------------------------------------
                # J. 骨架
                # --------------------------------------
                t_sk = skeletonization.skeletonization(surf1, 127)

                # --------------------------------------
                # K. 返回結果
                # --------------------------------------
                return [i_s, t_sk, surf1, t_b, fg_col, char_bbox1, text_bbox1, angle, font_name, text1]

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

def enqueue_data(queue, capacity):  
    
    np.random.seed()
    gen = datagen()
    while True:
        try:
            data = gen.gen_srnet_data_with_background()
        except Exception as e:
            pass
        if queue.qsize() < capacity:
            queue.put(data)

class multiprocess_datagen():
    
    def __init__(self, process_num, data_capacity):
        
        self.process_num = process_num
        self.data_capacity =  data_capacity
            
    def multiprocess_runningqueue(self):
        
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        self.pool = multiprocessing.Pool(processes = self.process_num)
        self.processes = []
        for _ in range(self.process_num):
            p = self.pool.apply_async(enqueue_data, args = (self.queue, self.data_capacity))
            self.processes.append(p)
        self.pool.close()
        
    def dequeue_data(self):
        
        while self.queue.empty():
            pass
        data = self.queue.get()
        return data
        '''
        data = None
        if not self.queue.empty():
            data = self.queue.get()
        return data
        '''

    def dequeue_batch(self, batch_size, data_shape):
        
        while self.queue.qsize() < batch_size:
            time.sleep(0.01)

        i_t_batch, i_s_batch = [], []
        t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
        mask_t_batch = []
        
        for i in range(batch_size):
            i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = self.dequeue_data()
            i_t_batch.append(i_t)
            i_s_batch.append(i_s)
            t_sk_batch.append(t_sk)
            t_t_batch.append(t_t)
            t_b_batch.append(t_b)
            t_f_batch.append(t_f)
            mask_t_batch.append(mask_t)
        
        w_sum = 0
        for t_b in t_b_batch:
            h, w = t_b.shape[:2]
            scale_ratio = data_shape[0] / h
            w_sum += int(w * scale_ratio)
        
        to_h = data_shape[0]
        to_w = w_sum // batch_size
        to_w = int(round(to_w / 8)) * 8
        to_size = (to_w, to_h) # w first for cv2
        for i in range(batch_size): 
            i_t_batch[i] = cv2.resize(i_t_batch[i], to_size)
            i_s_batch[i] = cv2.resize(i_s_batch[i], to_size)
            t_sk_batch[i] = cv2.resize(t_sk_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            t_t_batch[i] = cv2.resize(t_t_batch[i], to_size)
            t_b_batch[i] = cv2.resize(t_b_batch[i], to_size)
            t_f_batch[i] = cv2.resize(t_f_batch[i], to_size)
            mask_t_batch[i] = cv2.resize(mask_t_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            # eliminate the effect of resize on t_sk
            t_sk_batch[i] = skeletonization.skeletonization(mask_t_batch[i], 127)

        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)
        t_sk_batch = np.expand_dims(np.stack(t_sk_batch), axis = -1)
        t_t_batch = np.stack(t_t_batch)
        t_b_batch = np.stack(t_b_batch)
        t_f_batch = np.stack(t_f_batch)
        mask_t_batch = np.expand_dims(np.stack(mask_t_batch), axis = -1)
        
        i_t_batch = i_t_batch.astype(np.float32) / 127.5 - 1. 
        i_s_batch = i_s_batch.astype(np.float32) / 127.5 - 1. 
        t_sk_batch = t_sk_batch.astype(np.float32) / 255. 
        t_t_batch = t_t_batch.astype(np.float32) / 127.5 - 1. 
        t_b_batch = t_b_batch.astype(np.float32) / 127.5 - 1. 
        t_f_batch = t_f_batch.astype(np.float32) / 127.5 - 1.
        mask_t_batch = mask_t_batch.astype(np.float32) / 255.
        
        return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]
    
    def get_queue_size(self):
        
        return self.queue.qsize()
    
    def terminate_pool(self):
        
        self.pool.terminate()
