# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""

import os
import cv2
import math
import numpy as np
import pygame
from pygame import freetype
import random
import multiprocessing
import queue
import Augmentor

from . import render_text_mask_lp2022 as render_text_mask
from . import colorize
from . import skeletonization
from . import render_standard_text_LP2022 as render_standard_text
from . import data_cfg_lp2022 as data_cfg
import pickle as cp

def random_plate():
    plate_list = []
    letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'

    numbers = '0123456789'
    # numbers_wo0 = '123456789'

    def get_plate_7num():
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        #generate 4 randomly chosen numbers, N1, N2, N3, N4
        for i in range(7):
            text_choice = random.choice(['char','num'])
            if text_choice == 'char':
                text = random.choice(letters)
            else:
                text = random.choice(numbers)
            plate.append(text)
            if i == 2:
                plate.append('·')
        plate = ''.join(plate)

        return plate


    def get_plate_6num():
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        insert = np.random.randint(1,4)
        for i in range(6):
            text_choice = random.choice(['char','num'])
            if text_choice == 'char':
                text = random.choice(letters)
            else:
                text = random.choice(numbers)
            plate.append(text)
            if i == insert:
                plate.append('·')
        plate = ''.join(plate)

        return plate

    def get_plate_5num(): # 針對舊式車牌(AB-001)
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        insert = np.random.randint(1,3)
        for i in range(5):
            text_choice = random.choice(['char','num'])
            if text_choice == 'char':
                text = random.choice(letters)
            else:
                text = random.choice(numbers)
            plate.append(text)
            if i == insert:
                plate.append('·')
        plate = ''.join(plate)

        return plate

    def get_plate_4num(): # 針對舊式車牌(00-BA)
        #generate 3 randomly chosen letters, L1, L2, L3
        plate = []
        # insert = np.random.randint(2,3)
        for i in range(4):
            text_choice = random.choice(['char','num'])
            if text_choice == 'char':
                text = random.choice(letters)
            else:
                text = random.choice(numbers)
            plate.append(text)
            if i == 1:
                plate.append('·')
        plate = ''.join(plate)

        return plate

    random_return_num = random.choice([4,5,6,7])  # 隨機選一個return

    if (random_return_num == 4):
        return  get_plate_4num() #針對舊式車牌(01-BA)
    elif (random_return_num == 5):  
        return  get_plate_5num()    #針對新式車牌(BA-001),(001-BA)
    elif (random_return_num == 6): 
        return  get_plate_6num()    #針對舊式車牌(AB0-001),(AB-0001),(0001-BA)
    elif (random_return_num == 7):  
        return  get_plate_7num() #針對舊式車牌(ABC-0001)

       

class datagen():

    def __init__(self):
        
        freetype.init()
        cur_file_path = os.path.dirname(__file__)
        
        font_dir = os.path.join(cur_file_path, data_cfg.font_dir)
        self.font_list = os.listdir(font_dir)
        self.font_list = [os.path.join(font_dir, font_name) for font_name in self.font_list]
        
        self.standard_font_path = os.path.join(cur_file_path, data_cfg.standard_font_path)
        
        color_filepath = os.path.join(cur_file_path, data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)
        
        text_filepath = os.path.join(cur_file_path, data_cfg.text_filepath)
        with open(text_filepath, 'r') as fw:
            self.text_list = fw.readlines()

        self.text_list = [text.strip('\n') for text in self.text_list]
        # print(self.text_list)

        
        bg_filepath = os.path.join(cur_file_path, data_cfg.bg_filepath)

        with open(bg_filepath, 'rb') as f:
	        self.bg_list = set(cp.load(f))
            
        self.bg_list = [data_cfg.temp_bg_path+img_path.strip() for img_path in self.bg_list]
        
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = data_cfg.elastic_rate,
            grid_width = data_cfg.elastic_grid_size, grid_height = data_cfg.elastic_grid_size,
            magnitude = data_cfg.elastic_magnitude)
        
        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(probability = 1, 
            min_factor = data_cfg.brightness_min, max_factor = data_cfg.brightness_max)
        self.bg_augmentor.random_color(probability = data_cfg.color_rate, 
            min_factor = data_cfg.color_min, max_factor = data_cfg.color_max)
        self.bg_augmentor.random_contrast(probability = data_cfg.contrast_rate, 
            min_factor = data_cfg.contrast_min, max_factor = data_cfg.contrast_max)

    def append_dash(text):
        if len(text)<=5:
            loc_v = [2,-2]
            loc = np.random.choice(loc_v,1)
            text = text.insert(loc,)
     
 

    def gen_srnet_data_with_background(self):
        
        while True:
            # choose font, text and bg
            font = np.random.choice(self.font_list)
            # print(font)
            #text1, text2 = np.random.choice(self.text_list), np.random.choice(self.text_list)
            text1 = 'PTR0·Z2'
            # text1 = random_plate()
            # text2 = random_plate()
            text2 = 'PTR0·Z2'

            
            '''

            # text1,text2 = ''.join(text1), ''.join(text2)
            
            
            # region_list =["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

            region_list = ['\u7696','\u6CAA','\u6D25','\u6E1D','\u5180','\u664B','\u8499','\u8FBD','\u5409',"\u9ED1","\u82CF","\u6D59",'\u4EAC','\u95FD','\u8D63','\u9C81',\
                "\u8C6B","\u9102","\u6E58","\u6FB3","\u6842","\u743C",'\u5DDD','\u8D35','\u4E91','\u85CF','\u9655','\u7518','\u9752','\u5B81','\u65B0','\u8B66','\u5B66','\u004F']

            # alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z', 'O']

            alphabet = ['\u0041','\u0042','\u0043','\u0044','\u0045','\u0046','\u0047','\u0048','\u004A','\u004B','\u004C','\u004D','\u004E','\u0050','\u0051','\u0052','\u0053','\u0054','\u0055','\u0056','\u0057','\u0058','\u0059','\u005A','\u004F']

            # character = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
            character = ['\u0041','\u0042','\u0043','\u0044','\u0045','\u0046','\u0047','\u0048','\u004A','\u004B','\u004C','\u004D','\u004E','\u0050','\u0051','\u0052','\u0053','\u0054','\u0055','\u0056','\u0057','\u0058','\u0059','\u005A','\u0030',\
                '\u0031','\u0032','\u0033','\u0034','\u0035','\u0036','\u0037','\u0038','\u0039','\u004F']
            
            character_list = [np.random.choice(character) for i in range(5)]
            character_str = ''.join(character_list).upper()
            character_list2 = [np.random.choice(character) for i in range(5)]
            character_str2 = ''.join(character_list2).upper()

            text1 = str(np.random.choice(region_list) + np.random.choice(alphabet) +"\u00B7" +character_str)
            text2 = str(np.random.choice(region_list) + np.random.choice(alphabet) +"\u00B7"  +character_str2)
            '''
            
            # character = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            # alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z', 'O']
            # word_num = np.random.choice(np.arange(4,8),1)
            # text1 = np.ndarray.tolist(np.reshape(np.random.choice(character,(1,int(word_num))),-1))
            # text2 = np.ndarray.tolist(np.reshape(np.random.choice(character,(1,int(word_num))),-1))

            bg = cv2.imread(random.choice(self.bg_list))

            # init font
            font_path = font
            font = freetype.Font(font)
            font.antialiased = True
            font.origin = True
            font.fixed_width
            # choose font style
            font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
            # font.underline = np.random.rand() < data_cfg.underline_rate
            font.strong = np.random.rand() < data_cfg.strong_rate
            font.oblique = np.random.rand() < data_cfg.oblique_rate
            # if font.size > data_cfg.font_size_gate:
            #     font.strength = data_cfg.strength_rate
            
            
            # render text to surf
            param = {
                        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() 
                                      + data_cfg.curve_rate_param[1],
                        'curve_center': np.random.randint(0, len(text1)),
                        # 'padding': np.random.randint(0, 1)
                    }
            surf1, bbs1, char_bbox_list1 = render_text_mask.render_text(font, text1) # render_text_mask: text -> array: surf1(i_s)
            
            param['curve_center'] = int(param['curve_center'] / len(text1) * len(text2))



            surf2, bbs2, char_bbox_list2 = render_text_mask.render_text(font, text2) # render_text_mask: text -> array: surf1(mask)
            

            text_list = []
            mask_surf_list = []
            n_char_bbox_list = []
            bbs_list= []
            for i in range(20):
                text = random_plate()
                surf_n, bbs_n, char_bbox_list_n  = render_text_mask.render_text(font, text)
                mask_surf_list.append(surf_n)
                text_list.append(text)
                n_char_bbox_list.append(char_bbox_list_n)
                bbs_list.append(bbs_n)


            # text1 = source_text

            # get padding
            padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
            padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
            padding = np.hstack((padding_ud, padding_lr))

            # perspect the surf
            rotate = np.random.choice(np.arange(data_cfg.rotate_param[0], data_cfg.rotate_param[1]))
            # zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
            shear = data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1]
            random_pers = np.random.randn(2)
            # perspect = data_cfg.perspect_param[0] * random_pers + data_cfg.perspect_param[1]
            Pitch = np.random.randint(data_cfg.rotate_param[0], data_cfg.rotate_param[1]+1)
            Yaw = np.random.randint(data_cfg.rotate_param[0], data_cfg.rotate_param[1]+1)
            Roll = np.random.randint(data_cfg.rotate_param[0], data_cfg.rotate_param[1]+1)
            focal = np.random.randint(data_cfg.focal_param[0], data_cfg.focal_param[1]+1)
            # theta = 0
            # phi = 50 
            # gamma = 0 #row
            # surf1 = render_text_mask.perspective(surf1, rotate, zoom, shear, perspect, padding) # w first
            # surf2 = render_text_mask.perspective(surf2, rotate, zoom, shear, perspect, padding) # w first
            # angle = render_text_mask.angle(surf1,surf2,[Pitch,Yaw,Roll])
            # angle = [Pitch,Yaw,Roll,focal]
            angle = [30,40,45,2000]
            

            surf1,text_bbox1,char_bbox1 = render_text_mask.perspective(surf1, angle, padding, char_bbox_list1,text1) # w first
            surf2,text_bbox2,char_bbox2 = render_text_mask.perspective(surf2, angle, padding, char_bbox_list2,text2)
            
            text_bbox_list = []
            char_bbox_list = []
            mask_per_surf_list=[]
            for text_n, surf_n, char_bbox_list_n in zip(text_list, mask_surf_list, n_char_bbox_list):
                surf_n, text_bbox_n, char_bbox_n = render_text_mask.perspective(surf_n, angle, padding, char_bbox_list_n, text_n)
                mask_per_surf_list.append(surf_n)
                text_bbox_list.append(text_bbox_n)
                char_bbox_list.append(char_bbox_n)

            surf_h_n = []
            surf_w_n = []
            
            for mask_per_surf_n in mask_per_surf_list:
                surf_h, surf_w = mask_per_surf_n.shape[:2]
                surf_h_n.append(surf_h) 
                surf_w_n.append(surf_w) 



            # choose a background
            surf1_h, surf1_w = surf1.shape[:2]
            surf2_h, surf2_w = surf2.shape[:2]
            surf_h_n.append(surf1_h) 
            surf_h_n.append(surf2_h) 
            surf_w_n.append(surf1_w) 
            surf_w_n.append(surf2_w) 

            surf_w = max(surf_w_n)
            surf_h = max(surf_h_n)


            # surf_h = max(surf1_h, surf2_h)
            # surf_w = max(surf1_w, surf2_w)

            # center2size: find all char ceneter, to make sure differnt text have same text center
            surf1,text_bbox1,char_bbox1 = render_text_mask.center2size(surf1, (surf_h, surf_w),text_bbox1,char_bbox1)
            surf2,text_bbox2,char_bbox2 = render_text_mask.center2size(surf2, (surf_h, surf_w),text_bbox2,char_bbox2)
            
            final_surf_list = []
            for surf_n,text_bbox_n,char_bbox_n in zip(mask_per_surf_list,text_bbox_list,char_bbox_list):
                mask_per_surf_center_n,text_bbox_n,char_bbox_n = render_text_mask.center2size(surf_n, (surf_h, surf_w),text_bbox_n,char_bbox_n)
                final_surf_list.append(mask_per_surf_center_n)


            bg_h, bg_w = bg.shape[:2]
            if bg_w < surf_w or bg_h < surf_h:
                continue

            x = np.random.randint(0, bg_w - surf_w + 1)
            y = np.random.randint(0, bg_h - surf_h + 1)
            t_b = bg[y:y+surf_h, x:x+surf_w, :]

          
            # augment surf
            # surfs = [[surf1, surf2]]
            # self.surf_augmentor.augmentor_images = surfs
            # surf1, surf2 = self.surf_augmentor.sample(1)[0]

            surfs = [[surf1, surf2]+final_surf_list]
            self.surf_augmentor.augmentor_images = surfs
            surfs_n = self.surf_augmentor.sample(1)[0]
            surf1, surf2 = surfs_n[:2]
            final_surf_list = surfs_n[2:]
            # bg augment
            bgs = [[t_b]]

            self.bg_augmentor.augmentor_images = bgs
            
             
            t_b = self.bg_augmentor.sample(1)[0][0]

            # 
            # render standard text

            i_t = render_standard_text.make_standard_text(font_path, text2, (surf_h, surf_w))
            

            # get min h of bbs

            min_h_n_list = [np.min(bbs_n[:, 3]) for bbs_n in bbs_list]

            min_h1 = np.min(bbs1[:, 3])
            
            min_h2 = np.min(bbs2[:, 3])
            min_h_n_list.append(min_h1)
            min_h_n_list.append(min_h2)

            # min_h = min(min_h1, min_h2)
            min_h = min(min_h_n_list)
            
            # get font color
            if np.random.rand() < data_cfg.use_random_color_rate:
                fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)

            else:
                fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)

            t_t2 = render_standard_text.make_standard_text2(self.standard_font_path, text2, (64, 128),color = fg_col)


            # colorful the surf and conbine foreground and background
            param = {
                        'is_border': np.random.rand() < data_cfg.is_border_rate,
                        'bordar_color': tuple(np.random.randint(0,256,3)),
                        'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
                        'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                                        + data_cfg.shadow_angle_param[0] * np.random.randn(),
                        'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                                        + data_cfg.shadow_shift_param[1, :],
                        'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn()
                                          + data_cfg.shadow_opacity_param[1]
                    }
            _, i_s = colorize.colorize(surf1, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            t_t, t_f = colorize.colorize(surf2, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            
            # skeletonization
            t_sk = skeletonization.skeletonization(surf2, 127)
            break
   
        return [i_t, i_s, t_sk, t_t,t_t2, t_b, t_f, surf1, surf2, final_surf_list, text1, text2,text_list,[text_bbox1,char_bbox1],[text_bbox2,char_bbox2],angle]

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
        
        self.process_num = 1
        self.data_capacity = 1
            
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
            pass

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
