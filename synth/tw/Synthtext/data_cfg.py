"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""
import numpy as np

# font
font_dir = 'datasets/fonts/LP2022_2'   # TWGen7_V1.ttf, the Taiwan plate face
# standard_font_path = './datasets/fonts/LP2022_2/TWGen7_V1.ttf'

# text
text_filepath = 'data/plate_list_final2.txt'        # LP-2025 stage 1
#text_filepath = 'data/ccpd_plate_list.txt'        # CCPD

# background
bg_filepath = './datasets/imnames.cp'
temp_bg_path = './datasets/bg_data/bg_img/'

# color
color_filepath = 'data/colors_new.cp'


# --- 1. 幾何參數 (Geometry) ---
# 字體大小分佈（單位：pixel，直接控制渲染字高）
# Gamma 分佈：大多落在 40~120px，長尾可到 200px
# shape=2.5, scale=28 => mean ~ 70px, 符合真實車牌字高分佈
# font_size_params = {'shape': 2.5, 'scale': 28.0, 'offset': 30, 'min_val': 30, 'max_val': 200}
font_size_params = {
    'mean_log': 4.65, 
    'std_log': 0.45, 
    'offset': 0, 
    'min_val': 48,   #[關鍵防禦] 絕對不生成低於 48px 的字，避免 VAE 崩壞
    'max_val': 280   # 限制最大值，避免 OOM 或超大特寫失去上下文
}
# text_height_ratio = [0.27, 0.45]
text_height_ratio = [0.45, 0.65]
# padding_x_ratio   = [0.02, 0.08]
padding_x_ratio = [0.03, 0.07]


# --- 2. 光照與顏色 (Appearance) ---
# brightness_params = {'mean': 136, 'std': 32, 'min_val': 60, 'max_val': 190}
# bg_color_shift = [-15, 15]
# bg_contrast_range = [0.9, 1.1]
## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.3
color_rate = 0.8
color_min =0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3
use_random_color_rate = 0.5

# Blur & Noise
blur_prob = 0.55
#blur_params = {'shape': 2.0, 'scale': 0.5, 'offset': 0.5, 'min_val': 0.5, 'max_val': 3.0}

# resize_prob = 0.65

noise_prob = 0.65
#noise_params = {'shape': 2.0, 'scale': 1.9, 'offset': 2.0, 'min_val': 2.0, 'max_val': 22.0}

# jpeg_prob = 0.15
# jpeg_quality_range = [85, 95]

# perspective
pitch_param = [-13, 13]
yaw_param   = [-18, 18]
roll_param = [-8, 8]
focal_param = [350, 750] # scale, shift for np.random.randn() [0.0, 1]

is_border_rate = 0
is_shadow_rate = 0
shadow_angle_degree = [1, 3, 5, 7] # shift for shadow_angle_param
shadow_angle_param = [0.5, None] # scale, shift for np.random.randn()
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32) # scale, shift for np.random.randn()
shadow_opacity_param = [0.1, 0.5] # shift for shadow_angle_param
