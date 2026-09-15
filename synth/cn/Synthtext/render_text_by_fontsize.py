# -*- coding: utf-8 -*-
"""
Rendering text mask with Container-First approach.
Modified for SRNet data generation.
"""

import cv2
import numpy as np
import pygame
from pygame import freetype
from math import pi

# ---------------------------------------------------------------------
# 1. 核心渲染函數：直接用指定字體大小渲染，canvas 由文字內容決定
# ---------------------------------------------------------------------
def render_text_by_fontsize(font, text, font_size, 
                             text_height_ratio,  # 字高佔canvas高的比例
                             padding_x_ratio):
    font.size = int(font_size)
    fixed_w = font.get_rect('M').width
    ref_h = font.get_rect('M').height

    char_spacing = np.random.normal(1.08, 0.02)
    total_text_w = int(fixed_w * char_spacing * len(text))
    padding_x = int(total_text_w * padding_x_ratio)
    surf_w = total_text_w + 2 * padding_x

    # 用比例反推canvas高度，而不是用ref_h加padding
    surf_h = int(ref_h / text_height_ratio)

    surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA, 32)

    char_bbox_list = []
    cursor_x = padding_x

    for ch in text:
        if ch == '-':
            ch = "·"
        char_rect = font.get_rect(ch)
        char_h = char_rect.height

        char_surf = pygame.Surface((fixed_w, char_h), pygame.SRCALPHA)
        font.render_to(char_surf, (0, 0), ch, fgcolor=(255, 255, 255))

        # 垂直置中
        y = (surf_h - char_h) // 2
        surf.blit(char_surf, (cursor_x, y))

        # bbox 用「實際字寬」而非固定的 'M' 寬, 避免窄字(如 '1')框過寬/越界;
        # cursor_x 仍用 fixed_w 前進, 保持等距排版。
        gw = max(char_rect.width, 1)
        lt = [cursor_x,      y]
        rt = [cursor_x + gw, y]
        rb = [cursor_x + gw, y + char_h]
        lb = [cursor_x,      y + char_h]
        char_bbox_list.append([lt, rt, rb, lb])

        cursor_x += int(fixed_w * char_spacing)

    # 轉 numpy
    mask_arr = pygame.surfarray.pixels_alpha(surf).swapaxes(0, 1)

    # 裁切實際有像素的區域
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        final_canvas = np.zeros((surf_h, surf_w), dtype=np.uint8)
        text_bbox = [[0,0],[surf_w,0],[surf_w,surf_h],[0,surf_h]]
        return final_canvas, text_bbox, char_bbox_list

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    char_h_actual = y_max - y_min
    char_w_actual = x_max - x_min

    # 用字高反推總canvas應有的高度
    target_canvas_h = int(char_h_actual / text_height_ratio)
    target_canvas_w = int(char_w_actual / (1 - 2 * padding_x_ratio))

    # 置中字在新canvas
    pad_y = (target_canvas_h - char_h_actual) // 2
    pad_x = (target_canvas_w - char_w_actual) // 2

    final_canvas = np.zeros((target_canvas_h, target_canvas_w), dtype=np.uint8)
    src = mask_arr[y_min:y_max+1, x_min:x_max+1]
    final_canvas[pad_y:pad_y+src.shape[0], pad_x:pad_x+src.shape[1]] = src

    H, W = final_canvas.shape[:2]
    text_bbox = [[0, 0], [W, 0], [W, H], [0, H]]

    # 更新字符 bbox（偏移裁切量）並 clip 進 canvas 範圍, 防止越界
    scaled_char_bbox = []
    for bbox_ch in char_bbox_list:
        new_bbox = [[min(max(x - x_min + pad_x, 0), W),
                     min(max(y - y_min + pad_y, 0), H)] for x, y in bbox_ch]
        scaled_char_bbox.append(new_bbox)
    return final_canvas, text_bbox, scaled_char_bbox

# ---------------------------------------------------------------------
# 2. 透視變換輔助函數
# ---------------------------------------------------------------------
def get_M(img_h, img_w, focal, theta, phi, gamma, dx, dy, dz):
    w = img_w
    h = img_h
    f = focal
    A1 = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1], [0, 0, 1]])
    RX = np.array([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    RY = np.array([[np.cos(phi),0,-np.sin(phi),0],[0,1,0,0],[np.sin(phi),0,np.cos(phi),0],[0,0,0,1]])
    RZ = np.array([[np.cos(gamma),-np.sin(gamma),0,0],[np.sin(gamma),np.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]])
    R = np.dot(np.dot(RX, RY), RZ)
    T = np.array([[1,0,0,dx],[0,1,0,dy],[0,0,1,dz],[0,0,0,1]])
    A2 = np.array([[f,0,w/2,0],[0,f,h/2,0],[0,0,1,0]])
    return np.dot(A2, np.dot(T, np.dot(R, A1)))

def get_transform_matrix(img_h, img_w, theta, phi, gamma, focal):
    rtheta = theta * pi / 180.0
    rphi = phi * pi / 180.0
    rgamma = gamma * pi / 180.0
    dx, dy, dz = 0, 0, focal
    return get_M(img_h, img_w, focal, rtheta, rphi, rgamma, dx, dy, dz)

def bbox_transform(M, char_bbox_list):
    if not char_bbox_list:
        return []
    arr = np.array(char_bbox_list).T
    ones = np.ones((1, arr.shape[1]))
    points_homo = np.vstack([arr, ones])
    trans_points = M.dot(points_homo)
    trans_points /= trans_points[2,:] + 1e-10
    return trans_points[:2,:].T.astype(int).tolist()

# ---------------------------------------------------------------------
# 3. 透視與 Padding
# ---------------------------------------------------------------------
def warpPerspectivePadded(src, M, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    h, w = src.shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float32).T
    corners = np.vstack((corners,np.ones((1,4))))
    new_corners = M.dot(corners)
    new_corners /= new_corners[2,:]
    min_x, max_x = int(np.min(new_corners[0])), int(np.max(new_corners[0]))
    min_y, max_y = int(np.min(new_corners[1])), int(np.max(new_corners[1]))
    anchor_x, anchor_y = -min_x, -min_y
    T = np.array([[1,0,anchor_x],[0,1,anchor_y],[0,0,1]])
    final_M = T.dot(M)
    out_w, out_h = max_x-min_x, max_y-min_y
    warped = cv2.warpPerspective(src, final_M, (out_w,out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=borderMode,
                                 borderValue=borderValue)
    return warped, final_M

def perspective(mask, bg, angle, text_bbox, char_bbox_list):
    """
    Apply perspective transform to both mask and bg.
    bg is resized to match mask size before transform.
    """
    pitch, yaw, roll, focal = angle
    h, w = mask.shape[:2]

    M = get_transform_matrix(h, w, pitch, yaw, roll, focal)

    warped_mask, final_M = warpPerspectivePadded(mask, M)
    warped_bg, _         = warpPerspectivePadded(bg, M, borderMode=cv2.BORDER_REFLECT)

    # 更新 text_bbox
    coords = np.argwhere(mask > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        ori_bbox = [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
        new_text_bbox = bbox_transform(final_M, ori_bbox)
    else:
        new_text_bbox = [[0,0],[0,0],[0,0],[0,0]]

    new_char_bbox_list = [bbox_transform(final_M, bbox) for bbox in char_bbox_list]

    return warped_mask, warped_bg, new_text_bbox, new_char_bbox_list
