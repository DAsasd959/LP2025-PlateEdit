import os
import os.path as osp
import json

import torch
import torch.nn.functional as F

from .base_model import ResNet, _build_vision_encode, _dtype_func


def get_param(state_dict):
    backbone_dict = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('module.visual.') and 'fpn_head' not in k:
            new_k = k[len('module.visual.'):]
            print(new_k, v.shape)
            backbone_dict[new_k] = v
    return backbone_dict

def convert_param_name(state_dict):
    backbone_dict = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('module.visual.') and 'fpn_head' not in k and 'attnpool' not in k:
            new_k = k[len('module.visual.'):]
            # update parameter name in stem layers
            if new_k.startswith('conv1'):
                new_k = new_k.replace('conv1', 'stem.0')
            elif new_k.startswith('bn1'):
                new_k = new_k.replace('bn1', 'stem.1')
            elif new_k.startswith('conv2'):
                new_k = new_k.replace('conv2', 'stem.3')
            elif new_k.startswith('bn2'):
                new_k = new_k.replace('bn2', 'stem.4')
            elif new_k.startswith('conv3'):
                new_k = new_k.replace('conv3', 'stem.6')
            elif new_k.startswith('bn3'):
                new_k = new_k.replace('bn3', 'stem.7')
            
            # update parameter name in bottleneck blocks
            new_k = new_k.replace('downsample.1', 'downsample.2').replace('downsample.0', 'downsample.1')
            new_k = 'backbone.' + new_k

            print(new_k, v.shape)
            backbone_dict[new_k] = v
    return backbone_dict

def convert_param_name_reverse(state_dict):
    backbone_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_k = k[len('backbone.'):]
            # update parameter name in stem layers
            if new_k.startswith('stem.0'):
                new_k = new_k.replace('stem.0', 'conv1')
            elif new_k.startswith('stem.1'):
                new_k = new_k.replace('stem.1', 'bn1')
            elif new_k.startswith('stem.3'):
                new_k = new_k.replace('stem.3', 'conv2')
            elif new_k.startswith('stem.4'):
                new_k = new_k.replace('stem.4', 'bn2')
            elif new_k.startswith('stem.6'):
                new_k = new_k.replace('stem.6', 'conv3')
            elif new_k.startswith('stem.7'):
                new_k = new_k.replace('stem.7', 'bn3')
            
            # update parameter name in bottleneck blocks
            new_k = new_k.replace('downsample.1', 'downsample.0').replace('downsample.2', 'downsample.1')

            print(new_k, v.shape)
            backbone_dict[new_k] = v
    return backbone_dict

class ODMLoss:
    def __init__(
        self,
        modelpath,
        w_loss_f=0.7,
        w_loss_1=0.1,
        w_loss_2=0.2,
        w_loss_3=0.3,
        w_loss_4=0.4,
        input_resolution=None,
    ):
        self.w_loss_f = w_loss_f
        self.w_loss_1 = w_loss_1
        self.w_loss_2 = w_loss_2
        self.w_loss_3 = w_loss_3
        self.w_loss_4 = w_loss_4

        current_directory = os.path.dirname(__file__)
        odm_json_file_path = os.path.join(current_directory, 'base_model/ODM.json')

        with open(odm_json_file_path, 'r') as f:
            ODM_model_info = json.load(f)
            embed_dim = ODM_model_info["embed_dim"]
            vision_cfg = ODM_model_info["vision_cfg"]
            if input_resolution is not None:
                vision_cfg['image_resolution'] = input_resolution
            vision_heads = vision_cfg['vision_width'] * 32 // 64
            self.ResNet1 = ResNet(
                layers=vision_cfg['vision_layers'],
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=vision_cfg['image_resolution'],
                width=vision_cfg['vision_width'],
        )
        self.ResNet1.requires_grad_(False).eval()
        self.dtype = _dtype_func(self.ResNet1)
        state_dict = torch.load(modelpath)
        new_state_dict = get_param(state_dict)
        # self.ResNet1.load_state_dict(new_state_dict, strict=True)
        post_state_dict = {}
        for key in new_state_dict:
            if not key.startswith('attnpool'):
                post_state_dict[key] = new_state_dict[key]
        self.ResNet1.load_state_dict(post_state_dict, strict=False)

    def loss(self, image_pred, imgs, mask=None, return_dict=False):
        pred_feature, p_att = self.ResNet1(image_pred.type(self.dtype))
        px1 = pred_feature[0]
        px2 = pred_feature[1]
        px3 = pred_feature[2]
        px4 = pred_feature[3]
        p_encoded_image = pred_feature[-1]
        p_encoded_image = p_encoded_image[1:]

        feature, att = self.ResNet1(imgs.type(self.dtype))
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        encoded_image = feature[-1]
        encoded_image = encoded_image[1:]

        if mask is not None:
            mask1 = F.interpolate(mask, scale_factor=0.25, mode='nearest')
            mask2 = F.interpolate(mask, scale_factor=0.125, mode='nearest')
            mask3 = F.interpolate(mask, scale_factor=0.0625, mode='nearest')
            mask4 = F.interpolate(mask, scale_factor=0.03125, mode='nearest')
            loss_x1 = torch.nn.functional.mse_loss(px1 * mask1, x1 * mask1, reduction="mean")
            loss_x2 = torch.nn.functional.mse_loss(px2 * mask2, x2 * mask2, reduction="mean")
            loss_x3 = torch.nn.functional.mse_loss(px3 * mask3, x3 * mask3, reduction="mean")
            loss_x4 = torch.nn.functional.mse_loss(px4 * mask4, x4 * mask4, reduction="mean")
            # loss_f = torch.nn.functional.mse_loss(p_encoded_image * mask, encoded_image * mask, reduction="mean")
            final_loss = self.w_loss_1 * loss_x1 + self.w_loss_2 * loss_x2 + self.w_loss_3 * loss_x3 + self.w_loss_4 * loss_x4
        else:
            loss_x1 = torch.nn.functional.mse_loss(px1, x1, reduction="mean")
            loss_x2 = torch.nn.functional.mse_loss(px2, x2, reduction="mean")
            loss_x3 = torch.nn.functional.mse_loss(px3, x3, reduction="mean")
            loss_x4 = torch.nn.functional.mse_loss(px4, x4, reduction="mean")
            loss_f = torch.nn.functional.mse_loss(p_encoded_image, encoded_image, reduction="mean")

            final_loss = self.w_loss_f * loss_f + self.w_loss_1 * loss_x1 + self.w_loss_2 * loss_x2 + self.w_loss_3 * loss_x3 + self.w_loss_4 * loss_x4
        if return_dict:
            return final_loss, {'final_loss': final_loss, 'loss_x1': loss_x1, 'loss_x2': loss_x2, 'loss_x3': loss_x3, 'loss_x4': loss_x4}
        return final_loss

if __name__ == '__main__':
    ODMLoss('/mnt/xmap_nas_ml/rlan/python_project/ODM_weights/epoch_100.pt')