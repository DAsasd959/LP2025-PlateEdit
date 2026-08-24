import cv2
import numpy as np
from easydict import EasyDict as edict
import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
from skimage.transform._geometric import _umeyama as get_sym_mat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../eval')))
from recognizer import TextRecognizer, create_predictor

PRINT_DEBUG = False

def min_bounding_rect(img):
    # print(img.dtype, img.shape)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('Bad contours, using fake bbox...')
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # sort
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])

def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0]-pts1[1]), np.linalg.norm(pts1[2]-pts1[3]))
    height = max(np.linalg.norm(pts1[0]-pts1[3]), np.linalg.norm(pts1[1]-pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True).to(img.dtype)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, :int(height), :int(width)]
    return result

def crop_image(src_img, mask):
    # import pdb
    # pdb.set_trace()
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result]*3, axis=-1)
    return result


class OCRLoss:
    def __init__(
        self,
        rec_model_dir,
        rec_char_dict_path,
        device,
        dtype,
        loss_alpha = 1,
        loss_beta = 1,
        latin_weight = 1,
        loss_type = 'l2',
    ):

        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.latin_weight = latin_weight
        self.device = device
        self.dtype = dtype

        # text_predictor = create_predictor(rec_model_dir).to(device, dtype=dtype).eval()
        text_predictor = create_predictor(rec_model_dir).to(device).eval()
        args_ocr = edict()
        args_ocr.rec_image_shape = "3, 48, 320"
        args_ocr.rec_batch_num = 6
        args_ocr.rec_char_dict_path = rec_char_dict_path
        args_ocr.use_fp16 = False
        self.cn_recognizer = TextRecognizer(args_ocr, text_predictor)
        # self.cn_recognizer.predictor.to(device).to(dtype)
        self.cn_recognizer.predictor.to(device)
        for param in text_predictor.parameters():
                param.requires_grad = False

        self.loss_type = loss_type

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def loss(self, image_pred, imgs, batch):
        bsz = image_pred.shape[0]
        bs_ocr_loss = []
        bs_ctc_loss = []

        lang_weight = []
        gt_texts = []
        x0_texts = []
        x0_texts_ori = []

        for i in range(bsz):
            n_lines = batch['n_lines'][i]  # batch size
            for j in range(n_lines):  # line
                lang = batch['language'][j][i]
                if lang == 'Chinese':
                    lang_weight += [1.0]
                elif lang == 'Latin':
                    lang_weight += [self.latin_weight]
                else:
                    lang_weight += [1.0]  # unsupport language, TODO
                gt_texts += [batch['texts'][j][i]]
                pos = batch['positions'][j][i]*255.
                # import pdb
                # pdb.set_trace()
                # pos = rearrange(pos, 'c h w -> h w c')
                np_pos = pos.detach().cpu().numpy().astype(np.uint8)
                x0_text = crop_image(image_pred[i], np_pos)
                x0_texts += [x0_text]
                x0_text_ori = crop_image(imgs[i], np_pos)
                x0_texts_ori += [x0_text_ori]

        if len(x0_texts) > 0:
            x0_list = x0_texts + x0_texts_ori
            x0_list = [x.to(imgs.dtype) for x in x0_list]
            # import pdb
            # pdb.set_trace()
            # preds shape: torch.Size([len(x0_list), 40, 6625])
            # preds_neck shape: torch.Size([len(x0_list), 40, 64])
            preds, preds_neck = self.cn_recognizer.pred_imglist(x0_list, show_debug=PRINT_DEBUG, norm=False)
            n_pairs = len(preds)//2
            preds_decode = preds[:n_pairs]  # preds_decode shape: torch.Size([len(x0_list) // 2, 40, 6625])
            preds_ori = preds[n_pairs:]     # preds_ori shape: torch.Size([len(x0_list) // 2, 40, 6625])
            preds_neck_decode = preds_neck[:n_pairs]    # preds_neck_decode shape: torch.Size([len(x0_list) // 2, 40, 64])
            preds_neck_ori = preds_neck[n_pairs:]   # preds_neck_ori shape: torch.Size([len(x0_list) // 2, 40, 64])
            lang_weight = torch.tensor(lang_weight).to(preds_neck.device)   # lang_weight shape: torch.Size([len(x0_list) // 2])
            # split to batches
            bs_preds_decode = []
            bs_preds_ori = []
            bs_preds_neck_decode = []
            bs_preds_neck_ori = []
            bs_lang_weight = []
            bs_gt_texts = []
            n_idx = 0
            for i in range(bsz):  # sample index in a batch
                n_lines = batch['n_lines'][i]
                bs_preds_decode += [preds_decode[n_idx:n_idx+n_lines]]
                bs_preds_ori += [preds_ori[n_idx:n_idx+n_lines]]
                bs_preds_neck_decode += [preds_neck_decode[n_idx:n_idx+n_lines]]
                bs_preds_neck_ori += [preds_neck_ori[n_idx:n_idx+n_lines]]
                bs_lang_weight += [lang_weight[n_idx:n_idx+n_lines]]
                bs_gt_texts += [gt_texts[n_idx:n_idx+n_lines]]
                n_idx += n_lines
            # calc loss
            ocr_loss_debug = []
            ctc_loss_debug = []

            for i in range(bsz):
                if len(bs_preds_neck_decode[i]) > 0:
                    if self.loss_alpha > 0:
                        sp_ocr_loss = self.get_loss(bs_preds_neck_decode[i], bs_preds_neck_ori[i], mean=False).mean([1, 2])
                        sp_ocr_loss *= bs_lang_weight[i]  # weighted by language
                        bs_ocr_loss += [sp_ocr_loss.mean()]
                        ocr_loss_debug += sp_ocr_loss.to(torch.float32).detach().cpu().numpy().tolist()
                    else:
                        bs_ocr_loss += [torch.tensor(0).float().to(pred_x0.device)]
                    if self.loss_beta > 0:
                        bs_preds_decode[i] = bs_preds_decode[i].to(torch.float32)
                        bs_lang_weight[i] = bs_lang_weight[i].to(torch.float32)
                        sp_ctc_loss = self.cn_recognizer.get_ctcloss(bs_preds_decode[i], bs_gt_texts[i], bs_lang_weight[i])
                        bs_ctc_loss += [sp_ctc_loss.mean()]
                        ctc_loss_debug += sp_ctc_loss.to(torch.float32).detach().cpu().numpy().tolist()
                    else:
                        bs_ctc_loss += [torch.tensor(0).float().to(pred_x0.device)]
                else:
                    bs_ocr_loss += [torch.tensor(0).float().to(pred_x0.device)]
                    bs_ctc_loss += [torch.tensor(0).float().to(pred_x0.device)]

            if PRINT_DEBUG and len(preds_decode) > 0:
                with torch.no_grad():
                    preds_all = preds_decode.softmax(dim=2)
                    preds_all_ori = preds_ori.softmax(dim=2)
                    for k in range(len(preds_all)):
                        pred = preds_all[k].to(torch.float32)
                        order, idx = self.cn_recognizer.decode(pred)
                        text = self.cn_recognizer.get_text(order)
                        pred_ori = preds_all_ori[k].to(torch.float32)
                        order, idx = self.cn_recognizer.decode(pred_ori)
                        text_ori = self.cn_recognizer.get_text(order)
                        str_log = f't = {t}, pred/ori/gt="{text}"/"{text_ori}"/"{gt_texts[k]}"'
                        if self.loss_alpha > 0:
                            str_log += f' ocr_loss={ocr_loss_debug[k]:.4f}'
                        if self.loss_beta > 0:
                            str_log += f' ctc_loss={ctc_loss_debug[k]:.4f}'
                        print(str_log)

            # loss_ocr += torch.stack(bs_ocr_loss) * self.loss_alpha * step_weight
            # loss_ctc += torch.stack(bs_ctc_loss) * self.loss_beta * step_weight
            # import pdb
            # pdb.set_trace()
            # loss_ocr += bs_ocr_loss[0] * self.loss_alpha
            # loss_ctc += bs_ctc_loss[0] * self.loss_beta
        
        step_weight = 1.0
        loss_ocr = torch.stack(bs_ocr_loss) * self.loss_alpha * step_weight
        loss_ctc = torch.stack(bs_ctc_loss) * self.loss_beta * step_weight

        res = {}
        res["loss_ocr"] = loss_ocr.mean()
        res["loss_ctc"] = loss_ctc.mean() 
        return res