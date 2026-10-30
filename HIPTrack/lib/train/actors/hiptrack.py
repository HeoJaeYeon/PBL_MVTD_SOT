from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.target_contrast_loss import target_contrast_loss
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import numpy as np
import cv2

class HIPTrackActor(BaseActor):
    """ Actor for training HIPTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None, multiFrame=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multiFrame = multiFrame

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        #import pdb
        #pdb.set_trace()
        if isinstance(out_dict, list):
            losses = None
            statuses = {}
            for idx, out in enumerate(out_dict):
                partData = {"search_anno" : data['search_anno'][idx].unsqueeze(0)}
                if 'search_scale' in data:
                    partData['search_scale'] = data['search_scale'][idx]  # (batch,) original target size in pixels
                loss, status = self.compute_losses(out, partData)
                if losses is None:
                    losses = loss
                else:
                    losses += loss

                for key, val in status.items():
                    statuses[key] = statuses.get(key, 0.0) + val

            # losses are summed over the search frames (as before); other metrics (IoU, ScaleW/*) are averaged
            for key in statuses:
                if not key.startswith('Loss/'):
                    statuses[key] = statuses[key] / len(out_dict)
            return losses, statuses

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        cv2.imwrite("imgDeNorm.jpg", img=img[:,:,::-1])
        return img

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 5

        template_annos = data['template_anno']
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        previous_imgs = None
        previous_annos = None
        search_imgs = [data['search_images'][i].view(-1, *data['search_images'].shape[2:]) for i in range(len(data['search_images']))]  # (N_search, batch, 3, 384, 384)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]
        if self.multiFrame:
            out_dict = self.net(template=template_list,
                                search=search_imgs,
                                search_after=data['search_images'][1:],
                                previous=previous_imgs,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                gtBoxes=data['search_anno'],
                                previousBoxes=previous_annos,
                                template_boxes=template_annos)
        else:
            out_dict = self.net(template=template_list,
                                search=search_imgs,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                template_boxes=template_annos)
        #self.visualizeCE(out_dict['removed_indexes_s'], out_dict['pred_boxes'].squeeze(1), data['search_images'][0], data['search_anno'][0])
        return out_dict

    def visualizeCE(self, ceMasks, predBoxes, imgs, gtBoxes):
        import pdb
        pdb.set_trace()
        for i in range(16):
            mask = np.ones((24, 24), dtype=np.uint8)
            img = imgs[i]
            img = self.deNorm(img)
            ce1 = ceMasks[0][i]
            ce2 = ceMasks[1][i]
            ce3 = ceMasks[2][i]
            ce = torch.cat([ce1, ce2, ce3], axis=0)
            for num in ce:
                x = int(num) // 24
                y = int(num) % 24
                mask[x][y] = 0
            box = (box_cxcywh_to_xyxy((predBoxes[i])) * 384).int()
            gtBox = (box_xywh_to_xyxy(gtBoxes[i]) * 384).int()
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            mask = np.stack([mask, mask, mask], axis=2) * 255
            mask = cv2.resize(mask, (384, 384))
            cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(mask, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.rectangle(img, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.imwrite(f"maskCE_{i}.jpg", mask[:,:,::-1])
            cv2.imwrite(f"img_{i}.jpg", img[:,:,::-1])

    def _scale_weights(self, gt_dict, num_queries, device):
        """Per-sample loss weights from the target size in the ORIGINAL image (pixels).

        w_raw = clamp((SCALE_REF_PX / s) ** SCALE_GAMMA, 1, SCALE_W_MAX), s = sqrt(w*h) before cropping.
        With SCALE_NORMALIZE the weights are divided by their batch mean, so only the relative emphasis on small
        (low-resolution) targets changes and the overall loss magnitude stays comparable to the baseline.
        """
        if 'search_scale' not in gt_dict:
            raise KeyError("SCALE_AWARE_LOSS needs 'search_scale' (original target size) from the data processing")
        cfg_t = self.cfg.TRAIN
        scale = gt_dict['search_scale'].to(device=device, dtype=torch.float32).view(-1)  # (B,)
        w_raw = (float(cfg_t.SCALE_REF_PX) / scale.clamp(min=1.0)).pow(float(cfg_t.SCALE_GAMMA))
        w_raw = w_raw.clamp(min=1.0, max=float(cfg_t.SCALE_W_MAX))
        w = w_raw / w_raw.mean() if cfg_t.SCALE_NORMALIZE else w_raw
        # pred_boxes (B, Nq, 4).view(-1, 4) is ordered b * Nq + q
        return w.repeat_interleave(num_queries), w_raw

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) -> (batch, 4), (x1, y1, w, h) normalized in search crop
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        device = pred_boxes.device
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # per-sample giou / l1 (BN,)
        try:
            giou, iou = generalized_box_iou(pred_boxes_vec, gt_boxes_vec)
            giou_per = 1.0 - giou
        except Exception:
            # degenerate boxes: same fallback as before (giou term 0, iou 0)
            giou_per = torch.zeros(pred_boxes_vec.shape[0], device=device)
            iou = torch.tensor(0.0, device=device)
        l1_per = self.objective['l1'](pred_boxes_vec, gt_boxes_vec, reduction='none').mean(-1)

        # scale-aware weighting (small targets in the original image get larger weights)
        w_raw = None
        if self.cfg.TRAIN.SCALE_AWARE_LOSS:
            w, w_raw = self._scale_weights(gt_dict, num_queries, device)
            giou_loss = (w * giou_per).mean()
            l1_loss = (w * l1_per).mean()
        else:
            giou_loss = giou_per.mean()
            l1_loss = l1_per.mean()

        # location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        # target-background contrastive loss on the head input feature
        contrast_loss = None
        if self.loss_weight.get('contrast', 0.0) > 0:
            if 'search_feat' not in pred_dict:
                raise KeyError("CONTRAST_WEIGHT > 0 but the model output has no 'search_feat'")
            contrast_loss = target_contrast_loss(pred_dict['search_feat'], gt_bbox, tau=float(self.cfg.TRAIN.CONTRAST_TAU))
            loss = loss + self.loss_weight['contrast'] * contrast_loss

        if return_status:
            # status for log (giou / l1 are logged unweighted so that runs stay comparable)
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_per.detach().mean().item(),
                      "Loss/l1": l1_per.detach().mean().item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if contrast_loss is not None:
                status["Loss/contrast"] = contrast_loss.item()
            if w_raw is not None:
                status["ScaleW/raw_mean"] = w_raw.mean().item()
                status["ScaleW/raw_max"] = w_raw.max().item()
            return loss, status
        else:
            return loss
