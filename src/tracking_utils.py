from numpy.lib import arraysetops
import torch
import numpy as np
import cv2
from sklearn.metrics import average_precision_score
import copy
import torch.nn as nn
from torch.autograd import Variable
from src.kalman import KalmanFilter
from cython_bbox import bbox_overlaps as bbox_ious
import torch.nn.functional as F
from configparser import ConfigParser

mot_fps = {
    'MOT17-13-SDP': 25,
    'MOT17-11-SDP': 30,
    'MOT17-10-SDP': 30,
    'MOT17-09-SDP': 30,
    'MOT17-05-SDP': 14,
    'MOT17-02-SDP': 30,
    'MOT17-04-SDP': 30,
    'MOT17-13-DMP': 25,
    'MOT17-11-DMP': 30,
    'MOT17-10-DMP': 30,
    'MOT17-09-DMP': 30,
    'MOT17-05-DMP': 14,
    'MOT17-02-DMP': 30,
    'MOT17-04-DMP': 30,
    'MOT17-13-FRCNN': 25,
    'MOT17-11-FRCNN': 30,
    'MOT17-10-FRCNN': 30,
    'MOT17-09-FRCNN': 30,
    'MOT17-05-FRCNN': 14,
    'MOT17-02-FRCNN': 30,
    'MOT17-04-FRCNN': 30,
}


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def bisoftmax(x, y):
    feats = torch.mm(x, y.t())/0.1
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    scores = (d2t_scores + t2d_scores) / 2
    return scores.numpy()


def get_proxy(curr_it, mode='inact', tracker_cfg=None, mv_avg=None):
    feats = list()
    avg = tracker_cfg['avg_' + mode]['num']
    proxy = tracker_cfg['avg_' + mode]['proxy']

    if avg != 'all':
        if proxy != 'mv_avg':
            avg = int(avg)
        else:
            avg = float(avg)

    for i, it in curr_it.items():
        # take last bb only
        if proxy == 'last':
            f = it.past_feats[-1]

        # take the first only
        elif avg == 'first':
            f = it.past_feats[0]

        # moving average of features
        elif proxy == 'mv_avg':
            if i not in mv_avg.keys():
                f = it.past_feats[-1]
            else:
                f = mv_avg[i] * avg + it.past_feats[-1] * (1-avg)
            mv_avg[i] = f

        # take all if all or number of features < avg
        elif avg == 'all' or len(it.past_feats) < avg:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats), dim=0)[0]
            elif proxy == 'meannorm':
                f = F.normalize(torch.mean(torch.stack(
                    it.past_feats), dim=0), p=2, dim=0)

        # get proxy of last avg number of frames
        else:
            if proxy == 'mean':
                f = torch.mean(torch.stack(it.past_feats[-avg:]), dim=0)
            elif proxy == 'median':
                f = torch.median(torch.stack(it.past_feats[-avg:]), dim=0)[0]
            elif proxy == 'mode':
                f = torch.mode(torch.stack(it.past_feats[-avg:]), dim=0)[0]
            elif proxy == 'meannorm':
                f = F.normalize(torch.mean(torch.stack(
                    it.past_feats[-avg:]), dim=0), p=2, dim=1)

        feats.append(f)

    if len(feats[0].shape) == 1:
        feats = torch.stack(feats)
    elif len(feats[0].shape) == 3:
        feats = torch.cat([f.unsqueeze(0) for f in feats], dim=0)
    elif len(feats) == 1:
        feats = feats
    else:
        feats = torch.cat(feats, dim=0)

    return feats


def get_center(pos):
    # adapted from tracktor
    if len(pos.shape) <= 1:
        x1 = pos[0]
        y1 = pos[1]
        x2 = pos[2]
        y2 = pos[3]
    else:
        x1 = pos[:, 0]
        y1 = pos[:, 1]
        x2 = pos[:, 2]
        y2 = pos[:, 3]
    if type(pos) == torch.Tensor:
        return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()
    else:
        return np.array([(x2 + x1) / 2, (y2 + y1) / 2])


def get_width(pos):
    # adapted from tracktor
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    # adapted from tracktor
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    # adapted from tracktor
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    # adapted from tracktor
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).numpy()


def bbox_overlaps(boxes, query_boxes):
    # adapted from tracktor
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        # If input is ndarray, turn the overlaps back to ndarray when return
        def out_fn(x): return x.numpy()
    else:
        def out_fn(x): return x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
        (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t(
    )) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t(
    )) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def is_moving(seq, log=False):

    if "MOT" not in seq and 'dance' not in seq:
        return True
    elif 'dance' in seq:
        return False
    elif seq.split('-')[1] in ['13', '11', '10', '05', '14', '12', '07', '06']:
        return True
    elif seq.split('-')[1] in ['09', '04', '02', '08', '03', '01']:
        return False
    else:
        assert False, 'Seqence not valid {}'.format(seq)


def frame_rate(seq):
    if 'dance' in seq:
        return 20
    elif seq.split('-')[1] in ['11', '10', '12', '07',  '09', '04', '02', '08', '03', '01']:
        return 30
    elif seq.split('-')[1] in ['05', '06']:
        return 14
    else:
        return 25


def tlrb_to_xyah(tlrb):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlrb).copy()
    ret[2] = ret[2] - ret[0]
    ret[3] = ret[3] - ret[1]
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


class Track():
    def __init__(
            self,
            track_id,
            bbox, feats,
            im_index,
            gt_id,
            vis,
            conf,
            frame,
            label,
            motion_model=0,  # 0-Linear, 1-KF, 2-SceneMotion
            kalman_filter=None):
        '''
        Args:
            - bbox: (4,)
        '''
        self.kalman = False
        self.scene_motion = False
        if motion_model == 1:
            self.kalman = True
        elif motion_model == 2:
            self.scene_motion = True
        
        self.xyah = tlrb_to_xyah(copy.deepcopy(bbox))
        if self.kalman:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(
                measurement=tlrb_to_xyah(bbox))
        
        self.track_id = track_id
        self.pos = bbox  ## xyxy
        self.bbox = list()
        self.bbox.append(bbox)

        # init variables for motion model
        self.last_pos = list()
        self.last_pos.append(self.pos)
        self.last_v = np.array([0, 0, 0, 0])
        self.last_vc = np.array([0, 0])

        # embedding feature list of detections
        self.past_feats = list()
        self.feats = feats
        self.past_feats.append(feats)

        # image index list of detections
        self.past_im_indices = list()
        self.past_im_indices.append(im_index)
        self.im_index = im_index

        # corresponding gt ids of detections
        self.gt_id = gt_id
        self.past_gt_ids = list()
        self.past_gt_ids.append(gt_id)

        # corresponding gt visibilities of detections
        self.gt_vis = vis
        self.past_gt_vis = list()
        self.past_gt_vis.append(vis)

        # initialize inactive count
        self.inactive_count = 0
        self.past_frames = list()
        self.past_frames.append(frame)

        # conf of current det
        self.conf = conf

        self.past_vs = list()

        # labels of detections
        self.label = list()
        self.label.append(label)

    def __len__(self):
        return len(self.last_pos)

    def add_detection(
            self, bbox, feats, im_index, gt_id, vis, conf, frame, label):
        # update all lists / states
        self.pos = bbox
        self.last_pos.append(bbox)
        self.bbox.append(bbox)

        self.feats = feats
        self.past_feats.append(feats)

        self.past_im_indices.append(im_index)
        self.im_index = im_index

        self.gt_id = gt_id
        self.past_gt_ids.append(gt_id)
        self.gt_vis = vis
        self.past_gt_vis.append(vis)

        self.conf = conf
        self.past_frames.append(frame)
        self.label.append(label)

        if self.kalman:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, tlrb_to_xyah(bbox))
        
        if self.scene_motion:
            pass

    def update_v(self, v):
        self.past_vs.append(v)
        self.last_v = v

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self.bbox.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


def multi_predict(active, inactive, shared_kalman):
    state = list()
    act = [v for v in active.values()]
    state.extend([1]*len(act))
    inact = [v for v in inactive.values()]
    state.extend([0]*len(inact))
    stracks = act + inact
    if len(stracks) > 0:
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, (st, act) in enumerate(zip(stracks, state)):
            if act != 1:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    return stracks


def get_iou_kalman(tracklets, detections):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """

    atlbrs = [track.tlbr for track in tracklets]
    btlbrs = [track['bbox'] for track in detections]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix.T


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def load_scene_model(motion_model_cfg, motion_model_path):
    cfg = ConfigParser()
    cfg.read(motion_model_cfg, encoding="UTF-8")
    motion_model_name = cfg["Motion Model"]["model_name"]
    if motion_model_name == "SceneMotionMultiHead":
        from trackers.scene_tracker.motion_model_multi_head import SceneMotionMultiHead as MotionModel
    elif motion_model_name == "SceneMotion":
        from trackers.scene_tracker.motion_model import SceneMotion as MotionModel

    scene_model = MotionModel(
        z_size=cfg.getint("Motion Model", "noise_dim"),
        inp_format=cfg.get("Motion Model", "inp_format"),
        encoder_h_dim=cfg.getint("Motion Model", "h_dim"),
        decoder_h_dim=cfg.getint("Motion Model", "decoder_h_dim"),
        social_feat_size=cfg.getint("Motion Model", "h_dim") if cfg.getint("Motion Model", "n_social_modules") > 0 else 0,
        embedding_dim=int(cfg.getint("Motion Model", "decoder_h_dim") // 2),
        num_gens=cfg.getint("Motion Model", "num_gens"),
        pred_len=cfg.getint("Motion Model", "pred_len"),
        pool_type=cfg.get("Motion Model", "pool_type"),
        num_social_modules=cfg.getint("Motion Model", "n_social_modules"),
        scene_dim=cfg.getint("Motion Model", "scene_dim"),
        use_pinet=cfg.getboolean("Motion Model", "use_pinet"),
        learn_prior=cfg.getboolean("Motion Model", "learn_prior"),
    )
    ckpt = torch.load(motion_model_path, map_location="cuda:0")
    scene_model.load_state_dict(ckpt["model"])
    scene_model.to("cuda:0")
    scene_model.eval()
    return scene_model


def scene_motion_predict(stracks, scene_motion_model, with_scene=False):
    # state = list()
    # act = [v for v in active.values()]
    # state.extend([1]*len(act))
    # inact = [v for v in inactive.values()]
    # state.extend([0]*len(inact))
    # stracks = act + inact
    if len(stracks) > 0:
        max_seq_len = 0
        for st in stracks:
            if max_seq_len < len(st.bbox):
                max_seq_len = len(st.bbox)
        multi_sequence = []
        multi_d_sequence = []
        multi_imgs = []
        for st in stracks:
            st_bbox = np.asarray(st.bbox)
            st_bbox[:, 2:] = st_bbox[:, 2:] - st_bbox[:, :2]  ## xyxy -> xywh
            st_sequence = torch.from_numpy(st_bbox).unsqueeze(0)  ## (1, seq_len, 4)
            st_d_sequence = torch.cat((torch.zeros((1,1,4)), st_sequence[:,1:,:]-st_sequence[:,:-1,:]), dim=1)
            if len(st.bbox) < max_seq_len:
                pad_dim = max_seq_len - len(st.bbox)
                multi_sequence.append(F.pad(st_sequence.permute(0,2,1), pad=(pad_dim,0,0,0), mode='constant', value=0).permute(0,2,1))
                multi_d_sequence.append(F.pad(st_d_sequence.permute(0,2,1), pad=(pad_dim,0,0,0), mode='constant', value=0).permute(0,2,1))
            else:
                multi_sequence.append(st_sequence)
                multi_d_sequence.append(st_d_sequence)
            if with_scene:
                multi_imgs.append(st.latest_img)

        multi_sequence = torch.cat(multi_sequence, dim=0).permute(1,0,2).cuda()  ## (seq_len, track_num, 4)
        multi_d_sequence = torch.cat(multi_d_sequence, dim=0).permute(1,0,2).cuda()  ## (seq_len, track_num, 4)
        if with_scene:
            multi_imgs = torch.cat(multi_imgs, dim=0).cuda()  ## (track_num, 3, 48, 16)
        else:
            multi_imgs = None
        multi_pred, _, _ = scene_motion_model(multi_sequence, multi_d_sequence, img=multi_imgs, num_samples=1)  ## (track_num, 4)
        for i in range(multi_pred.abs.shape[0]):
            stracks[i].pred = multi_pred.abs[i:i+1, :]
    
    return stracks


def crop_img(img, bbox_tlwh, scene_motion_cfg):
    '''
    Args:
        - img: Tensor[[1, h, w, 3]]
        - bboxtlwh: Tensor[[x, y, w, h]]
    Return:
        - img_tensor: Tensor(1, 3, 48, 16)
    '''
    buffer_scale_w = scene_motion_cfg['buffer_scale_w']
    buffer_scale_h = scene_motion_cfg['buffer_scale_h']
    resize_img_w = scene_motion_cfg['resize_img_w']
    resize_img_h = scene_motion_cfg['resize_img_h']
    bbox = bbox_tlwh[0].int().cpu().numpy()
    cv_img = img[0].type(torch.uint8).numpy()

    imgw = cv_img.shape[1]
    imgh = cv_img.shape[0]

    # buffer bbox region: (x-bw, y-bh, w+bw, h+bh)
    bx = int(bbox[0] - bbox[2] * buffer_scale_w)
    by = int(bbox[1] - bbox[3] * buffer_scale_h)
    bw = int(bbox[2] + bbox[2] * buffer_scale_w)
    bh = int(bbox[3] + bbox[3] * buffer_scale_h)
    # clip bboxes inside image (realized in gen_mggan_motion_data.py)
    xyxy = np.asarray([bx, by, bx+bw, by+bh])
    xyxy[0::2] = xyxy[0::2].clip(0, imgw)
    xyxy[1::2] = xyxy[1::2].clip(0, imgh)

    crop = cv_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]  ## [y1:y2, x1:x2]
    crop = cv2.resize(crop, (resize_img_w, resize_img_h), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(crop).unsqueeze(0).float().permute(0, 3, 1, 2).cuda()
    return img_tensor
