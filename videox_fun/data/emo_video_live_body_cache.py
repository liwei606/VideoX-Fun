from cmath import sqrt
import sys
import os
import random
from glob import glob
from pathlib import Path
from typing import List
import copy
import imageio
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from decord import cpu
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False) # NOTE: enforce single thread

CV2_INTERP = cv2.INTER_LINEAR
DTYPE = np.float32

sys.path.insert(0, str(Path(__file__).parent.parent))

def convert_bbox_to_square_bbox(bbox, max_h, max_w, scale=1.0):
    # Calculate width, height, and max_size of the bounding box
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    max_size = max(width, height) * scale

    # Calculate center of the bounding box
    center_x = (bbox[0][0] + bbox[1][0]) / 2
    center_y = (bbox[0][1] + bbox[1][1]) / 2

    # Calculate the left-up and right-bottom corners of the square bounding box
    half_size = max_size / 2
    left_top = [int(center_x - half_size), int(center_y - half_size)]
    right_bottom = [int(center_x + half_size), int(center_y + half_size)]

    # Ensure the square is within image bounds
    left_top[0] = max(0, left_top[0])  
    left_top[1] = max(0, left_top[1])
    right_bottom[0] = min(max_w, right_bottom[0])
    right_bottom[1] = min(max_h, right_bottom[1])

    # Return the new bounding box as a list of top-left and bottom-right coordinates
    return [left_top[0], left_top[1], right_bottom[0], right_bottom[1]]

def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    sw *= scale
    sh *= scale
    scale_bbox = [cx - sw, cy - sh, cx + sw, cy + sh]
    scale_bbox[0] = np.clip(scale_bbox[0], 0, w)
    scale_bbox[2] = np.clip(scale_bbox[2], 0, w)
    scale_bbox[1] = np.clip(scale_bbox[1], 0, h)
    scale_bbox[3] = np.clip(scale_bbox[3], 0, h)
    return scale_bbox

def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise Exception("Invalid mask")

    # sontime bbox is like this: array([ -8.84635544, 216.97692871, 192.20074463, 502.83700562])
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    # tgt_pose = np.zeros_like(tgt_img.asnumpy())
    tgt_pose = np.zeros((hd, wd, 3))
    tgt_pose[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :] = 255.0
    if return_pil:
        tgt_pose_pil = Image.fromarray(tgt_pose.astype(np.uint8))
        return tgt_pose_pil
    return tgt_pose

def get_move_area(bbox, fw, fh):
    move_area_bbox = [
        bbox[:, 0].min(),
        bbox[:, 1].min(),
        bbox[:, 2].max(),
        bbox[:, 3].max(),
    ]

    if move_area_bbox[0] < 0:
        move_area_bbox[0] = 0
    if move_area_bbox[1] < 0:
        move_area_bbox[1] = 0
    if move_area_bbox[2] > fw:
        move_area_bbox[2] = fw
    if move_area_bbox[3] > fh:
        move_area_bbox[3] = fh
    return move_area_bbox

body_level_dict = {
    1: [16, 14, 0, 15, 17], # 头
    2: [2, 1, 5], # 肩膀
    3: [8, 11], # 腰
    4: [9, 12], # 膝盖
    5: [10, 13] # 双脚
    # 3: [3, 6], # 胳膊肘
    # 4: [8, 11], # 腰
    # 5: [9, 12], # 膝盖
    # 6: [10, 13] # 双脚
}
body_level_prefix_dict = copy.deepcopy(body_level_dict)
for i in range(2, len(body_level_dict) + 1):
    body_level_prefix_dict[i] = body_level_prefix_dict[i] + body_level_prefix_dict[i - 1]

def find_continuous_video_segments(available_flags, frame_num):
    diff_flags = np.diff(available_flags.astype(int))

    start_indices = np.where(diff_flags == 1)[0] + 1  
    end_indices = np.where(diff_flags == -1)[0] 

    if available_flags[0]:  
        start_indices = np.insert(start_indices, 0, 0)
    if available_flags[-1]: 
        end_indices = np.append(end_indices, len(available_flags) - 1)

    # pick segment length >= frame_num
    segments = []
    for start, end in zip(start_indices, end_indices):
        if end - start + 1 >= frame_num:
            segments.append(list(range(start, end+1)))

    return segments

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
def get_hands(hands_mmposes, score_mmposes, cur_face_id, wd, hd):
    frames_len = len(hands_mmposes[0])
    hand_bboxs = [[] for i in range(0, frames_len)] # every frames, including list, list contains every hand bbox (x0, y0, x1, y1)
    for face_id in hands_mmposes:
        if face_id == cur_face_id: continue
        hands_mmpose, score_mmpose = hands_mmposes[face_id], score_mmposes[face_id]
        # print(f"{score_mmpose=}")
        # print(f"{score_mmpose.shape} {score_mmpose=}")
        score_mmpose = score_mmpose[:, 92:]
        for i, (scores_per_frame, hands_per_frame) in enumerate(zip(score_mmpose, hands_mmpose)):
            if np.any(scores_per_frame == -1):
                continue
            x0, y0, x1, y1 = wd, hd, 0, 0
            for score, point in zip(scores_per_frame[:21], hands_per_frame[:21]):
                x, y = int(point[0] * wd), int(point[1] * hd)
                if 0 <= x < wd and 0 <= y < hd and score > 3.0:
                    x0, y0 = min(x0, x), min(y0, y)
                    x1, y1 = max(x1, x), max(y1, y)
            if x0 <= x1 and y0 <= y1:
                hand_bboxs[i].append([x0, y0, x1, y1])
                
            x0, y0, x1, y1 = wd, hd, 0, 0
            for score, point in zip(scores_per_frame[21:], hands_per_frame[21:]):
                x, y = int(point[0] * wd), int(point[1] * hd)
                if 0 <= x < wd and 0 <= y < hd and score > 3.0:
                    x0, y0 = min(x0, x), min(y0, y)
                    x1, y1 = max(x1, x), max(y1, y)
            if x0 <= x1 and y0 <= y1:
                hand_bboxs[i].append([x0, y0, x1, y1])
    return hand_bboxs

def overlap_face(hands_boxes_cur, det):
    overlap_sum_value = 0
    for hands_box in hands_boxes_cur:
        if hands_box[0] == -1:
            continue
        overlap_left = max(hands_box[0], det[0])
        overlap_top = max(hands_box[1], det[1])
        overlap_right = min(hands_box[2], det[2])
        overlap_bottom = min(hands_box[3], det[3])
        overlap_width = max(0, overlap_right - overlap_left)
        overlap_height = max(0, overlap_bottom - overlap_top)
        overlap_area = overlap_width * overlap_height
        area_face = (det[2] - det[0]) * (det[3] - det[1])
        area_hand = (hands_box[2] - hands_box[0]) * (hands_box[3] - hands_box[1])
        # overlap_sum_value += overlap_area / (area_face + area_hand - overlap_area)
        overlap_sum_value += overlap_area / area_face
    return overlap_sum_value >= 0.01

def get_subtitle(text_boxes, len_text_boxes, video_height):
    min_height_ratio=0.05
    max_height_ratio=0.2

    upper_half_y_threshold = video_height * 0.2
    lower_half_y_threshold = video_height * 0.6

    text_detections = []
    for box in text_boxes:
        x_min, y_min, x_max, y_max = box
        y_center = (y_min + y_max) / 2

        height = y_max - y_min
        width = x_max - x_min
    
        if ((y_min > lower_half_y_threshold or y_max < upper_half_y_threshold) and 
            height / video_height < max_height_ratio and 
            width / height > 4):
            text_detections.append(box)

    return text_detections     

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

from math import sin, cos, acos, degrees
def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    """ conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]

def parse_pt2_from_pt4(pt4, use_lip=False):
    """
    parsing the 2 points according to the 68 points, which cancels the roll
    """
    mouth_p, nose_p, left_eye_p, right_eye_p = pt4
    if use_lip:
        pt2 = np.stack([
            (left_eye_p + right_eye_p) / 2,
            # mouth_p
            nose_p
        ], axis=0)
    else:
        pt2 = np.stack([
            left_eye_p,  # left eye
            right_eye_p,  # right eye
        ], axis=0)

    return pt2

def parse_rect_from_landmark(
    pt4,
    pts,
    scale=1.5,
    ratio_scale=1.0,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt4(pt4, use_lip=kwargs.get('use_lip', False))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if (size[0] / size[1]) > ratio_scale:
        size[1] = size[0] / ratio_scale
    else:
        size[0] = size[1] * ratio_scale
    # print(f"{size=} {size[0] / size[1]=} {ratio_scale=}")

    size *= scale  # scale size
    center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
    b_center = center
    center = center + ux * (vx_ratio * size[0]) + uy * \
        (vy_ratio * size[1])  # considering the offset in vx and vy direction
    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle

def parse_bbox_from_landmark(pt4, pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pt4, pts, **kwargs)
    cx, cy = center
    w, h = size

    # calculate the vertex positions before rotation
    bbox = np.array([
        [cx-w/2, cy-h/2],  # left, top
        [cx+w/2, cy-h/2],
        [cx+w/2, cy+h/2],  # right, bottom
        [cx-w/2, cy+h/2]
    ], dtype=DTYPE)

    # construct rotation matrix
    bbox_rot = bbox.copy()
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ], dtype=DTYPE)

    # calculate the relative position of each vertex from the rotation center, then rotate these positions, and finally add the coordinates of the rotation center
    bbox_rot = (bbox_rot - center) @ R.T + center

    return {
        'center': center,  # 2x1
        'size': size,  # scalar
        'angle': angle,  # rad, counterclockwise
        'bbox': bbox,  # 4x2
        'bbox_rot': bbox_rot,  # 4x2
    }

def _estimate_similar_transform_from_pts(
    pt4,
    pts,
    dsize,
    scale=1.5,
    vx_ratio=0,
    vy_ratio=-0.1,
    flag_do_rot=True,
    **kwargs
):
    """ calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pt4, pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get('use_lip', False)
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=DTYPE
        )
    else:
        M_INV = np.array(
            [[s, 0, tgt_center[0] - s * center[0]],
             [0, s, tgt_center[1] - s * center[1]]],
            dtype=DTYPE
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]

def crop_image(imgs, pt4: np.ndarray, pts: np.ndarray, **kwargs):
    dsize = kwargs.get('dsize', 224)
    scale = kwargs.get('scale', 1.5)  # 1.5 | 1.6
    vy_ratio = kwargs.get('vy_ratio', -0.1)  # -0.0625 | -0.1

    M_INV, _ = _estimate_similar_transform_from_pts(
        pt4,
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get('flag_do_rot', True),
    )

    img_crops = [_transform_img(img, M_INV, dsize) for img in imgs]
    # pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    ret_dct = {
        'M_o2c': M_o2c,  # from the original image to the cropped image 3x3
        'M_c2o': M_c2o,  # from the cropped image to the original image 3x3
        'img_crops': img_crops,  # the cropped image
        # 'pt_crop': pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct

def crop_image_by_bbox(img, bbox, lmk=None, dsize=(512, 768), angle=None, flag_rot=False, **kwargs):
    # dsize: (width, height)
    left, top, right, bot = bbox
    size = (right - left, bot - top)

    src_center = np.array([(left + right) / 2, (top + bot) / 2], dtype=DTYPE)
    tgt_center = np.array([dsize[0] / 2, dsize[1] / 2], dtype=DTYPE)

    s = dsize[0] / size[0]  # scale
    if flag_rot and angle is not None:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = src_center[0], src_center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_o2c = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=DTYPE
        )
    else:
        M_o2c = np.array(
            [[s, 0, tgt_center[0] - s * src_center[0]],
             [0, s, tgt_center[1] - s * src_center[1]]],
            dtype=DTYPE
        )

    img_crop = _transform_img(img, M_o2c, dsize=dsize, borderMode=kwargs.get('borderMode', None))

    M_o2c = np.vstack([M_o2c, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    return img_crop

def average_bbox_lst(bbox_lst):
    if len(bbox_lst) == 0:
        return None
    bbox_arr = np.array(bbox_lst)
    return np.mean(bbox_arr, axis=0).tolist()

def mid_bbox_lst(bbox_lst):
    if len(bbox_lst) == 0:
        return None
    bbox_arr = np.array(bbox_lst)
    return bbox_arr[len(bbox_arr) // 2].tolist()

class LiveVideoDataset(Dataset):
    def __init__(
        self,
        cfg=None,
        width=512,
        height=512,
        split='train',
        enable_bucket=False,
        resume_step=0,
        test_size=100,
        save_gt=False,
        start_id=0,
        end_id=-1,
        sample_times=1,

        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_gt = save_gt
        self.img_size = (width, height)
        self.width = width
        self.height = height
        self.split = split
        self.drop_n_frame = 3 # drop first and last [drop_n_frame] frame to filter out unreasonable scene transition
        self.zero_to_one = self.cfg.data.get('zero_to_one', False)
        self.enable_bucket = enable_bucket

        if split == 'train': 
            dataset_file_path_list = cfg.data.dataset_file_path
            cache_file_path_list = cfg.data.get("cache_file_path", [])
            self.n_sample_frames = cfg.data.n_sample_frames
            self.past_n = cfg.data.past_n
            self.union_bbox_scale = self.cfg.data.union_bbox_scale
            self.flip_aug = self.cfg.data.get('flip_aug', False)
            self.resume_step = resume_step
        else: 
            dataset_file_path_list = cfg.val_data.dataset_file_path
            cache_file_path_list = []
            self.n_sample_frames = cfg.val_data.n_sample_frames
            self.past_n = cfg.val_data.past_n
            self.union_bbox_scale = self.cfg.val_data.union_bbox_scale
            self.flip_aug = self.cfg.val_data.get('flip_aug', False)
            self.resume_step = 0
        
        vid_path = []
        for dataset_file_path, weights in dataset_file_path_list:
            cur_res = list(np.load(dataset_file_path, allow_pickle=True)["arr_0"])
            cur_res = [(x, "video") for x in cur_res]
            print(f"Find {len(cur_res)} item in {dataset_file_path}, weights is {weights}")
            vid_path += cur_res * weights
        
        # DEBUG -------------------------------------------
        # debug_vid_path = []
        # for data in vid_path:
        #     video_dir_root, video_name = data
        #     if "00701738-Scene-023" in video_name:
        #         debug_vid_path.append(data)
        # vid_path = debug_vid_path
        # DEBUG -------------------------------------------
        
        #### for debug 
        self.debug = cfg.data.get("debug", False)

        self.target_fps = cfg.data.train_fps
        print(f"origin {len(vid_path)=}")
        
        self.vid_path = vid_path[start_id: ] if end_id == -1 else vid_path[start_id: end_id]
        print(f"After CLIP {len(self.vid_path)=}")
        self.dataset = self.vid_path * sample_times



        self.pixel_norm = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                
            ]
        )
        
        if 'color_jitter' in cfg.data and cfg.data.color_jitter: 
            self.color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        else:
            self.color_jitter = None

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        
        if isinstance(images, List):
            transformed_images = [transform(Image.fromarray(img))[None] for img in images]
            transformed_images = torch.cat(transformed_images)
            return transformed_images  # (f, c, h, w)
        else:
            return transform(Image.fromarray(images))  # (c, h, w)

    def __len__(self):
        return len(self.dataset)

    def get_item(self, index):
        (folder, name), data_type = self.dataset[index]
        is_anime_data = False
        if ("anime" in folder) or ("Anime" in folder):
            is_anime_data = True
        vid_path = f'{folder}/videos_resampled/{name}+resampled.mp4'
        metadata_file = f"{folder}/metadata/{name}/metadata_mmpose.npz"
        if not os.path.exists(metadata_file):
            metadata_file = f"{folder}/metadata/{name}/metadata.npz"
        if not os.path.exists(vid_path):
            vid_path = f'{folder}/videos/{name}.mp4'

        video_metadata = dict(np.load(metadata_file, allow_pickle=True))["arr_0"].item()
        
        video_path = vid_path
        assert Path(video_path).exists(), f"{video_path=} not exists."

        pre_processed = video_metadata.get("bounding_box_union", "") == "pre_processed"
        
        bounding_box_dict = video_metadata["frame_data"]["bounding_box"]
        valid_face_ids = [x for x in bounding_box_dict]
        if "tracks_synced" in video_metadata:
            valid_face_ids = []
            video_sync_results = video_metadata["tracks_synced"]
            for track_id, sync_results in video_sync_results.items():
                meets_thresh = (
                    abs(sync_results["offset"]) <= self.cfg.data.get("audio_offset_thresh", 100)
                    and abs(sync_results["conf"]) >= self.cfg.data.get("audio_conf_thresh", 1)
                )
                if meets_thresh:
                    valid_face_ids.append(track_id)
        
        if "face_id_pair_audiov3" in video_metadata:
            face_audio_pair = video_metadata["face_id_pair_audiov3"]
            valid_dyadic_faces = 0
            lose_conf_list = []
            for dyadic_face in face_audio_pair:
                if face_audio_pair[dyadic_face]["conf"] >= self.cfg.data.get("audio_dyadic_conf_thresh", 0):
                    valid_dyadic_faces += 1
                else:
                    lose_conf_list.append(face_audio_pair[dyadic_face]["conf"])
            assert valid_dyadic_faces > 0, f"For Dyadic data, not enough face, {lose_conf_list=}"
            face_audio_pair = video_metadata["face_id_pair_audiov3"]
            face_id = random.choice(list(face_audio_pair.keys()))
        else:
            face_id = -1
            avg_face_dist_max = 0
            for track_id in valid_face_ids:
                bounding_box = bounding_box_dict[track_id]
                batch_bbox = bounding_box[...]
                batch_bbox_fliter = []
                face_valid_frames = []
                for idx, det in enumerate(batch_bbox):
                    if det[0] == -1 or det[1] == -1 or det[2] == -1 or det[3] == -1:
                        face_valid_frames.append(0)
                        continue
                    batch_bbox_fliter.append(det)
                batch_bbox = np.array(batch_bbox_fliter)
                bbox_x0, bbox_y0, bbox_x1, bbox_y1 = batch_bbox[:, 0].mean(), batch_bbox[:, 1].mean(), batch_bbox[:, 2].mean(), batch_bbox[:, 3].mean()
                # avg_face_dist = (bbox_y1 - bbox_y0 + bbox_x1 - bbox_x0) / 2
                avg_face_dist = np.sqrt((bbox_y1 - bbox_y0) * (bbox_x1 - bbox_x0))
                if avg_face_dist > avg_face_dist_max or face_id == -1:
                    avg_face_dist_max = avg_face_dist
                    face_id = track_id
        assert face_id != -1, f"Error, face_id is {face_id}"

        # Get faces_bbox_other for fliter overlap case with current select face_id

        def check_bbox(bbox):
            if not np.isfinite(bbox).all():
                # import pdb; pdb.set_trace()
                raise Exception(f"Some of the bboxes have non-finite values.")
            bbox_extend = bbox[:, 1] - bbox[:, 0]
            if bbox_extend.min() <= 0:
                # print(vid_path)
                # import pdb; pdb.set_trace()
                raise Exception(f"0, Some of the bboxes are invalid -- they extend zero area.")

        bounding_box = bounding_box_dict[face_id]
        if isinstance(bounding_box, list):
            bounding_box = np.array(bounding_box)

        target_fps = self.cfg.data.train_fps

        # Get correct index to sample from when we resample video to target fps
        try:
            video_reader = VideoReader(vid_path, ctx=cpu(0))
            hd, wd, _ = video_reader[0].asnumpy().shape
        except:
            video_reader = VideoReader(vid_path, ctx=cpu(0))
            hd, wd, _ = video_reader[0].asnumpy().shape

        original_fps = video_reader.get_avg_fps()
        original_frame_count = len(video_reader)
        
        # assert abs(round(original_fps) - original_fps) <= 0.03, f"Not support {original_fps=}"
        # original_fps = round(original_fps)
        target_frame_count = int((original_frame_count / original_fps) * target_fps)
        target_frame_indices = [
            min(int((target_fps_pos / target_fps) * original_fps), original_frame_count - 1)
            for target_fps_pos in range(target_frame_count)
        ]
        target_frame_indices = np.array(target_frame_indices)
        target_frame_indices = target_frame_indices[self.drop_n_frame:-self.drop_n_frame]

        assert original_frame_count == len(
            bounding_box
        ), f"{original_frame_count = } != {len(bounding_box) = } in {video_path}"

        clip_length = self.n_sample_frames + self.past_n

        # check miss face kps
        face_keypoints = video_metadata['frame_data']['keypoints'][face_id][target_frame_indices]
        miss_face_kps = (face_keypoints.reshape(len(face_keypoints), -1) == -1).all(-1)
        available_flag = ~miss_face_kps
                
        available_indices = np.where(available_flag == True)[0].tolist()
        if len(available_indices) < clip_length:
            raise Exception('no available segemnt')
        available_segment_list = find_continuous_video_segments(available_flag, clip_length)
        
        if len(available_segment_list) == 0:
            raise Exception('no available segemnt')
        else:
            target_frame_indices_new = target_frame_indices[np.array(random.choice(available_segment_list))]
        # Get crop bbox by liveportrat
        scale_crop_driving_video: float = self.cfg.data.driving_video_scale  # 2.0 # scale factor for cropping driving video
        vx_ratio_crop_driving_video: float = self.cfg.data.get("vx_ratio_crop", 0)  # adjust x offset
        vy_ratio_crop_driving_video: float = self.cfg.data.get("vy_ratio_crop", -0.2)  # adjust y offset
        
        face_keypoints = video_metadata['frame_data']['keypoints'][face_id]
        bodies_mmpose = video_metadata["frame_data"]["bodies_mmpose"][face_id]
        scores_mmpose = video_metadata["frame_data"]["score_mmpose"][face_id]
        body_score_thresh = self.cfg.data.get("body_score_thresh", 0)
        body_mmpose_list = [[] for i in range(18)]
        mouth_p_list = []
        nose_p_list = []
        right_shoulder_p_list = []
        left_shoulder_p_list = []
        lip_movements = {}
        for tar_idx in target_frame_indices_new:
            face_keypoint_perframe = face_keypoints[tar_idx].copy()
            body_mmpose = bodies_mmpose[tar_idx].copy()
            body_score = scores_mmpose[tar_idx, :18].copy()
            body_mmpose[:, 0] = np.clip(body_mmpose[:, 0], 0, 1) * wd
            body_mmpose[:, 1] = np.clip(body_mmpose[:, 1], 0, 1) * hd
            body_mmpose = body_mmpose.astype("int32")
            
            up_mouth_landmarks = [88, 87, 14, 317, 402, 318]
            down_mouth_landmarks = [80, 82, 13, 312, 311, 310]
            up_mouth_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in up_mouth_landmarks]
            down_mouth_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in down_mouth_landmarks]
            delta_mouth_y = np.array([abs(x - y) for x, y in zip(up_mouth_y, down_mouth_y)]).mean()

            mouth_landmarks = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321, 321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267, 269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81, 81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308,]
            mouth_x = [int(np.clip(face_keypoint_perframe[idx, 0], 0, 1) * wd) for idx in mouth_landmarks]
            mouth_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in mouth_landmarks]
            mouth_bbox = [(min(mouth_x), min(mouth_y)), (max(mouth_x), max(mouth_y))]
            mouth_p = np.array([(mouth_bbox[0][0] + mouth_bbox[1][0]) / 2, (mouth_bbox[1][0] + mouth_bbox[1][1]) / 2])
            delta_mouth_y = delta_mouth_y / abs(mouth_bbox[1][1] - mouth_bbox[0][1])
            # lip_movements.append(delta_mouth_y)
            lip_movements[tar_idx] = delta_mouth_y

            nose_landmarks = [48, 115, 220, 45, 4, 275, 440, 344, 278]
            nose_x = [int(np.clip(face_keypoint_perframe[idx, 0], 0, 1) * wd) for idx in nose_landmarks]
            nose_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in nose_landmarks]
            nose_bbox = [(min(nose_x), min(nose_y)), (max(nose_x), max(nose_y))]
            nose_p = np.array([(nose_bbox[0][0] + nose_bbox[1][0]) / 2, (nose_bbox[1][0] + nose_bbox[1][1]) / 2])

            left_eye_landmarks = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
            right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

            left_eye_x = [int(np.clip(face_keypoint_perframe[idx, 0], 0, 1) * wd) for idx in left_eye_landmarks]
            left_eye_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in left_eye_landmarks]
            left_eye_bbox = [(min(left_eye_x), min(left_eye_y)), (max(left_eye_x), max(left_eye_y))]
            left_eye_p = np.array([(left_eye_bbox[0][0] + left_eye_bbox[1][0]) / 2, (left_eye_bbox[1][0] + left_eye_bbox[1][1]) / 2])
            
            right_eye_x = [int(np.clip(face_keypoint_perframe[idx, 0], 0, 1) * wd) for idx in right_eye_landmarks]
            right_eye_y = [int(np.clip(face_keypoint_perframe[idx, 1], 0, 1) * hd) for idx in right_eye_landmarks]
            right_eye_bbox = [(min(right_eye_x), min(right_eye_y)), (max(right_eye_x), max(right_eye_y))]
            right_eye_p = np.array([(right_eye_bbox[0][0] + right_eye_bbox[1][0]) / 2, (right_eye_bbox[1][0] + right_eye_bbox[1][1]) / 2])

            mouth_p_list.append(mouth_p)
            nose_p_list.append(nose_p)
            if body_score[2] >= body_score_thresh:
                right_shoulder_p_list.append(body_mmpose[2])
            if body_score[5] >= body_score_thresh:
                left_shoulder_p_list.append(body_mmpose[5])

            for p_i, (body_mmpose_i, body_score_i) in enumerate(zip(body_mmpose, body_score)):
                if body_score_i >= body_score_thresh:
                    body_mmpose_list[p_i].append(body_mmpose_i)
            
        del face_keypoint_perframe, mouth_landmarks, mouth_x, mouth_y, mouth_bbox, mouth_p, nose_landmarks, nose_x, nose_y, nose_bbox, nose_p, left_eye_landmarks, right_eye_landmarks, left_eye_x, left_eye_y, left_eye_bbox, left_eye_p, right_eye_x, right_eye_y, right_eye_bbox, right_eye_p, body_mmpose, body_score
        body_level_idx = body_level_prefix_dict[self.cfg.data.get("body_level", 3)]
        body_mmpose_list = [body_mmpose_list[i] for i in body_level_idx]
        body_mmpose_array = np.array([np.array(body_mmpose).mean(axis=0) for body_mmpose in body_mmpose_list])
        mouth_p = np.array(mouth_p_list).mean(axis=0)
        nose_p = np.array(nose_p_list).mean(axis=0)
        right_shoulder_p = np.array(right_shoulder_p_list).mean(axis=0)
        left_shoulder_p = np.array(left_shoulder_p_list).mean(axis=0)
        if left_shoulder_p[0] > right_shoulder_p[0]:
            left_shoulder_p, right_shoulder_p = right_shoulder_p, left_shoulder_p
        pts = np.array([mouth_p, nose_p, right_shoulder_p, left_shoulder_p])
        ret_bbox = parse_bbox_from_landmark(
            pts, 
            # face_keypoint_perframe[:, :2],
            body_mmpose_array,
            scale=scale_crop_driving_video,
            vx_ratio=vx_ratio_crop_driving_video,
            vy_ratio=vy_ratio_crop_driving_video,
            ratio_scale=(self.img_size[0] / self.img_size[1]),
        )["bbox"]
        # crop_bbox = np.array([
        #     ret_bbox[0, 0],
        #     ret_bbox[0, 1],
        #     ret_bbox[2, 0],
        #     ret_bbox[2, 1],
        # ])  # 4,
        crop_bbox = np.array([
            max(ret_bbox[0, 0], 0),
            max(ret_bbox[0, 1], 0),
            min(ret_bbox[2, 0], wd),
            min(ret_bbox[2, 1], hd),
        ])
        cur_img_size = (int(crop_bbox[2] - crop_bbox[0]), int(crop_bbox[3] - crop_bbox[1]))
        # cur_img_size = (crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1])
        del body_mmpose_list, mouth_p, mouth_p_list, nose_p, nose_p_list, right_shoulder_p, right_shoulder_p_list, left_shoulder_p, left_shoulder_p_list, pts, ret_bbox, body_mmpose_array
        
        # crop_bbox = average_bbox_lst(bbox_list)
        # crop_bbox = mid_bbox_lst(bbox_list)
        center_x = (crop_bbox[0] + crop_bbox[2]) / 2
        center_y = (crop_bbox[1] + crop_bbox[3]) / 2
        center = [int(center_y), int(center_x)]

        ### filtering large translation by union_mask_img
        vid_batch_bbox = bounding_box[target_frame_indices_new]
        union_bbox_full_video = get_move_area(vid_batch_bbox, wd, hd)
        
        # 1. Fliter hand overlap: checking whether hand is appear in crop bbox
        if not is_anime_data:
            hands_boxes = get_hands(video_metadata["frame_data"]["hands_mmpose"],
                            video_metadata["frame_data"]["score_mmpose"],
                            face_id,
                            wd, hd)
        else:
            hands_boxes = [[] for i in range(0, len(video_metadata['frame_data']['keypoints'][face_id]))]
        available_flag = np.ones(len(target_frame_indices_new), dtype=bool)
        
        # 2. text overlap: Get Text bbox for fliter subtitle
        if 'text_detection' in video_metadata:
            text_boxes = video_metadata['text_detection']['text_boxes']
            text_bounding_boxes_num = len(text_boxes)
            # print(f"{text_bounding_boxes_num=}, text_detection")
        elif 'text_bounding_boxes' in video_metadata:
            text_boxes = video_metadata['text_bounding_boxes']
            text_bounding_boxes_num = len(text_boxes)
            # print(f"{text_bounding_boxes_num=}, text_bounding_boxes")
        elif is_anime_data:
            text_boxes = []
            text_bounding_boxes_num = 0
            # print(f"{text_bounding_boxes_num=}, don't have any")
        else:
            raise ValueError("This is not an anime data, We cannot check text")
        
        if not self.cfg.data.get("fliter_subtitle", True):
            text_bounding_boxes_num = 0
        # Some of them are watermarks that can be used as background. This data forces skipping the subtitle detection
        if "TheSkinDeep" in folder: 
            text_bounding_boxes_num = 0
        text_boxes = get_subtitle(text_boxes, text_bounding_boxes_num, hd)
        
        # # 3. other human face or body overlap: Get faces_bbox_other for fliter overlap case with current select face_id
        bbox_others = [[] for i in range(0, len(bounding_box_dict[face_id]))]
        bounding_box_human = None
        for track_id in bounding_box_dict:
            if track_id == face_id:
                continue
            if "bounding_box_human" in video_metadata["frame_data"]:
                bounding_box_human = video_metadata["frame_data"]["bounding_box_human"][track_id]
            else:
                bounding_box_human = bounding_box_dict[track_id]
            for idx, det in enumerate(bounding_box_human):
                if det[0] == -1 or det[1] == -1 or det[2] == -1 or det[3] == -1:
                    continue
                bbox_others[idx].append(det)
        del bounding_box_human, track_id, det
        # 4. Head pose fliter: Removes frames with too large a face orientation deviation
        if "face_id_pair_hopenet" in video_metadata:
            hopenet_params = video_metadata["face_id_pair_hopenet"][face_id]
        elif "pose_max" in self.cfg.data:
            raise ValueError(f"There is not hopenet_params in {video_path}")
        else:
            hopenet_params = [[0, 0, 0] for i in range(0, len(bounding_box_dict[face_id]))]
        # 5. IQA Fliter: Removes frames with blurred faces
        if "hyperIQA_frame_data" in video_metadata:
            hyperIQA_params = video_metadata["hyperIQA_frame_data"][face_id]
        elif "hyperIQA_min" in self.cfg.data:
            raise ValueError(f"There is not hyperIQA_params in {video_path}")
        else:
            hyperIQA_params = [0 for i in range(0, len(bounding_box_dict[face_id]))]

        # start fliter 1,2,3,4
        if text_bounding_boxes_num > 0:
            assert not overlap_face(text_boxes, crop_bbox), "Text appear, we need to skip it"
        hopenet_pose_max = self.cfg.data.get("pose_max", [1e5, 1e5, 1e5])
        hopenet_pose_delta = self.cfg.data.get("pose_delta", [1e5, 1e5, 1e5])
        hyperIQA_min = self.cfg.data.get("hyperIQA_min", 0)
        lip_open_ratio = self.cfg.data.get("lip_open_ratio", 0)
        for i in range(len(available_flag)):
            available_flag[i] = available_flag[i] & (not overlap_face(hands_boxes[target_frame_indices_new[i]], crop_bbox))
            available_flag[i] = available_flag[i] & (not overlap_face(bbox_others[target_frame_indices_new[i]], crop_bbox))
            yaw, pitch, roll = hopenet_params[target_frame_indices_new[i]]
            available_flag[i] = available_flag[i] & (abs(yaw) <= hopenet_pose_max[0] and abs(pitch) <= hopenet_pose_max[1] and abs(roll) <= hopenet_pose_max[2])
            hyperIQA_per_frame = hyperIQA_params[target_frame_indices_new[i]]
            available_flag[i] = available_flag[i] & (hyperIQA_per_frame >= hyperIQA_min)
            available_flag[i] = available_flag[i] & (lip_open_ratio <= lip_movements[target_frame_indices_new[i]])

        available_indices = np.where(available_flag == True)[0].tolist()
        if len(available_indices) < clip_length:
            raise Exception('no available segemnt')
        available_segment_list = find_continuous_video_segments(available_flag, clip_length)
        if len(available_segment_list) == 0:
            raise Exception('no available segemnt')
        tfi_copy = target_frame_indices_new.copy()
        target_frame_indices_new = target_frame_indices_new[np.array(random.choice(available_segment_list))]

        # Sample indices for the training run
        video_length = len(target_frame_indices_new)
        if self.n_sample_frames == -1: 
            start_idx = 0
            all_idx = list(range(video_length))
        else:
            # import pdb; pdb.set_trace()
            start_idx = np.random.randint(0, video_length - clip_length+1)
            all_idx = list(range(start_idx, start_idx + clip_length))
            all_idx = [idx for idx in all_idx if idx < video_length]

        past_batch_index = all_idx[:self.past_n]
        tgt_batch_index = all_idx[self.past_n:]
        ref_img_idx = random.choice(available_indices)

        if self.split != 'train': 
            ref_img_idx = 0
            start_idx = 1
            all_idx = list(range(start_idx, start_idx + clip_length))
            all_idx = [idx for idx in all_idx if idx < video_length]
            past_batch_index = all_idx[: self.past_n]
            tgt_batch_index = all_idx[self.past_n :]
        
        assert len(all_idx) == clip_length
        
        # jump case that head pose has a large distance with reference image
        all_indices = target_frame_indices_new[tgt_batch_index].tolist() + [tfi_copy[ref_img_idx]] + target_frame_indices_new[past_batch_index].tolist()
        yaw_ref, pitch_ref, roll_ref = hopenet_params[tfi_copy[ref_img_idx]]
        for idx in all_indices:
            yaw, pitch, roll = hopenet_params[idx]
            assert abs(yaw_ref - yaw) < hopenet_pose_delta[0], f"To Big!!! Current yaw is {yaw}, ref-image yaw is {yaw_ref}"
            assert abs(pitch_ref - pitch) < hopenet_pose_delta[1], f"To Big!!! Current pitch is {pitch}, ref-image pitch is {pitch_ref}"
            assert abs(roll_ref - roll) < hopenet_pose_delta[2], f"To Big!!! Current roll is {roll}, ref-image roll is {roll_ref}"
        if self.cfg.data.hybrid_face_mask:
            ## generate face mask for crop image
            face_keypoints = video_metadata['frame_data']['keypoints'][face_id][all_indices]
            miss_face_kps = (face_keypoints.reshape(len(face_keypoints), -1) == -1).all(-1)
            if miss_face_kps.any():
                print(f"{miss_face_kps=}")
                # print(f"{all_indices=} {target_frame_indices[ref_img_idx]=} {target_frame_indices_new[ref_img_idx]=} {ref_img_idx=}")
                raise Exception('This video has missing face keypoints in some frames') 
            
            for i in all_indices:
                assert not overlap_face(hands_boxes[i], crop_bbox), f"Error, At {i} frame, hand is appear"
                assert not overlap_face(bbox_others[i], crop_bbox), f"Error, At {i} frame, Other Face is appear"
                
        # Read target frames
        def get_imgs_from_idx(idx):
            bs_img = video_reader.get_batch(target_frame_indices_new[idx]).asnumpy()
            return [Image.fromarray(bs_img[idxl]) for idxl in range(len(bs_img))]
        
        try:
            # using gpu 
            vid_pil_image_past = get_imgs_from_idx(past_batch_index)
            vid_pil_image_list = get_imgs_from_idx(tgt_batch_index)
            ref_img = Image.fromarray(video_reader[tfi_copy[ref_img_idx]].asnumpy())
        except:
            # when fail to retrive frames on gpu 
            video_reader = VideoReader(vid_path, ctx=cpu(0))
            vid_pil_image_past = get_imgs_from_idx(past_batch_index)
            vid_pil_image_list = get_imgs_from_idx(tgt_batch_index)
            ref_img = Image.fromarray(video_reader[tfi_copy[ref_img_idx]].asnumpy())

        for vid_pil_image in vid_pil_image_list:
            assert np.array(vid_pil_image).mean() >= 0.2, "Meet all black frames, skip It !!!"

        ## crop image
        
        ref_img = crop_image_by_bbox(np.array(ref_img), crop_bbox, dsize=cur_img_size)
        vid_pil_image_list = [crop_image_by_bbox(np.array(img), crop_bbox, dsize=cur_img_size) for img in vid_pil_image_list]
        vid_pil_image_past = [crop_image_by_bbox(np.array(img), crop_bbox, dsize=cur_img_size) for img in vid_pil_image_past]

        clip_st = target_frame_indices_new[past_batch_index[0]] / original_fps
        clip_et = target_frame_indices_new[tgt_batch_index[-1]] / original_fps
        wav_st = int(clip_st * target_fps)
        wav_et = int(clip_et * target_fps)
        sample = dict(
            folder=folder,
            name=name,
            video_path=video_path,
            data_type=data_type,
            pixel_values_vid_original=np.array(vid_pil_image_list),
            pixel_values_past_frames_original=np.array(vid_pil_image_past),
            face_id=face_id,
            crop_bbox=crop_bbox,
            clip_target_idx=target_frame_indices_new[past_batch_index + tgt_batch_index],
        )

        if self.save_gt:
            if os.path.exists(video_path.replace("mp4", "wav")):
                gt_filt_audio = video_path.replace("mp4", "wav")
            elif os.path.exists(video_path.replace("+resampled.mp4", "+audio.wav")):
                gt_filt_audio = video_path.replace("+resampled.mp4", "+audio.wav")
            else:
                gt_filt_audio = video_path.replace("+resampled.mp4", "+audiov4.wav")
            audio_self_path = ""
            audio_other_path = ""
            if "face_id_pair_audiov3" in video_metadata:
                face_audio_pair = video_metadata["face_id_pair_audiov3"]
                self_face_id, other_face_id = face_audio_pair.keys()
                if face_id != self_face_id:
                    self_face_id, other_face_id = other_face_id, self_face_id
                audio_self_path = video_path.replace("+resampled.mp4", f"+audio_v3_{self_face_id}.wav")
                audio_other_path = video_path.replace("+resampled.mp4", f"+audio_v3_{other_face_id}.wav")
            sample.update(
                dict(
                    video_path=video_path,
                    audio_path=gt_filt_audio,
                    audio_self_path=audio_self_path,
                    audio_other_path=audio_other_path,
                    clip_st=clip_st,
                    clip_et=clip_et,
                )
            )

        return sample


    def post_process_item(self, process_dict, index):
        pixel_values = np.concatenate([process_dict["pixel_values_past_frames_original"], 
                                       process_dict["pixel_values_vid_original"]], axis=0)
        # pixel_values = torch.cat([process_dict["pixel_values_past_frames_original"], 
        #                           process_dict["pixel_values_vid_original"]], dim=0)
        # pixel_values = (pixel_values + 1.) / 2.
        text = "A person is speaking with lively facial expressions and clearly synchronized lip movements. The person's emotion remains calm throughout. The camera and the background behind the person are perfectly static, with no shaking or movement. The person's body movements are natural, with occasional gentle hand gestures. The video is highly realistic and sharp, with clear details on the face and hands."
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "data_type": process_dict["data_type"],
            "idx": index,
            "video_path": process_dict["video_path"],
            "folder": process_dict["folder"],
            "name": process_dict["name"],
            "face_id": process_dict["face_id"],
            "crop_bbox": process_dict["crop_bbox"],
            "clip_target_idx": process_dict["clip_target_idx"],
        }
        if not self.enable_bucket:
            state = torch.get_rng_state()
            # print(f"{pixel_values.shape=} {np.unique(pixel_values)=}")
            pixel_values = [x for x in pixel_values]
            pixel_values = self.augmentation(pixel_values, self.pixel_norm, state)
            sample["pixel_values"] = pixel_values
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values
        
        return sample

    def __getitem__(self, index):
        # return self.get_item(index)
        while True:
            try:
                sample = self.get_item((index + self.resume_step) % len(self))
                break
            except Exception as e:
                # import traceback;traceback.print_exc()
                # You can optionally log the error here
                skip_index = (index + self.resume_step) % len(self)
                print(f"Skipping index {skip_index} due to: {str(e)}")
                # Return None, which will be filtered out by the collate_fn
                # return None
                index = np.random.randint(0, len(self))

        if not self.cfg.data.get("get_double_sample", False): 
            return self.post_process_item(sample, index)
        
        while True:
            try:
                other_sample = self.get_item((index + self.resume_step) % len(self))
                break
            except Exception as e:
                # import traceback;traceback.print_exc()
                # You can optionally log the error here
                skip_index = (index + self.resume_step) % len(self)
                print(f"Skipping index {skip_index} due to: {str(e)}")
                # Return None, which will be filtered out by the collate_fn
                # return None
                index = np.random.randint(0, len(self))

        for k, v in other_sample.items():
            sample[f"{k}_other"] = v
        return sample

# You can
import argparse
import multiprocessing
import os, shutil
import imageio
import pickle
import time
from tqdm import tqdm
from einops import rearrange
from accelerate.utils import set_seed

def save_video_as_image(func_args):
    pixel_values, v2i_dir, save_dir_item, meta_result = func_args
    for i, pixel_values_i in enumerate(pixel_values):
        Image.fromarray(pixel_values_i).save(os.path.join(v2i_dir, f"{i}.png"))
    with open(os.path.join(save_dir_item, "meta_result.pkl"), 'wb') as f:
        pickle.dump(meta_result, f) 

if __name__ == "__main__":
    # How to use this script ?
    # save_visual is default mode, save visual video, mask and i2v condition
    # /home/weili/miniconda3/envs/wan21_xc/bin/python 
    #    videox_fun/data/emo_video_live_body_cache.py 
    #    save cache ---------------------------------------------
    #    --save_cache --no_save_visual --dataset_file_path 

    #    save cache, not visual data ----------------------------
    #    --save_dir /home/weili/RealTimeV3Cache/20250429/
    #    --seed default 42
    #    --sample_times 3
    #    --start_id 0
    #    --end_id -1
    #    or not save cache, visual data ----------------------------
    #    --save_fps
    #    --save_origin_video 

    #    video dataloader args ------------------------------------------------
    #    --train_width default 544, when enable bucket, it is useless
    #    --train_height default 960, when enable bucket, it is useless
    #    --n_sample_frames default 77 
    #    --past_n default 4
    #    --train_fps default 25
    #    --ignore_hyperIQA # if haven't process this params, store true to ignore it !
    #    --driving_video_scale default 1.2
    #    --vx_ratio_crop default -0.20
    #    --vy_ratio_crop default 0
    #    --lip_open_ratio default 0.
    #    --hyperIQA_min default 30
    #    --audio_dyadic_conf_thresh 0 # suggest 6 for dyadic
    #    --body_score_thresh 3 
    #    --body_level 3 
    #    --num_workers default=24
    #    --writer_num_workers default=48
    #    --no_save_visual
    #    --save_origin_video

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    # cache data or visual data ----------------------------------------
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--sample_times", type=int, default=3)
    
    parser.add_argument("--no_save_visual", action="store_true")
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--save_fps", type=int, default=None)
    parser.add_argument("--save_origin_video", action="store_true")
    # video dataloader args ------------------------------------------------
    parser.add_argument("--train_width", type=int, default=544)
    parser.add_argument("--train_height", type=int, default=960)
    parser.add_argument("--n_sample_frames", type=int, default=77)
    parser.add_argument("--past_n", type=int, default=4)
    parser.add_argument("--train_fps", type=int, default=25)
    # LivePortrait args
    parser.add_argument("--driving_video_scale", type=float, default=1.2)
    parser.add_argument("--vx_ratio_crop", type=float, default=-0.20)
    parser.add_argument("--vy_ratio_crop", type=float, default=0)
    # HopeNet args
    parser.add_argument("--yaw_max", type=float, default=45)
    parser.add_argument("--pitch_max", type=float, default=40)
    parser.add_argument("--roll_max", type=float, default=30)
    parser.add_argument("--yaw_delta", type=float, default=40)
    parser.add_argument("--pitch_delta", type=float, default=25)
    parser.add_argument("--roll_delta", type=float, default=20)
    # HyperNet IQA args
    parser.add_argument("--ignore_hyperIQA", action="store_true")
    parser.add_argument("--hyperIQA_min", type=float, default=30)
    # Body part args
    parser.add_argument("--body_score_thresh", type=float, default=3)
    parser.add_argument("--body_level", type=float, default=3)
    # other args
    parser.add_argument("--lip_open_ratio", type=float, default=0)
    parser.add_argument("--audio_dyadic_conf_thresh", type=float, default=0)
    
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--writer_num_workers", type=int, default=48)

    args = parser.parse_args()
    
    if args.save_fps is None: args.save_fps = args.train_fps
        
    # config = "configs/train/head_animator_mix_LIA_visual.yaml"
    # config = OmegaConf.load(config)
    config = {"data": {}}
    set_seed(args.seed)
    # visual norm data
    dataset_file_path = args.dataset_file_path
    visual_dir_name = "liveBody_" + os.path.basename(dataset_file_path)[:-4]
    config["data"]["dataset_file_path"] = [[dataset_file_path, 1]]
    
    config["data"]["train_width"] = args.train_width
    config["data"]["train_height"] = args.train_height
    config["data"]["n_sample_frames"] = args.n_sample_frames
    config["data"]["past_n"] = args.past_n
    config["data"]["train_fps"] = args.train_fps
    
    config["data"]["mouth_bbox_scale"] = 1.5
    config["data"]["eye_bbox_scale"] = 2.0
    config["data"]["hybrid_face_mask"] = True
    config["data"]["union_bbox_scale"] = [1.2, 1.4]
    config["data"]["fliter_subtitle"] = True
    config["data"]["flip_aug"] = True
    config["data"]["driving_video_scale"] = args.driving_video_scale
    config["data"]["vx_ratio_crop"] = args.vx_ratio_crop
    config["data"]["vy_ratio_crop"] = args.vy_ratio_crop
    config["data"]["pose_max"] = [args.yaw_max, args.pitch_max, args.roll_max]
    config["data"]["pose_delta"] = [args.yaw_delta, args.pitch_delta, args.roll_delta]
    config["data"]["lip_open_ratio"] = args.lip_open_ratio
    config["data"]["hyperIQA_min"] = args.hyperIQA_min
    config["data"]["body_score_thresh"] = args.body_score_thresh
    config["data"]["body_level"] = args.body_level
    
    config["data"]["audio_dyadic_conf_thresh"] = args.audio_dyadic_conf_thresh
    del config["data"]["pose_max"], config["data"]["pose_delta"]
    if args.ignore_hyperIQA:
        del config["data"]["hyperIQA_min"]
    config = OmegaConf.create(config)
    train_dataset = LiveVideoDataset(
        width=config.data.train_width,
        height=config.data.train_height,
        cfg=config,
        split='train',
        enable_bucket=args.no_save_visual,
        resume_step=0,
        start_id=args.start_id,
        end_id=args.end_id,
        sample_times=args.sample_times,
        save_gt=True,
    )
    def custom_collate_fn(batch):
        # Filter out None values (failed samples)
        batch = [sample for sample in batch if sample is not None]
        if not batch:  # if batch is empty
            # Create a dummy batch with the same structure but zero tensors
            # You may need to adjust this based on your actual data structure
            return None
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
            # print(f"Collate error: {str(e)}")
            for sample in batch:
                for key, value in sample.items():
                    if key in [
                            "video_path",
                            "audio_path",
                            "audio_self_path",
                            "audio_other_path",
                            "clip_st",
                            "clip_et",]:
                        print(f"{key=} {value=}")
                    else:
                        print(f"{key=} {value.shape=}")
            import traceback;traceback.print_exc()
            return None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=8,
        # drop_last=True
    )
    if args.save_cache:
        writer_pool = multiprocessing.Pool(processes=args.writer_num_workers)
        writer_results = []
    
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # continue
        # del batch["video_path"]
        # for k, v in batch.items():
        #     print(k, v.shape, v.unique())
        # exit(0)
        video_path = batch["video_path"][0]
        folder = batch["folder"][0]
        name = batch["name"][0]
        pixel_values = batch["pixel_values"][0]
        face_id = batch["face_id"][0].cpu().item()
        crop_bbox = batch["crop_bbox"][0].cpu().numpy()
        clip_target_idx = batch["clip_target_idx"][0].cpu().numpy()
        if not args.no_save_visual:
            save_dir = visual_dir_name + f"_{args.driving_video_scale}"
            os.makedirs(save_dir, exist_ok=True)
            mask_pixel_values = batch["mask_pixel_values"][0]
            mask = batch["mask"][0]
            clip_pixel_values = batch["clip_pixel_values"][0]
            clip_pixel_values = clip_pixel_values.cpu().detach().numpy().astype("uint8")
            
            visual_list = []
            ref_img_original_vis = None
            target_vid_original_vis = []
            masked_ref_img_vis = None
            masked_target_vid_vis = []
            for (pixel_values_i, 
                mask_pixel_values_i, 
                mask_i, ) \
                in zip(pixel_values, 
                    mask_pixel_values, 
                    mask,):
                pixel_values_i = (((pixel_values_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
                mask_pixel_values_i = (((mask_pixel_values_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
                mask_i = (mask_i.repeat(3, 1, 1).cpu().numpy() * 255.).transpose(1, 2, 0).astype("uint8")
                visuals = np.concatenate([pixel_values_i, 
                                        mask_pixel_values_i, 
                                        mask_i,
                                        clip_pixel_values,
                                        ], axis=1)
                visual_list.append(visuals)
            
            video_base = os.path.basename(video_path)[:-4]
            save_path = f"./{save_dir}/{i}_{video_base}.mp4"
            imageio.mimwrite(save_path, visual_list, fps=args.save_fps)
        
        if args.save_origin_video:
            shutil.copy(video_path, save_dir)
        
        if args.save_cache:
            video_base = os.path.basename(video_path)[:-4]
            save_dir = os.path.join(args.save_dir, video_base)
            os.makedirs(save_dir, exist_ok=True)
            repeated_times = os.listdir(save_dir)
            cur_time = 0
            for repeated_time in repeated_times:
                cur_time = max(cur_time, int(repeated_time))
            if cur_time + 1 > args.sample_times:
                continue
            cur_time = str((cur_time + 1)).zfill(5)
            save_dir_item = os.path.join(save_dir, cur_time)
            os.makedirs(save_dir_item, exist_ok=True)
            
            v2i_dir = os.path.join(save_dir_item, "pixel_values")
            os.makedirs(v2i_dir, exist_ok=True)
            pixel_values = pixel_values.cpu().numpy().astype("uint8")
            B, height, width, C = pixel_values.shape
            meta_result = {
                "height": height,
                "width": width,
                "folder": folder,
                "name": name,
                "face_id": face_id,
                "crop_bbox": crop_bbox,
                "clip_target_idx": clip_target_idx,
            }
            func_args = (pixel_values, v2i_dir, save_dir_item, meta_result)
            writer_result = writer_pool.apply_async(save_video_as_image, args=(func_args,))
            writer_results.append(writer_result)
            while True:
                writer_results = [r for r in writer_results if not r.ready()]
                if len(writer_results) >= args.writer_num_workers + 2:
                    time.sleep(0.3)
                else:
                    break
    if args.save_cache:
        writer_pool.close()
        writer_pool.join()
        print(f"Finish cache all data")
            
            
             