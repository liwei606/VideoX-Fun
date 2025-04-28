from cmath import sqrt
import sys
import os
import urllib.request
import random
from glob import glob
from pathlib import Path
from typing import List
import pickle
import imageio
import matplotlib.pyplot as plt
import numpy as np
import orjson
import torch
import torchvision.transforms.v2 as transforms
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import ffmpeg
from decord import cpu, gpu
from time import time
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False) # NOTE: enforce single thread

CV2_INTERP = cv2.INTER_LINEAR
DTYPE = np.float32

sys.path.insert(0, str(Path(__file__).parent.parent))

from pycocotools import mask as maskUtils
def polygons_to_mask(polygons, height, width):
    """
    将 polygon 格式转换为二值 mask。

    Args:
        polygons: List of polygons（每个 polygon 是一组 [x1, y1, x2, y2, ..., xn, yn]）
        height: mask 高度（原图的高）
        width:  mask 宽度（原图的宽）

    Returns:
        numpy array, shape = (height, width), dtype=np.uint8
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)  # 合并多个 polygon 的 RLE
    mask = maskUtils.decode(rle)
    return mask

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

def convert_cuts_format(vid_meta):
    split_origin_dict = {}
    for folder, name in vid_meta:
        ori_name = folder + "_" + name.split("video-Scene-")[0]
        if ori_name not in split_origin_dict:
            split_origin_dict[ori_name] = []
        split_origin_dict[ori_name].append((folder, name))
    convert_list = []
    for folder, name in vid_meta:
        ori_name = folder + "_" + name.split("video-Scene-")[0]
        convert_list.append((folder, name, split_origin_dict[ori_name]))
    return convert_list

def dfs_object_random(frame_idxs, 
                      match_object_dict, 
                      cur_cut_name,
                      skip_self, 
                      score_down, 
                      score_up):
    obj_avaible_dict = {}
    for cut_name, cut_dict in match_object_dict.items():
        if cur_cut_name == cut_name and skip_self: continue
        for frame_idx in frame_idxs:
            if frame_idx not in cut_dict: continue
            frame_dict = cut_dict[frame_idx]
            for other_frame_idx, other_frame_dict in frame_dict.items():
                for obj_idx, obj_dict in other_frame_dict.items():
                    for other_obj_idx, match_score in obj_dict.items():
                        if match_score >= score_down and match_score <= score_up:
                            obj_avaible_dict[obj_idx] = [] if obj_idx not in obj_avaible_dict else obj_avaible_dict[obj_idx]
                            obj_avaible_dict[obj_idx].append((cut_name, other_frame_idx, other_obj_idx))
    for obj_idx in obj_avaible_dict:
        obj_avaible_dict[obj_idx] = random.choice(obj_avaible_dict[obj_idx])
    return obj_avaible_dict

class LiveVideoDataset(Dataset):
    def __init__(
        self,
        cfg=None,
        width=512,
        height=512,
        split='train',
        resume_step=0,
        test_size=100,
        save_gt=False,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_gt = save_gt
        self.img_size = (height, width)
        self.width = width
        self.height = height
        self.split = split
        self.drop_n_frame = 3 # drop first and last [drop_n_frame] frame to filter out unreasonable scene transition
        self.zero_to_one = self.cfg.data.get('zero_to_one', False)
        self.clip_object_sim_up = self.cfg.data.get("clip_object_sim_up", 0.95)
        self.clip_object_sim_down = self.cfg.data.get("clip_object_sim_down", 0.75)
        self.arc_face_sim_down = self.cfg.data.get("arc_face_sim_down", 0.75)

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
            # cur_res = convert_cuts_format(cur_res)
            print(f"Find {len(cur_res)} item in {dataset_file_path}, weights is {weights}")
            vid_path += cur_res * weights
        
        #### for debug 
        self.debug = cfg.data.get("debug", False)
        if self.debug:
            # vid_path = [vid_path[134]] + [vid_path[138]]
            # vid_path = vid_path[:10]
            vid_path = [vid_path[9]]
            print('Debug Mode!!!')

        self.target_fps = cfg.data.train_fps
        self.random_sample = cfg.data.random_sample
        print(f"origin {len(vid_path)=}")
        
        if split == 'train':
            self.dataset = vid_path * int(5e7 // len(vid_path))
        else:
            self.dataset = vid_path
            if "pose_max" in cfg.data: del cfg.data["pose_max"]
            if "pose_delta" in cfg.data: del cfg.data["pose_delta"]
            print(f"{self.dataset[::5]=}")
        print(f"finish repeat vid_path{len(self.dataset)=}")
        
        self.pixel_transform = transforms.Compose(
            [    
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

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
        ori_name, cur_cut_name = name.split("video-Scene-")
        
        is_anime_data = False
        if ("anime" in folder) or ("Anime" in folder):
            is_anime_data = True
        vid_path = f'{folder}/videos_resampled/{name}+resampled.mp4'
        metadata_file = f"{folder}/metadata/{name}/metadata_mmpose.pkl"
        if not os.path.exists(metadata_file):
            metadata_file = f"{folder}/metadata/{name}/metadata.pkl"
        if not os.path.exists(vid_path):
            vid_path = f'{folder}/videos/{name}.mp4'
        with open(metadata_file, 'rb') as f:
            video_metadata = pickle.load(f)
        
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
        
        match_object_dict = video_metadata["match_object_dict"]
        match_face_dict = video_metadata["match_face_dict"]
        
        obj_select_dict = dfs_object_random(target_frame_indices_new[tgt_batch_index], 
                                            match_object_dict,
                                            cur_cut_name,
                                            True,
                                            self.clip_object_sim_down,
                                            self.clip_object_sim_up)
        face_select_dict = dfs_object_random(target_frame_indices_new[tgt_batch_index], 
                                            match_face_dict,
                                            cur_cut_name,
                                            True,
                                            self.arc_face_sim_down,
                                            1)
        object_refers = []
        for obj_select_key, obj_select_value in obj_select_dict.items():
            cut_name, other_frame_idx, other_obj_idx = obj_select_value
            other_name = f"{ori_name}video-Scene-{cut_name}"
            other_vid_path = f'{folder}/videos_resampled/{other_name}+resampled.mp4'
            other_metadata_file = f"{folder}/metadata/{other_name}/metadata_mmpose.pkl"
            if not os.path.exists(other_metadata_file):
                other_metadata_file = f"{folder}/metadata/{other_name}/metadata.pkl"
            with open(other_metadata_file, 'rb') as f:
                other_video_metadata = pickle.load(f)
            other_frame = VideoReader(other_vid_path, ctx=cpu(0))[other_frame_idx].asnumpy()
            ohd, owd, _ = other_frame.shape
            other_xy = other_video_metadata["objs_segments"][other_frame_idx][other_obj_idx]
            polygon_coco = [other_xy.flatten().tolist()]
            other_object = other_frame * polygons_to_mask(polygon_coco, ohd, owd)[:, :, None]
            object_refers.append(other_object)
            
        face_refers = []
        for face_select_key, face_select_dict in face_select_dict.items():
            cut_name, other_frame_idx, other_face_id = face_select_dict
            other_name = f"{ori_name}video-Scene-{cut_name}"
            other_vid_path = f'{folder}/videos_resampled/{other_name}+resampled.mp4'
            other_metadata_file = f"{folder}/metadata/{other_name}/metadata_mmpose.pkl"
            if not os.path.exists(other_metadata_file):
                other_metadata_file = f"{folder}/metadata/{other_name}/metadata.pkl"
            with open(other_metadata_file, 'rb') as f:
                other_video_metadata = pickle.load(f)
            other_frame = VideoReader(other_vid_path, ctx=cpu(0))[other_frame_idx].asnumpy()
            ohd, owd, _ = other_frame.shape
            x_min, y_min, x_max, y_max = other_video_metadata["frame_data"]["bounding_box"][other_face_id][other_frame_idx]
            other_face = other_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            face_refers.append(other_face)
            
                        
        # Read target frames
        def get_imgs_from_idx(idx):
            bs_img = video_reader.get_batch(target_frame_indices_new[idx]).asnumpy()
            return [Image.fromarray(bs_img[idxl]) for idxl in range(len(bs_img))]
        
        try:
            # using gpu 
            vid_pil_image_past = get_imgs_from_idx(past_batch_index)
            vid_pil_image_list = get_imgs_from_idx(tgt_batch_index)
        except:
            # when fail to retrive frames on gpu 
            video_reader = VideoReader(vid_path, ctx=cpu(0))
            vid_pil_image_past = get_imgs_from_idx(past_batch_index)
            vid_pil_image_list = get_imgs_from_idx(tgt_batch_index)

        for vid_pil_image in vid_pil_image_list:
            assert np.array(vid_pil_image).mean() >= 0.2, "Meet all black frames, skip It !!!"

        ## crop image
        
        vid_pil_image_list = [np.array(img) for img in vid_pil_image_list]
        vid_pil_image_past = [np.array(img) for img in vid_pil_image_past]
        pixel_values = np.array(vid_pil_image_past + vid_pil_image_list)
        if len(object_refers) == 0:
            object_refers = [np.zeros_like(vid_pil_image_list[0])]
        else:
            object_refers = [np.array(Image.fromarray(x).resize((wd, hd))) for x in object_refers]
        if len(face_refers) == 0:
            face_refers = [np.zeros_like(vid_pil_image_list[0])]
        else:
            object_refers = [np.array(Image.fromarray(x).resize((wd, hd))) for x in object_refers]
        sample = dict(
            video_path = video_path,
            pixel_values = pixel_values,
            face_refers = face_refers,
            object_refers = object_refers,
            text = "",
            data_type = data_type,
            idx = index,
            
        )

        # FLIP_FLAG = False
        # if self.flip_aug and random.random() < 0.5:
        #     vid_pil_image_list = [np.flip(img, axis=1).copy() for img in vid_pil_image_list]
        #     vid_pil_image_past = [np.flip(img, axis=1).copy() for img in vid_pil_image_past]
        #     # union_mask_img = np.flip(union_mask_img, axis=1).copy()
        #     FLIP_FLAG = True
        
        # Transform
        # state = torch.get_rng_state()

        # pixel_values_vid_original = self.augmentation(vid_pil_image_list, self.pixel_transform, state)
        # pixel_values_past_frames_original = self.augmentation(vid_pil_image_past, self.pixel_transform, state)
        # if len(object_refers) == 0:
        #     object_refers = [np.zeros_like(vid_pil_image_list[0])]
        # if len(face_refers) == 0:
        #     face_refers = [np.zeros_like(vid_pil_image_list[0])]
        # pixel_values_object_refers = self.augmentation(object_refers, self.pixel_transform, state)
        # pixel_values_face_refers = self.augmentation(face_refers, self.pixel_transform, state)
        

        # assert pixel_values_vid_original.min() < 0 and pixel_values_vid_original.max() <= 1
        # assert pixel_values_past_frames_original.min() < 0 and pixel_values_past_frames_original.max() <= 1
        
        # # Make sure this feature is right: Get target wav features

        # clip_st = target_frame_indices_new[past_batch_index[0]] / original_fps
        # clip_et = target_frame_indices_new[tgt_batch_index[-1]] / original_fps
        # wav_st = int(clip_st * target_fps)
        # wav_et = int(clip_et * target_fps)
        
        # ## Add wav_feat here
        # # if wav_et >= len(wav_fea):
        # #     wav_et = len(wav_fea)
        # # target_wav_fea = wav_fea[wav_et - self.n_sample_frames : wav_et]
        # assert len(pixel_values_vid_original) == self.n_sample_frames and len(pixel_values_past_frames_original) == self.past_n, \
        #     "pixel_values cannot meet length threshold"
        
        # if os.path.exists(video_path.replace("mp4", "wav")):
        #     gt_filt_audio = video_path.replace("mp4", "wav")
        # elif os.path.exists(video_path.replace("+resampled.mp4", "+audio.wav")):
        #     gt_filt_audio = video_path.replace("+resampled.mp4", "+audio.wav")
        # else:
        #     gt_filt_audio = video_path.replace("+resampled.mp4", "+audiov4.wav")
        # sample = dict(
        #     video_path=video_path,
        #     pixel_values_vid_original=pixel_values_vid_original,
        #     pixel_values_past_frames_original=pixel_values_past_frames_original,
        #     pixel_values_object_refers=pixel_values_object_refers,
        #     pixel_values_face_refers=pixel_values_face_refers,
            
        # )

        # if self.save_gt:
        #     if os.path.exists(video_path.replace("mp4", "wav")):
        #         gt_filt_audio = video_path.replace("mp4", "wav")
        #     elif os.path.exists(video_path.replace("+resampled.mp4", "+audio.wav")):
        #         gt_filt_audio = video_path.replace("+resampled.mp4", "+audio.wav")
        #     else:
        #         gt_filt_audio = video_path.replace("+resampled.mp4", "+audiov4.wav")
        #     audio_self_path = ""
        #     audio_other_path = ""
        #     if "face_id_pair_audiov3" in video_metadata:
        #         face_audio_pair = video_metadata["face_id_pair_audiov3"]
        #         self_face_id, other_face_id = face_audio_pair.keys()
        #         if face_id != self_face_id:
        #             self_face_id, other_face_id = other_face_id, self_face_id
        #         audio_self_path = video_path.replace("+resampled.mp4", f"+audio_v3_{self_face_id}.wav")
        #         audio_other_path = video_path.replace("+resampled.mp4", f"+audio_v3_{other_face_id}.wav")
        #     sample.update(
        #         dict(
        #             video_path=video_path,
        #             audio_path=gt_filt_audio,
        #             audio_self_path=audio_self_path,
        #             audio_other_path=audio_other_path,
        #             clip_st=clip_st,
        #             clip_et=clip_et,
        #         )
        #     )

        return sample

    def __getitem__(self, index):
        # return self.get_item(index)
        start_folder, start_data_type = self.dataset[index]
        while True:
            try:
                folder, data_type = self.dataset[index]
                if start_data_type != data_type: continue
                sample = self.get_item(index)
                break
            except Exception as e:
                # import traceback;traceback.print_exc()
                # You can optionally log the error here
                skip_index = index
                print(f"Skipping index {skip_index} due to: {str(e)}")
                # Return None, which will be filtered out by the collate_fn
                # return None
                index = np.random.randint(0, len(self))
        return sample

# You can
import argparse
import os, shutil
from tqdm import tqdm
from einops import rearrange
from lightning import seed_everything
if __name__ == "__main__":
    # RUN_FUNC = os.environ["RUN_FUNC"]
    # RUN_FUNC = 'test_fast_video_dataset'
    # print(f"{RUN_FUNC=}")
    # eval(RUN_FUNC)()
    # /home/weili/miniconda3/envs/wan21_xc/bin/python 
    #    videox_fun/data/segobj_dataloader.py 
    #    --dataset_file_path 
    #    --visual_type norm cache
    #    --n_sample_frames 77
    #    --past_n 4 or default=4
    #    --train_fps 25 or default=8
    #    --save_fps 1 or train_fps
    #    --get_data_type animation
    #    --seed 
    #    --past_frame_is_ref 
    #    --ignore_hope # if haven't process this params, store true to ignore it !
    #    --ignore_hyperIQA # if haven't process this params, store true to ignore it !
    #    --driving_video_scale 1.65
    #    --lip_open_ratio 0.
    #    --hyperIQA_min 0. # suggest 45
    #    --audio_dyadic_conf_thresh 0 # suggest 6 for dyadic
    #    --num_workers default=32
    #    --no_save_visual
    #    --save_origin_video
    #    --save_split # store true, suggest for dyadic
    #    --save_lip_visual # visual lip movements value

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--visual_type", type=str, default="norm")
    parser.add_argument("--train_width", type=int, default=544)
    parser.add_argument("--train_height", type=int, default=960)

    parser.add_argument("--n_sample_frames", type=int, default=77)
    parser.add_argument("--past_n", type=int, default=4)
    parser.add_argument("--train_fps", type=int, default=25)
    parser.add_argument("--save_fps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--past_frame_is_ref", action="store_true")
    
    parser.add_argument("--ignore_hope", action="store_true")
    parser.add_argument("--ignore_hyperIQA", action="store_true")
    parser.add_argument("--hyperIQA_min", type=float, default=0)
    parser.add_argument("--lip_open_ratio", type=float, default=0)
    parser.add_argument("--audio_dyadic_conf_thresh", type=float, default=0)

    parser.add_argument("--get_data_type", type=str, default="animation")
    parser.add_argument("--no_save_visual", action="store_true")
    parser.add_argument("--save_origin_video", action="store_true")
    parser.add_argument("--save_split", action="store_true")
    parser.add_argument("--save_lip_visual", action="store_true")
    
    # Parameters to sample video -----------------------------
    parser.add_argument("--driving_video_scale", type=float, default=1.65)
    parser.add_argument("--yaw_max", type=float, default=45)
    parser.add_argument("--pitch_max", type=float, default=40)
    parser.add_argument("--roll_max", type=float, default=30)
    parser.add_argument("--yaw_delta", type=float, default=40)
    parser.add_argument("--pitch_delta", type=float, default=25)
    parser.add_argument("--roll_delta", type=float, default=20)
    
    parser.add_argument("--num_workers", type=int, default=32)

    args = parser.parse_args()
    
    if args.save_fps is None: args.save_fps = args.train_fps
        
    config = "/mnt/weka/hw_workspace/sr_workspace/real-time-video-gen/configs/train/head_animator_mix_LIA_visual.yaml"
    config = OmegaConf.load(config)
    seed_everything(args.seed)
    if args.visual_type == "norm":
        # visual norm data
        dataset_file_path = args.dataset_file_path
        visual_dir_name = "SEGOBJECT_" + os.path.basename(dataset_file_path)[:-4]
        config["data"]["dataset_file_path"] = [[dataset_file_path, 1]]
        
    if args.visual_type == "cache":
        # visual cache data
        cache_file_path = args.dataset_file_path
        visual_dir_name = "SEGOBJECT_" + os.path.basename(cache_file_path)
        config["data"]["dataset_file_path"] = []
        config["data"]["cache_file_path"] = [[cache_file_path, 1]]
    config["data"]["train_width"] = args.train_width
    config["data"]["train_height"] = args.train_height

    config["data"]["n_sample_frames"] = args.n_sample_frames
    config["data"]["past_n"] = args.past_n
    config["data"]["train_fps"] = args.train_fps
    config["data"]["visual_dir"] = visual_dir_name
    config["data"]["driving_video_scale"] = args.driving_video_scale
    config["data"]["pose_max"] = [args.yaw_max, args.pitch_max, args.roll_max]
    config["data"]["pose_delta"] = [args.yaw_delta, args.pitch_delta, args.roll_delta]
    config["data"]["past_frame_is_ref"] = args.past_frame_is_ref
    config["data"]["lip_open_ratio"] = args.lip_open_ratio
    config["data"]["hyperIQA_min"] = args.hyperIQA_min
    config["data"]["audio_dyadic_conf_thresh"] = args.audio_dyadic_conf_thresh
    config["data"]["get_data_type"] = args.get_data_type
    if args.ignore_hope:
        del config["data"]["pose_max"], config["data"]["pose_delta"]
    if args.ignore_hyperIQA:
        del config["data"]["hyperIQA_min"]
    train_dataset = LiveVideoDataset(
        width=config.data.train_width,
        height=config.data.train_height,
        cfg=config,
        split='train',
        resume_step=0,
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
        batch_size=config.data.train_bs,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        # persistent_workers=True,
        # prefetch_factor=8,
        # drop_last=True
    )
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # continue
        # del batch["video_path"]
        # for k, v in batch.items():
        #     print(k, v.shape, v.unique())
        # exit(0)
        # when not zero_to_one, all of bellow is [-1, 1]

        target_vid_original = batch['pixel_values']
        # past_frames = batch['pixel_values_past_frames_original']
        object_refers = batch["face_refers"]
        face_refers = batch["object_refers"]
        # target_vid_original = torch.cat([past_frames, target_vid_original], dim=1)
        
        face_refer = face_refers[:, 0, ...][:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        face_refer = rearrange(face_refer, "b t c h w -> (b t) c h w")
        object_refers_vis = []
        for object_refer in object_refers:
            object_refer = object_refer[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
            object_refer = rearrange(object_refer, "b t c h w -> (b t) c h w")
            object_refers_vis.append(object_refer)
        
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")
        

        # ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        # ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        # target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        
        video_path = batch["video_path"][0]
        audio_path = batch["audio_path"][0]
        clip_st = batch["clip_st"][0]
        clip_et = batch["clip_et"][0]
        audio_self_path = batch["audio_self_path"][0]
        audio_other_path = batch["audio_other_path"][0]
        print(f"{video_path=}")
        print(f"{audio_self_path=}")
        print(f"{audio_other_path=}")
        
        import imageio
        visual_list = []
        ref_img_original_vis = None
        target_vid_original_vis = []
        masked_ref_img_vis = None
        masked_target_vid_vis = []
        for j, (
            # ref_img_original_i, 
            target_vid_original_i, 
            face_refer_i
            ) \
            in enumerate(zip(
                #    ref_img_original, 
                   target_vid_original, 
                   face_refer,
                   )):
            # target_vid_original_i = (((target_vid_original_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
            # object_refer_i = [(((object_refer[j].cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8") for object_refer in object_refers_vis]
            # face_refer_i = (((face_refer_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
            target_vid_original_i = target_vid_original_i.cpu().numpy().astype("uint8")
            object_refer_i = [object_refer[j].cpu().numpy().astype("uint8") for object_refer in object_refers_vis]
            face_refer_i = face_refer_i.cpu().numpy().astype("uint8")
                
            visuals = np.concatenate([
                                      target_vid_original_i, 
                                      face_refer_i
                                    ] + object_refer_i, axis=1)
            visual_list.append(visuals)
        
        if args.visual_type == "norm":
            video_base = os.path.basename(video_path)[:-4]
            save_path = f"./{config.data.visual_dir}/{i}_{video_base}.mp4"
        if args.visual_type == "cache":
            video_base, video_times = os.path.basename(str(Path(video_path).parent)), os.path.basename(video_path)
            save_path = f"./{config.data.visual_dir}/{i}_{video_base}_{video_times}.mp4"
        os.makedirs(config.data.visual_dir, exist_ok=True)
        if args.save_origin_video:
            shutil.copy(video_path, config.data.visual_dir)
        if not args.no_save_visual:
            imageio.mimwrite(save_path, visual_list, fps=config.data.train_fps)
        if args.save_lip_visual:
            lip_visuals = []
            for frame, batch_lip_movement in zip(visual_list, batch_lip_movements):
                draw_frame = frame.copy()
                cv2.putText(draw_frame, str(batch_lip_movement.item()), (20, 0 * 100 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 0, 200), 2)
                lip_visuals.append(draw_frame)
            save_path = f"./{config.data.visual_dir}/{i}_{video_base}_lip_visual.mp4"
            imageio.mimwrite(save_path, lip_visuals, fps=config.data.train_fps)
            
        if args.save_split:
            Image.fromarray(ref_img_original_vis).save(f"./{config.data.visual_dir}/{i}_{video_base}_ref_img.png")
            Image.fromarray(masked_ref_img_vis).save(f"./{config.data.visual_dir}/{i}_{video_base}_masked_ref.png")
            target_vid_save_path_tmp = f"./{config.data.visual_dir}/{i}_{video_base}_target_vid_no_audio.mp4"
            target_vid_save_path = f"./{config.data.visual_dir}/{i}_{video_base}_target_vid.mp4"
            imageio.mimwrite(target_vid_save_path_tmp,
                             target_vid_original_vis,
                             fps=config.data.train_fps)
            imageio.mimwrite(f"./{config.data.visual_dir}/{i}_{video_base}_masked_target_vid.mp4",
                             masked_target_vid_vis,
                             fps=config.data.train_fps)
            
            audio_clip_path = f"./{config.data.visual_dir}/{i}_{video_base}.wav"
            ffmpeg.input(audio_path, ss=clip_st, to=clip_et).audio.output(audio_clip_path).run()
            if audio_self_path != "":
                audio_self_clip_path = f"./{config.data.visual_dir}/{i}_{video_base}_self.wav"
                ffmpeg.input(audio_self_path, ss=clip_st, to=clip_et).audio.output(audio_self_clip_path).run()
            if audio_other_path != "":
                audio_other_clip_path = f"./{config.data.visual_dir}/{i}_{video_base}_other.wav"
                ffmpeg.input(audio_other_path, ss=clip_st, to=clip_et).audio.output(audio_other_clip_path).run()
            
            ffmpeg.output(
                (ffmpeg.input(target_vid_save_path_tmp)).video,
                (ffmpeg.input(audio_clip_path)).audio,
                target_vid_save_path,
                vcodec="copy",
                acodec="aac",
                shortest=None,
                loglevel="error",
            ).run()
            os.remove(target_vid_save_path_tmp)
        # visual -----------------------------------