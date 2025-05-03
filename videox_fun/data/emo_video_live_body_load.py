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
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from decord import cpu
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from videox_fun.data.emo_video_live_body_cache import convert_bbox_to_square_bbox, get_mask, scale_bbox
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False) # NOTE: enforce single thread

CV2_INTERP = cv2.INTER_LINEAR
DTYPE = np.float32

def get_cache_file_list(cache_file_path, data_type):
    if not os.path.exists(cache_file_path):
        return []
    results = []
    video_names = os.listdir(cache_file_path)
    for video_name in video_names:
        video_total_path = os.path.join(cache_file_path, video_name)
        item_paths = [(os.path.join(video_total_path, x), data_type) for x in os.listdir(video_total_path)]
        results.extend(item_paths)
    return results

def get_audio_features(audio_processor, audio_path, start_time, end_time):
    sr = 16000
    audio_input, sample_rate = librosa.load(audio_path, sr=sr)  # 采样率为 16kHz

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    try:
        audio_segment = audio_input[start_sample: end_sample]
    except:
        audio_segment = audio_input

    input_values = audio_processor(
        audio_segment, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values.cpu()
    return input_values


class LiveVideoLoadDataset(Dataset):
    def __init__(
        self,
        cfg=None,
        width=512,
        height=512,
        split='train',
        enable_bucket=False,
        resume_step=0,
        save_gt=False,
        wav2vec_processor=None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_gt = save_gt
        assert enable_bucket, "use LiveVideoLoadDataset, must enable_bucket"
        
        self.wav2vec_processor = wav2vec_processor
        if wav2vec_processor is not None:
            self.ai2v_enable = True
        else:
            self.ai2v_enable = False

        cache_file_path_list = cfg.data.get("cache_file_path", [])
        self.resume_step = resume_step
        
        vid_path = []
        for cache_file_path, weights, data_type in cache_file_path_list:
            cur_res = get_cache_file_list(cache_file_path, data_type)
            print(f"Find {len(cur_res)} item in {cache_file_path}, weights is {weights}")
            vid_path += cur_res * weights
        
        self.dataset = vid_path * int(5e7 // len(vid_path))

    def __len__(self):
        return len(self.dataset)

    def get_item(self, index):
        folder, data_type = self.dataset[index]
        is_anime_data = ("anime" in folder or "Anime" in folder)
        pixel_values_dir = os.path.join(folder, "pixel_values")
        img_list = sorted([os.path.join(pixel_values_dir, x) for x in os.listdir(pixel_values_dir)], key=lambda y: int(os.path.basename(y)[:-4]))
        pixel_values = []
        for img_path in img_list:
            pixel_values.append(np.array(Image.open(img_path)))
        pixel_values = np.array(pixel_values)
        # text = "A person is speaking with lively facial expressions and clearly synchronized lip movements. The person's emotion remains calm throughout. The camera and the background behind the person are perfectly static, with no shaking or movement. The person's body movements are natural, with occasional gentle hand gestures. The video is highly realistic and sharp, with clear details on the face and hands."
        text = "A person is speaking."
        with open(os.path.join(folder, "meta_result.pkl"), 'rb') as f:
            meta_result = pickle.load(f)
        folder, name = meta_result["folder"], meta_result["name"]
        
        metadata_file = f"{folder}/metadata/{name}/metadata_mmpose.npz"
        if not os.path.exists(metadata_file):
            metadata_file = f"{folder}/metadata/{name}/metadata.npz"
        video_metadata = dict(np.load(metadata_file, allow_pickle=True))["arr_0"].item()
        video_path = f'{folder}/videos_resampled/{name}+resampled.mp4'
        if not os.path.exists(video_path):
            video_path = f'{folder}/videos/{name}.mp4'
        video_reader = VideoReader(video_path)
        original_fps = video_reader.get_avg_fps()
        hd, wd, _ = video_reader[0].asnumpy().shape
        
        clip_target_idx = meta_result["clip_target_idx"].tolist()
        clip_st = clip_target_idx[0] / original_fps
        clip_et = (clip_target_idx[-1] + 1) / original_fps
        face_id = meta_result["face_id"]
        crop_bbox = meta_result["crop_bbox"]
        
        center_x = (crop_bbox[0] + crop_bbox[2]) / 2
        center_y = (crop_bbox[1] + crop_bbox[3]) / 2
        updated_h, updated_w = int((crop_bbox[3] - crop_bbox[1])), int((crop_bbox[2] - crop_bbox[0]))
        face_keypoints = video_metadata['frame_data']['keypoints'][face_id][clip_target_idx]
        face_keypoints[:, :, 0] *= wd
        face_keypoints[:, :, 1] *= hd
        face_keypoints[:, :, 0] += updated_w / 2 - center_x
        face_keypoints[:, :, 1] += updated_h / 2 - center_y
        face_keypoints[:, :, 0] = np.clip(face_keypoints[:, :, 0], 0, updated_w)
        face_keypoints[:, :, 1] = np.clip(face_keypoints[:, :, 1], 0, updated_h)
        
        def get_face_box(landmark_name, face_landmarks, max_w, max_h, eye_bbox_scale=1.5):
            if not is_anime_data:
                if landmark_name == 'left_eye':
                    landmarks = [398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
                elif landmark_name == 'right_eye':
                    landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                elif landmark_name == 'mouth':
                    landmarks = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321, 321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267, 269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81, 81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308]
            else:
                if landmark_name == 'left_eye':
                    landmarks = [18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42]
                elif landmark_name == 'right_eye':
                    landmarks = [23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48]
                elif landmark_name == 'mouth':
                    landmarks = list(range(49, 69))
                landmarks = [x - 1 for x in landmarks] 
            
            landmarks_x = [int(np.clip(face_landmarks[idx][0], 0, max_w)) for idx in landmarks]
            landmarks_y = [int(np.clip(face_landmarks[idx][1], 0, max_h)) for idx in landmarks]
            bbox = [(min(landmarks_x), min(landmarks_y)), (max(landmarks_x), max(landmarks_y))]
            bbox = convert_bbox_to_square_bbox(bbox, max_h, max_w, scale=eye_bbox_scale)
            return bbox, (max(landmarks_x) - min(landmarks_x)), (max(landmarks_y) - min(landmarks_y))
        
        if self.ai2v_enable:
            audio_path = f'{folder}/videos_resampled/{name}+audio.wav'
            if not os.path.exists(audio_path):
                audio_path = f'{folder}/videos_resampled/{name}+audiov4.wav'
            assert os.path.exists(audio_path)
            audio_feature = get_audio_features(self.wav2vec_processor, 
                                               audio_path,
                                               clip_st,
                                               clip_et,)
            assert audio_feature.size(1) == 51840, f"{audio_feature.size(1)=} != 51840"
            # if audio_feature.size(1) != 51840:
            #     print(f"{audio_feature.shape=} {audio_path=} {clip_st=} {clip_et=} {clip_target_idx=}")
        mouth_bboxs = []
        left_eye_bboxs = []
        right_eye_bboxs = []
        left_eye_masks = []
        right_eye_masks = []
        mouth_masks = []
        for kps in face_keypoints:
            # Get eye, mouth, scaling mask
            left_eye_bbox, xbz, ybz = get_face_box('left_eye', kps, updated_w, updated_h, self.cfg.data.eye_bbox_scale)
            right_eye_bbox, xbz, ybz = get_face_box('right_eye', kps, updated_w, updated_h, self.cfg.data.eye_bbox_scale)
            mouth_bbox, xbz, ybz = get_face_box('mouth',  kps, updated_w, updated_h, self.cfg.data.mouth_bbox_scale)
            left_eye_bboxs.append(left_eye_bbox)
            right_eye_bboxs.append(right_eye_bbox)
            mouth_bboxs.append(mouth_bbox)
            left_eye_masks.append(np.array(get_mask(left_eye_bbox, updated_h, updated_w, scale=1.0)))
            right_eye_masks.append(np.array(get_mask(right_eye_bbox, updated_h, updated_w, scale=1.0)))
            mouth_masks.append(np.array(get_mask(mouth_bbox, updated_h, updated_w, scale=1.0)))
        
        mouth_masks = np.array(mouth_masks)
        union_left_eye_masks = np.array(get_mask(np.array(left_eye_bboxs).max(axis=0), updated_h, updated_w, scale=1.0))
        union_right_eye_masks = np.array(get_mask(np.array(right_eye_bboxs).max(axis=0), updated_h, updated_w, scale=1.0))
        union_mouth_masks = np.array(get_mask(np.array(mouth_bboxs).max(axis=0), updated_h, updated_w, scale=1.0))
        
        sample = {
            "pixel_values": pixel_values,
            "union_mouth_masks": union_mouth_masks,
            "mouth_masks": mouth_masks,
            "text": text,
            "data_type": data_type,
            "idx": index,
            "clip_st": clip_st,
            "clip_et": clip_et,
        }
        sample["video_path"] = video_path
        if self.ai2v_enable:
            sample["audio_path"] = audio_path
            sample["audio_feature"] = audio_feature[0]
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
                skip_index = index
                print(f"Skipping index {skip_index} due to: {str(e)}")
                index = np.random.randint(0, len(self))

        return sample

# You can
import argparse
import multiprocessing
import os, shutil
import imageio
import pickle
import time
import ffmpeg
from tqdm import tqdm
from einops import rearrange
from accelerate.utils import set_seed

def save_video(func_args):
    # import pdb; pdb.set_trace()
    save_path, visual_list, fps, audio_path, clip_st, clip_et = func_args
    audio_clip_path = save_path[:-4] + ".wav"
    save_path_tmp = save_path[:-4] + "_tmp.mp4"
    imageio.mimwrite(save_path_tmp, visual_list, fps=fps)
    ffmpeg.input(audio_path, ss=clip_st, to=clip_et).audio.output(audio_clip_path).run(overwrite_output=True)
    ffmpeg.output(
        (ffmpeg.input(save_path_tmp)).video,
        (ffmpeg.input(audio_clip_path)).audio,
        save_path,
        vcodec="copy",
        acodec="aac",
        shortest=None,
        loglevel="error",
    ).run(overwrite_output=True)
    os.remove(save_path_tmp)
    os.remove(audio_clip_path)

if __name__ == "__main__":
    # /home/weili/miniconda3/envs/wan21_xc/bin/python 
    #    videox_fun/data/emo_video_live_body_load.py 
    #    --dataset_file_path 
    #    --data_type default video
    #    --save_fps default 25
    #    --seed default 42
    #    --save_origin_video 
    #    --eye_bbox_scale default 1.5
    #    --mouth_bbox_scale default 2.0
    #    --num_workers 32
    #    --writer_num_workers 32

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--wav2vec_model_dir", type=str, default="models/wav2vec2-base-960h",)
    parser.add_argument("--data_type", type=str, default="video")
    
    parser.add_argument("--save_fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_origin_video", action="store_true")
    
    # mask scale args
    parser.add_argument("--eye_bbox_scale", type=float, default=1.5)
    parser.add_argument("--mouth_bbox_scale", type=float, default=2.0)
    
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--writer_num_workers", type=int, default=32)

    args = parser.parse_args()
    
    config = {"data": {}}
    set_seed(args.seed)
    # visual norm data
    dataset_file_path = args.dataset_file_path
    visual_dir_name = "liveLoad_" + os.path.basename(dataset_file_path)
    config["data"]["cache_file_path"] = [[dataset_file_path, 1, args.data_type]]
    config["data"]["eye_bbox_scale"] = args.eye_bbox_scale
    config["data"]["mouth_bbox_scale"] = args.mouth_bbox_scale
    config = OmegaConf.create(config)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
    train_dataset = LiveVideoLoadDataset(
        # width=config.data.train_width,
        # height=config.data.train_height,
        cfg=config,
        split='train',
        enable_bucket=True,
        wav2vec_processor=wav2vec_processor,
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
    if args.num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        persistent_workers = True
        prefetch_factor = 8
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        # drop_last=True
    )
    writer_pool = multiprocessing.Pool(processes=args.writer_num_workers)
    writer_results = []
    
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # continue
        # del batch["video_path"]
        # for k, v in batch.items():
        #     print(k, v.shape, v.unique())
        # exit(0)
        video_path = batch["video_path"][0]
        audio_path = batch["audio_path"][0]
        audio_feature = batch["audio_feature"]
        pixel_values = batch["pixel_values"][0]
        union_mouth_masks = batch["union_mouth_masks"][0]
        mouth_masks = batch["mouth_masks"][0]
        clip_st = batch["clip_st"][0].cpu().item()
        clip_et = batch["clip_et"][0].cpu().item()
        # import pdb; pdb.set_trace()

        save_dir = visual_dir_name
        os.makedirs(save_dir, exist_ok=True)
        
        union_mouth_masks = union_mouth_masks.cpu().numpy().astype("uint8")
        mouth_masks = mouth_masks.cpu().numpy().astype("uint8")
        visual_list = []
        for pixel_values_i, mouth_masks_i in zip(pixel_values, mouth_masks):
            pixel_values_i = pixel_values_i.cpu().numpy().astype("uint8")
            visual_img = np.concatenate([
                pixel_values_i, 
                pixel_values_i * (union_mouth_masks == 255),
                union_mouth_masks,
            ], axis=1)
            # visual_img = np.concatenate([
            #     pixel_values_i, 
            #     pixel_values_i * (mouth_masks_i == 255),
            #     mouth_masks_i,
            #     union_mouth_masks,
            # ], axis=1)
            visual_list.append(visual_img.astype("uint8"))
        
        video_base = os.path.basename(video_path)[:-4]
        save_path = f"./{save_dir}/{i}_{video_base}.mp4"
        
        if args.save_origin_video:
            shutil.copy(video_path, save_dir)
        func_args = (save_path, visual_list, args.save_fps, audio_path, clip_st, clip_et)
        # save_video(func_args)
        writer_result = writer_pool.apply_async(save_video, args=(func_args,))
        writer_results.append(writer_result)
        while True:
            
            writer_results = [r for r in writer_results if not r.ready()]
            if len(writer_results) >= args.writer_num_workers + 2:
                time.sleep(0.5)
            else:
                break
                
    writer_pool.close()
    writer_pool.join()
    print(f"Finish cache all data")
            
            
             