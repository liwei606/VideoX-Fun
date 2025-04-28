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
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_gt = save_gt
        assert enable_bucket, "use LiveVideoLoadDataset, must enable_bucket"

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
        video_path = f'{folder}/videos_resampled/{name}+resampled.mp4'
        if not os.path.exists(video_path):
            video_path = f'{folder}/videos/{name}.mp4'
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "data_type": data_type,
            "idx": index,
            "clip_st": meta_result["clip_st"],
            "clip_et": meta_result["clip_et"],
        }
        sample["video_path"] = video_path
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
from tqdm import tqdm
from einops import rearrange
from accelerate.utils import set_seed

def save_video(func_args):
    save_path, visual_list, fps = func_args
    imageio.mimwrite(save_path, visual_list, fps=fps)

if __name__ == "__main__":
    # RUN_FUNC = os.environ["RUN_FUNC"]
    # RUN_FUNC = 'test_fast_video_dataset'
    # print(f"{RUN_FUNC=}")
    # eval(RUN_FUNC)()
    # /home/weili/miniconda3/envs/wan21_xc/bin/python 
    #    videox_fun/data/emo_video_live_body_load.py 
    #    --dataset_file_path 
    #    --data_type 
    #    --save_fps 
    #    --seed 
    #    --save_origin_video 
    #    --num_workers 32
    #    --writer_num_workers 32

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="video")
    
    parser.add_argument("--save_fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_origin_video", action="store_true")
    
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--writer_num_workers", type=int, default=32)

    args = parser.parse_args()
    
    config = {"data": {}}
    set_seed(args.seed)
    # visual norm data
    dataset_file_path = args.dataset_file_path
    visual_dir_name = "liveLoad_" + os.path.basename(dataset_file_path)
    config["data"]["cache_file_path"] = [[dataset_file_path, 1, args.data_type]]
    
    config = OmegaConf.create(config)
    train_dataset = LiveVideoLoadDataset(
        # width=config.data.train_width,
        # height=config.data.train_height,
        cfg=config,
        split='train',
        enable_bucket=True,
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
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=8,
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
        pixel_values = batch["pixel_values"][0]
        clip_st = batch["clip_st"][0].cpu().item()
        clip_et = batch["clip_et"][0].cpu().item()

        save_dir = visual_dir_name
        os.makedirs(save_dir, exist_ok=True)
        
        visual_list = []
        for pixel_values_i in pixel_values:
            visual_list.append(pixel_values_i.cpu().numpy().astype("uint8"))
        
        video_base = os.path.basename(video_path)[:-4]
        save_path = f"./{save_dir}/{i}_{video_base}.mp4"
        
        if args.save_origin_video:
            shutil.copy(video_path, save_dir)
        func_args = (save_path, visual_list, args.save_fps)
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
            
            
             