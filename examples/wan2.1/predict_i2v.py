import os
import sys

import numpy as np
import torch
import shutil
import torch.distributed as dist

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from tqdm import tqdm
from videox_fun.data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanI2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# GPU memory mode, which can be choosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# GPU_memory_mode     = "sequential_cpu_offload"
GPU_memory_mode     = "no_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.

fsdp_dit            = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit.
compile_dit         = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None
import argparse
if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/weili/miniconda3/envs/wan21_xc/bin/python 
    #   /home/weili/miniconda3/envs/wan21_xc/bin/torchrun --nproc-per-node 4 --master-port 25083
    #   examples/wan2.1/predict_i2v.py 
    #   --validation_config config/wan2.1/sky_i2v_1.3B.yaml 
    #   --model_name default models/SkyReels-V2-I2V-1.3B-540P 
    #   --save_dir_path default basename of validation_config
    #   --save_dir_path validations_outputs/xxx
    #   --save_parent default validations_outputs
    #   --ulysses_degree 1
    #   --ring_degree 1
    #   --video_sample_size default 720 
    #   --video_length default 81 
    #   --enable_teacache 
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="models/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--save_parent", type=str, default="validations_outputs")
    parser.add_argument("--save_dir_path", type=str, default=None)
    parser.add_argument("--video_sample_size", type=int, default=720,)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    # Other params
    valid_config = OmegaConf.load(args.validation_config)
    if args.save_dir_path is None:
        args.save_dir_path = os.path.join(args.save_parent, os.path.basename(args.validation_config)[:-5])
    video_length        = args.video_length
    fps                 = args.fps
    model_name          = args.model_name
    ulysses_degree      = args.ulysses_degree
    ring_degree         = args.ring_degree
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype            = torch.bfloat16
    
    # Support TeaCache.
    enable_teacache     = args.enable_teacache
    # Recommended to be set between 0.05 and 0.20. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    teacache_threshold  = 0.10
    # The number of steps to skip TeaCache at the beginning of the inference process, which can
    # reduce the impact of TeaCache on generated video quality.
    num_skip_start_steps = 5
    # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
    teacache_offload    = False

    # prompts
    # prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
    # negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    # prompt              = "A person is speaking."
    # negative_prompt     = "oversaturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, still frame, overall grayish, worst quality, low quality, JPEG artifacts, ugly, broken, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, people walking backward"

    guidance_scale      = 6.0
    seed                = 43
    num_inference_steps = 50
    lora_weight         = 0.55

    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    is_main_process = False
    if ulysses_degree * ring_degree > 1:
        if dist.get_rank() == 0:
            is_main_process = True
    else:

        is_main_process = True
    if not os.path.exists(args.save_dir_path) and is_main_process:
        os.makedirs(args.save_dir_path, exist_ok=True)
    config = OmegaConf.load(config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=False if not fsdp_dit else False,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanI2VPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

    for validation_image_start, validation_prompt, validation_neg_prompt in tqdm(valid_config.data.test_cases):
        basename = os.path.basename(validation_image_start)[:-4]
        save_path = os.path.join(args.save_dir_path, basename)
        if os.path.exists(save_path): continue
        with torch.no_grad():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

            if enable_riflex:
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
            image_check = Image.open(validation_image_start).convert("RGB")
            img_check_width, img_check_height = image_check.size
            aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            closest_size, closest_ratio = get_closest_ratio(img_check_height, img_check_width, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]
            closest_size = list(map(lambda x: int(x), closest_size))
            if closest_size[0] / img_check_height > closest_size[1] / img_check_width:
                resize_size = closest_size[0], int(img_check_width * closest_size[0] / img_check_height)
            else:
                resize_size = int(img_check_height * closest_size[1] / img_check_width), closest_size[1]
            valid_video_height, valid_video_width = closest_size
            input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, None, video_length=video_length, sample_size=resize_size, center_crop_size=closest_size)
            sample = pipeline(
                validation_prompt, 
                num_frames = video_length,
                negative_prompt = validation_neg_prompt,
                height      = valid_video_height,
                width       = valid_video_width,
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video      = input_video,
                mask_video   = input_video_mask,
                clip_image = clip_image,
                cfg_skip_ratio = cfg_skip_ratio,
                shift = shift,
            ).videos

        def save_results():
            shutil.copy(validation_image_start, args.save_dir_path)
            save_path = os.path.join(args.save_dir_path, basename)
            if video_length == 1:
                save_path = save_path + ".png"
                image = sample[0, :, 0]
                image = image.transpose(0, 1).transpose(1, 2)
                image = (image * 255).numpy().astype(np.uint8)
                image = Image.fromarray(image)
                image.save(save_path)
            else:
                save_path = save_path + ".mp4"
                save_videos_grid(sample, save_path, fps=fps)
        if is_main_process:
            save_results()
        if ulysses_degree > 1 or ring_degree > 1:
            dist.barrier()
        
    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)