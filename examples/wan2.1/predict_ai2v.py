import os
import sys
import librosa
import audiofile
import ffmpeg
import numpy as np
import torch
import shutil
import copy
import torch.distributed as dist

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from transformers import Wav2Vec2Model, Wav2Vec2Processor

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
from videox_fun.pipeline import WanAI2VPipeline, WanI2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ultimatevocalremover_api.src import models as demucs_models
from ultimatevocalremover_api.src.utils.fastio import read as demucs_read_audio

def get_wav_vocal(
    demucs_model,
    wav_path,
    demucsv4,
    is_main_process,
):
    process_vocals_path = wav_path[:-4] + ("_demucs.wav" if not demucsv4 else "_demucsv4.wav")
    if not os.path.exists(process_vocals_path) and is_main_process:
        wav_path = str(wav_path)
        # audio, sampling_rate = demucs_read_audio(wav_path)
        # audio_duration = len(audio) / sampling_rate
        # if audio_duration <= 8:
        res = demucs_model(wav_path)
        vocals = res["vocals"].cpu()
        # else:
        #     clip_len = 8 * sampling_rate
        #     stride = 6 * sampling_rate
        #     vocals = np.zeros_like(audio)
        #     weights = np.zeros_like(audio)
        #     for audio_clip_s in tqdm(range(0, len(audio), stride), desc="Preprocess Demucs for audio"):
        #         audio_item = audio[audio_clip_s: audio_clip_s + clip_len]
        #         demucs_model = copy.deepcopy(demucs_model_ori)
        #         res = demucs_model(audio_item, sampling_rate)
        #         vocal = res["vocals"].cpu()
        #         end_idx = audio_clip_s + clip_len
        #         vocals[audio_clip_s:end_idx] += vocal
        #         weights[audio_clip_s:end_idx] += 1.0
        #     weights = np.clip(weights, min=1e-8)
        #     vocals /= weights
        audiofile.write(process_vocals_path, vocals, demucs_model.sample_rate)
    return process_vocals_path

def get_audio_features(wav2vec, audio_processor, audio_path, fps, num_frames):
    sr = 16000
    audio_input, sample_rate = librosa.load(audio_path, sr=sr)  # 采样率为 16kHz

    start_time = 0
    # end_time = (0 + (num_frames - 1) * 1) / fps
    end_time = num_frames / fps

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    try:
        audio_segment = audio_input[start_sample:end_sample]
    except:
        audio_segment = audio_input

    input_values = audio_processor(
        audio_segment, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values.to("cuda")

    with torch.no_grad():
        fea = wav2vec(input_values).last_hidden_state

    return fea, start_time, end_time

def save_video_with_audio(save_path, save_path_tmp, audio_path, clip_st, clip_et):
    # import pdb; pdb.set_trace()
    audio_clip_path = save_path[:-4] + ".wav"
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
vae_path            = None
lora_path           = None
import argparse
if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
    #   /home/weili/miniconda3/envs/wan21_xc/bin/torchrun --nproc-per-node 4 --master-port 25083
    #   examples/wan2.1/predict_ai2v.py 
    #   --validation_config default config/wan2.1/ai2v_1.3B_base_infer.yaml 
    #   --infer_mode default ai2v
    #   --model_name default models/Wan2.1-Fun-1.3B-InP  
    #   --wav2vec_model_dir default models/wav2vec2-base-960h
    #   --transformer_path default None
    
    #   --save_dir_path default basename of validation_config
    #   --save_dir_path validations_outputs/xxx
    #   --save_parent default validations_outputs
    
    #   --video_sample_size default 720 
    #   --video_length default 81 
    #   --demucsv4 
    #   --process_audio_only 
    
    #   --ulysses_degree 2
    #   --ring_degree 2
    #   --enable_teacache 
    
    #   --num_inference_steps
    #   --sample_shift
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_config", type=str, default="config/wan2.1/ai2v_1.3B_base_infer.yaml")
    # model path, model architecture and infer mode
    parser.add_argument("--infer_mode", type=str, default="ai2v",) # select ai2v or i2v
    parser.add_argument("--model_name", type=str, default="models/Wan2.1-Fun-1.3B-InP")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--wav2vec_model_dir",type=str,default="models/wav2vec2-base-960h",)
    parser.add_argument("--audio_in_dim", type=int, default=768, help="Audio Feature inner dim.")
    parser.add_argument("--audio_proj_dim", type=int, default=768, help="Audio Feature projection dim.")
    # save path related
    parser.add_argument("--save_parent", type=str, default="validations_outputs")
    parser.add_argument("--save_dir_path", type=str, default=None)
    # data related
    parser.add_argument("--video_sample_size", type=int, default=720,)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--demucsv4", action="store_true")
    parser.add_argument("--process_audio_only", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    # Diffusion related
    parser.add_argument("--num_inference_steps", type=int, default=50,)
    parser.add_argument("--sample_shift", type=float, default=0,)
    # Parallel related
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--enable_teacache", action="store_true")
    
    args = parser.parse_args()
    # Other params
    valid_config = OmegaConf.load(args.validation_config)
    if args.save_dir_path is None:
        args.save_dir_path = os.path.join(args.save_parent, os.path.basename(args.validation_config)[:-5])
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
    num_inference_steps = args.num_inference_steps
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

    # demucs model
    demucs_model = demucs_models.Demucs(
        name="hdemucs_mmi" if not args.demucsv4 else "htdemucs_ft",
        other_metadata={"segment": 2, "split": True},
        device="cuda:0",
        logger=None,
    )

    config["transformer_additional_kwargs"]["ai2v_enable"] = args.infer_mode == "ai2v"
    config["transformer_additional_kwargs"]["audio_in_dim"] = args.audio_in_dim
    config["transformer_additional_kwargs"]["audio_proj_dim"] = args.audio_proj_dim
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=False if not fsdp_dit else False,
        torch_dtype=weight_dtype,
    )

    transformer_path = args.transformer_path
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
    
    # Get Audio Processor
    if args.infer_mode == "ai2v":
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
        wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir).cuda().eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = args.sample_shift
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    PipelineClass = WanAI2VPipeline if args.infer_mode == "ai2v" else WanI2VPipeline
    pipeline = PipelineClass(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
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

    for (validation_image_start, 
        validation_prompt, 
        validation_neg_prompt,
        validation_audio_path) in tqdm(valid_config.data.test_cases):
        basename = os.path.basename(validation_image_start)[:-4]
        save_path = os.path.join(args.save_dir_path, basename)
        if os.path.exists(save_path): continue
        with torch.no_grad():
            if args.infer_mode == "ai2v":
                audio_vocal_path = get_wav_vocal(demucs_model, validation_audio_path, args.demucsv4, is_main_process)
                if ulysses_degree > 1 or ring_degree > 1:
                    dist.barrier()
                audio_wav2vec_fea, start_time, end_time = get_audio_features(
                    wav2vec, wav2vec_processor, audio_vocal_path, args.fps, args.video_length
                )
            if args.process_audio_only: continue
            video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1
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
                audio_wav2vec_fea = audio_wav2vec_fea if args.infer_mode == "ai2v" else None,
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
                save_path_tmp = save_path + ("_tmp.mp4" if args.infer_mode == "ai2v" else ".mp4")
                save_path = save_path + ".mp4"
                save_videos_grid(sample, save_path_tmp, fps=fps)
                if args.infer_mode == "ai2v":
                    save_video_with_audio(save_path, save_path_tmp, validation_audio_path, start_time, end_time)
                
        if is_main_process:
            save_results()
        if ulysses_degree > 1 or ring_degree > 1:
            dist.barrier()
        
    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)