import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2

import random
import time
import pdb

from pipelines_ootd.pipeline_ootd import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

# Absolute paths - CWD se independent (Intel Iris Xe CPU-only fix)
_CKPT_ROOT = Path(__file__).absolute().parents[1] / "checkpoints"

_VIT_LOCAL = _CKPT_ROOT / "clip-vit-large-patch14"
# Local folder ho to use karo, warna HuggingFace se auto-download
VIT_PATH   = str(_VIT_LOCAL) if _VIT_LOCAL.exists() else "openai/clip-vit-large-patch14"
VAE_PATH   = str(_CKPT_ROOT / "ootd")
UNET_PATH  = str(_CKPT_ROOT / "ootd" / "ootd_dc" / "checkpoint-36000")
MODEL_PATH = str(_CKPT_ROOT / "ootd")

class OOTDiffusionDC:

    def __init__(self, gpu_id):
        # gpu_id=-1 یا CUDA نہ ہو تو CPU استعمال کریں، ورنہ cuda:0 وغیرہ
        if torch.cuda.is_available() and gpu_id is not None and int(gpu_id) >= 0:
            self.gpu_id = 'cuda:' + str(gpu_id)
        else:
            self.gpu_id = 'cpu'
        print(f"[OOTDiffusionDC] Using device: {self.gpu_id}", flush=True)

        # CPU پر float16 support نہیں، CUDA پر float16 استعمال کریں
        self.dtype = torch.float16 if self.gpu_id != 'cpu' else torch.float32
        use_fp16_variant = (self.gpu_id != 'cpu')

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=self.dtype,
        )

        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm",
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton",
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        pipe_kwargs = dict(
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if use_fp16_variant:
            pipe_kwargs["variant"] = "fp16"

        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            **pipe_kwargs,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # CPU (Intel Iris Xe, 16GB RAM) ke liye memory optimization:
        # OotdPipeline custom hai - sirf attention_slicing support karta hai
        self.pipe.enable_attention_slicing(1)   # attention 1 head at a time → 8x kam RAM
        print("[OOTDiffusionDC] Memory optimization enabled: attention_slicing(1)", flush=True)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if model_type == 'hd':
                prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3).to(self.gpu_id))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images

        return images
