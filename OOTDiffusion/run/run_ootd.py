# from pathlib import Path
# import sys
# from PIL import Image
# from utils_ootd import get_mask_location

# PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# sys.path.insert(0, str(PROJECT_ROOT))

# from preprocess.openpose.run_openpose import OpenPose
# from preprocess.humanparsing.run_parsing import Parsing
# from ootd.inference_ootd_hd import OOTDiffusionHD
# from ootd.inference_ootd_dc import OOTDiffusionDC


# import argparse
# parser = argparse.ArgumentParser(description='run ootd')
# parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
# parser.add_argument('--model_path', type=str, default="", required=True)
# parser.add_argument('--cloth_path', type=str, default="", required=True)
# parser.add_argument('--model_type', type=str, default="hd", required=False)
# parser.add_argument('--category', '-c', type=int, default=0, required=False)
# parser.add_argument('--scale', type=float, default=2.0, required=False)
# parser.add_argument('--step', type=int, default=20, required=False)
# parser.add_argument('--sample', type=int, default=4, required=False)
# parser.add_argument('--seed', type=int, default=-1, required=False)
# args = parser.parse_args()


# openpose_model = OpenPose(args.gpu_id)
# parsing_model = Parsing(args.gpu_id)


# category_dict = ['upperbody', 'lowerbody', 'dress']
# category_dict_utils = ['upper_body', 'lower_body', 'dresses']

# model_type = args.model_type # "hd" or "dc"
# category = args.category # 0:upperbody; 1:lowerbody; 2:dress
# cloth_path = args.cloth_path
# model_path = args.model_path

# image_scale = args.scale
# n_steps = args.step
# n_samples = args.sample
# seed = args.seed

# if model_type == "hd":
#     model = OOTDiffusionHD(args.gpu_id)
# elif model_type == "dc":
#     model = OOTDiffusionDC(args.gpu_id)
# else:
#     raise ValueError("model_type must be \'hd\' or \'dc\'!")


# if __name__ == '__main__':

#     if model_type == 'hd' and category != 0:
#         raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

#     cloth_img = Image.open(cloth_path).resize((768, 1024))
#     model_img = Image.open(model_path).resize((768, 1024))
#     keypoints = openpose_model(model_img.resize((384, 512)))
#     model_parse, _ = parsing_model(model_img.resize((384, 512)))

#     mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
#     mask = mask.resize((768, 1024), Image.NEAREST)
#     mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
#     masked_vton_img = Image.composite(mask_gray, model_img, mask)
#     masked_vton_img.save('./images_output/mask.jpg')

#     images = model(
#         model_type=model_type,
#         category=category_dict[category],
#         image_garm=cloth_img,
#         image_vton=masked_vton_img,
#         mask=mask,
#         image_ori=model_img,
#         num_samples=n_samples,
#         num_steps=n_steps,
#         image_scale=image_scale,
#         seed=seed,
#     )

#     image_idx = 0
#     for image in images:
#         image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
#         image_idx += 1
from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import torch

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()

# CUDA نہ ہو تو CPU fallback (-1) — inference files یہ handle کرتی ہیں
if not torch.cuda.is_available():
    print("[WARNING] CUDA not available — falling back to CPU. This will be slow.", flush=True)
    args.gpu_id = -1
else:
    print(f"[INFO] CUDA available — GPU: {torch.cuda.get_device_name(args.gpu_id)}", flush=True)

openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type  # "hd" or "dc"
category = args.category      # 0: upperbody; 1: lowerbody; 2: dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be 'hd' or 'dc'!")


if __name__ == '__main__':

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")

    # 16GB RAM, Intel Iris Xe ke liye best balance:
    # 576x768 = same 3:4 ratio as 768x1024, ~5-6GB RAM use karta hai (fits in 8.5GB free)
    # 768x1024 = 38GB chahiye (crash), 384x512 = bad quality
    IMG_W, IMG_H = 576, 768
    cloth_img = Image.open(cloth_path).resize((IMG_W, IMG_H))
    model_img = Image.open(model_path).resize((IMG_W, IMG_H))
    # Openpose hamesha exactly (384, 512) expect karta hai - fixed rakhna hai
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((IMG_W, IMG_H), Image.NEAREST)
    mask_gray = mask_gray.resize((IMG_W, IMG_H), Image.NEAREST)

    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    Path('./images_output').mkdir(parents=True, exist_ok=True)
    masked_vton_img.save('./images_output/mask.jpg')

    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )


    image_idx = 0
    for image in images:
        image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1