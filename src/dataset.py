import json
import torch
import os
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
import re
from colmap_loader import read_cameras_binary, read_images_binary
from geometry_utils import colmap_params_to_intrinsics, colmap_pose_to_c2w, resize_intrinsics

def load_pfm(file_path):

    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if 'PF' in header:
            color = True
        elif 'Pf' in header:
            color = False
        else:
            raise ValueError(f"Invalid PFM header: {header}")

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError(f"Invalid PFM header: {header}")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data).copy()

        return data, scale

def load_depth_conf(depth_path, conf_path):
    depth_map,_ = load_pfm(depth_path)
    conf_map = np.load(conf_path)

    return depth_map, conf_map



class DepthPairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, mode='resize', height=576, width=1024, tokenizer=None, conf_threshold=0.0, gt_colmap=None):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.mode = mode
        self.in_size = (540,960)
        self.crop_size = (512,512)
        self.depth_size = (576, 960)
        self.conf_threshold = conf_threshold
        self.gt_path = gt_colmap

        assert gt_colmap and Path(gt_colmap).exists(), "Error: gt_colmap path is invalid or does not exist!!!"

        print("collecting colmap info of all scenes...")
        self.colmap_info = {}
        scene_ids = os.listdir(self.gt_path)
        for scene_id in tqdm(scene_ids, desc="collecting scene info"):
            colmap_path = os.path.join(self.gt_path, scene_id, "gaussian_splat/sparse/0/")
            if not os.path.exists(colmap_path):
                print(f"no colmap or .cache file in: {colmap_path}")
                continue
            cameras = read_cameras_binary(os.path.join(colmap_path, "cameras.bin"))
            images = read_images_binary(os.path.join(colmap_path, "images.bin"))

            self.colmap_info[scene_id] = {
                "orig_size": (cameras[1].width, cameras[1].height),
                "intrinsic": colmap_params_to_intrinsics(cameras[1].params), # [3,3] Tensor
                "extrinsics": images,
            }

        print("colmap info collected successfully")

        def resize_fn(img):
            return F.resize(img, self.image_size)

        def center_crop_fn(img):
            return F.center_crop(img, self.crop_size)

        mode_fns = {
            "resize": resize_fn,
            "center_crop": center_crop_fn,
        }
        assert mode in mode_fns, f"mode must be one of {list(mode_fns.keys())}, got {mode}"
        self.transform_fn = mode_fns[mode]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id].get("ref_image")
        caption = self.data[img_id]["prompt"]
        ref_depth = self.data[img_id]["ref_depth"]
        ref_depth_confidence = self.data[img_id]["ref_depth_confidence"]

        scene_id = ref_img.split("/")[-4]
        colmap_info = self.colmap_info[scene_id]
        intrinsic = colmap_info["intrinsic"]
        extrinsics = colmap_info["extrinsics"]
        ref_img_idx = int(ref_img.split("/")[-1].split("_")[1].split(".")[0])
        target_img_idx = int(input_img.split("/")[-1].split("_")[1].split(".")[0])

        ref_qvec, ref_tvec = extrinsics[ref_img_idx].qvec, extrinsics[ref_img_idx].tvec
        target_qvec, target_tvec = extrinsics[target_img_idx].qvec, extrinsics[target_img_idx].tvec

        ref_pose = torch.from_numpy(colmap_pose_to_c2w(ref_qvec, ref_tvec)).float()
        target_pose = torch.from_numpy(colmap_pose_to_c2w(target_qvec, target_tvec)).float()

        if self.mode == "resize":
            intrinsic = resize_intrinsics(
                intrinsic,
                colmap_info["orig_size"],
                target_size=(self.image_size[1], self.image_size[0]),
            )
        elif self.mode == "center_crop":
            intrinsic = resize_intrinsics(
                intrinsic,
                colmap_info["orig_size"],
                target_size=(self.in_size[1], self.in_size[0]),
                crop_size=self.crop_size,
            )

        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except Exception:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        input_t = F.to_tensor(input_img)
        input_t = self.transform_fn(input_t)
        input_t = F.normalize(input_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = self.transform_fn(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_img)
            ref_t = self.transform_fn(ref_t)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])

            input_t = torch.stack([input_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)
        else:
            input_t = input_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        ref_depth, ref_depth_conf = load_depth_conf(ref_depth, ref_depth_confidence)
        conf_mask = ref_depth_conf / 100.0 > self.conf_threshold
        ref_depth = ref_depth * conf_mask

        _, h, w = input_t.shape[-3:]
        ref_depth = torch.from_numpy(ref_depth).float().unsqueeze(0).unsqueeze(0)
        if self.mode == "resize":
            ref_depth = torch.nn.functional.interpolate(ref_depth, size=(h, w), mode="nearest")
            ref_depth = ref_depth.squeeze(0)
        elif self.mode == "center_crop":
            ref_depth = torch.nn.functional.interpolate(ref_depth, size=(540, 960), mode="nearest")
            ref_depth = ref_depth.squeeze(0)
            ref_depth = F.center_crop(ref_depth, (h, w))

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": input_t,
            "caption": caption,
            "ref_depth": ref_depth,
            "ref_pose": ref_pose,
            "target_pose": target_pose,
            "intrinsic": intrinsic,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            out["input_ids"] = input_ids

        return out
