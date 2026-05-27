import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import GeoQuery
from dataset import DepthPairedDataset
from loss import gram_loss
from training_utils import compute_psnr, save_ckpt, load_ckpt_from_state_dict




def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_geoquery = GeoQuery(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        neighborhood_size=args.neighborhood_size,
        low_res_only=args.low_res_only,
    )
    net_geoquery.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_geoquery.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        net_geoquery.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features.eval()
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    layers_to_opt = []
    layers_to_opt += list(net_geoquery.unet.parameters())
    layers_to_opt += list(net_geoquery.geo_modules.parameters())

    for n, _p in net_geoquery.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    layers_to_opt = layers_to_opt + \
        list(net_geoquery.vae.decoder.skip_conv_1.parameters()) + \
        list(net_geoquery.vae.decoder.skip_conv_2.parameters()) + \
        list(net_geoquery.vae.decoder.skip_conv_3.parameters()) + \
        list(net_geoquery.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    print(f"Input mode: {args.input_mode}")
    print(f"Neighborhood size: {args.neighborhood_size}")
    print(f"GCA modules count: {len(net_geoquery.geo_modules)}")

    dataset_train = DepthPairedDataset(
        dataset_path=args.dataset_path,
        split="train",
        mode=args.input_mode,
        tokenizer=net_geoquery.tokenizer,
        conf_threshold=args.conf_threshold,
        gt_colmap=args.gt_colmap
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )

    dataset_val = DepthPairedDataset(
        dataset_path=args.dataset_path,
        split="test",
        mode=args.input_mode,
        tokenizer=net_geoquery.tokenizer,
        conf_threshold=args.conf_threshold,
        gt_colmap=args.gt_colmap
    )
    random.Random(42).shuffle(dataset_val.img_ids)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_geoquery, optimizer = load_ckpt_from_state_dict(net_geoquery, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_geoquery, optimizer = load_ckpt_from_state_dict(net_geoquery, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_geoquery.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)

    net_geoquery, optimizer, dl_train, dl_val, lr_scheduler = accelerator.prepare(
        net_geoquery, optimizer, dl_train, dl_val, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)

    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_geoquery]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                ref_pose, target_pose = batch["ref_pose"], batch["target_pose"]
                K, ref_depth = batch["intrinsic"], batch["ref_depth"]

                geometry_inputs = {
                    'ref_depth': ref_depth,
                    'K': K,
                    'ref_pose': ref_pose,
                    'target_pose': target_pose,
                }

                B, V, C, H, W = x_tgt.shape

                x_tgt_pred = net_geoquery(
                    x_src,
                    prompt_tokens=batch["input_ids"],
                    geometry_inputs=geometry_inputs,
                )

                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_total"] = loss.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_geoquery), optimizer, outf)

                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_psnr = [], [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"]
                            x_tgt = batch_val["output_pixel_values"]

                            ref_pose_val = batch_val["ref_pose"]
                            target_pose_val = batch_val["target_pose"]
                            K_val = batch_val["intrinsic"]
                            ref_depth_val = batch_val["ref_depth"]

                            geometry_inputs_val = {
                                'ref_depth': ref_depth_val,
                                'K': K_val,
                                'ref_pose': ref_pose_val,
                                'target_pose': target_pose_val,
                            }

                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred = accelerator.unwrap_model(net_geoquery)(
                                    x_src,
                                    prompt_tokens=batch_val["input_ids"].cuda(),
                                    geometry_inputs=geometry_inputs_val
                                )

                                if step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))

                                x_tgt = x_tgt[:, 0]
                                x_tgt_pred = x_tgt_pred[:, 0]

                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                psnr = compute_psnr(x_tgt_pred.float(), x_tgt.float())

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(psnr)

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/psnr"] = np.mean(l_psnr)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--neighborhood_size", default=5, type=int,
                       help="Neighborhood size for geo attention (e.g., 3, 5, 7)")
    parser.add_argument("--low_res_only", action="store_true", default=False)

    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--input_mode", default="resize", type=str, choices=["resize", "center_crop"])

    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=100)
    parser.add_argument("--tracker_project_name", type=str, default="geoquery")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    parser.add_argument("--resume", default=None, type=str)

    parser.add_argument("--conf_threshold", default=0.0, type=float)
    parser.add_argument("--gt_colmap", type=str, default=None)

    args = parser.parse_args()

    main(args)
