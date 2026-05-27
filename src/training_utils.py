import math

import torch
import torch.nn.functional as F


def compute_psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0.0:
        return float("inf")
    psnr = 20 * math.log10(data_range) - 10 * torch.log10(mse)
    return psnr.item()

def save_ckpt(net_geoquery, optimizer, outf):
    """Save a GeoQuery checkpoint."""
    sd = {}
    sd["vae_lora_target_modules"] = net_geoquery.target_modules_vae
    sd["rank_vae"] = net_geoquery.lora_rank_vae
    sd["state_dict_unet"] = net_geoquery.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_geoquery.vae.state_dict().items() if "lora" in k or "skip" in k}

    sd["state_dict_geo_modules"] = net_geoquery.geo_modules.state_dict()

    sd["optimizer"] = optimizer.state_dict()
    torch.save(sd, outf)
    print(f"Saved checkpoint to {outf}")


def load_ckpt_from_state_dict(net_geoquery, optimizer, pretrained_path):
    """Load a GeoQuery checkpoint."""
    sd = torch.load(pretrained_path, map_location="cpu")

    if "state_dict_vae" in sd:
        _sd_vae = net_geoquery.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_geoquery.vae.load_state_dict(_sd_vae)

    _sd_unet = net_geoquery.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_geoquery.unet.load_state_dict(_sd_unet)

    if "state_dict_geo_modules" in sd:
        net_geoquery.geo_modules.load_state_dict(sd["state_dict_geo_modules"])
        print("Loaded geo_modules from checkpoint")

    optimizer.load_state_dict(sd["optimizer"])

    return net_geoquery, optimizer