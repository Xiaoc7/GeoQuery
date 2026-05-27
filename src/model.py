import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class GeoQuery(torch.nn.Module):

    def __init__(
        self,
        pretrained_path=None,
        lora_rank_vae=4,
        timestep=199,
        neighborhood_size=5,
        low_res_only=False,
    ):
        super().__init__()

        print("="*70)
        print("Initializing GeoQuery")
        print("="*70)

        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        self.neighborhood_size = neighborhood_size
        self.low_res_only = low_res_only

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False


        from geoquery_unet import UNet2DConditionModel, init_geoquery_modules, set_geometry_context, clear_geometry_context
        print("Initializing multi-view UNet with GeoQuery attention...")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")


        self.set_geometry_context = set_geometry_context
        self.clear_geometry_context = clear_geometry_context

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                        target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            if "state_dict_geo_modules" in sd:
                unet.to("cuda")
                self.geo_modules = init_geoquery_modules(unet, neighborhood_size=neighborhood_size, low_res_only=low_res_only)
                self.geo_modules.load_state_dict(sd["state_dict_geo_modules"])
                print("Loaded geo_modules from checkpoint")
            else:
                unet.to("cuda")
                self.geo_modules = init_geoquery_modules(unet, neighborhood_size=neighborhood_size, low_res_only=low_res_only)
                print("Initialized new geo_modules")

        else:
            print("Initializing model with random weights")

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]

            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

            unet.to("cuda")
            self.geo_modules = init_geoquery_modules(unet, neighborhood_size=neighborhood_size, low_res_only=low_res_only)
            print(f"Initialized {len(self.geo_modules)} GeoQuery modules")

        vae.to("cuda")
        self.geo_modules.cuda()

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)


        self._print_model_stats()

    def _print_model_stats(self):
        print("="*70)
        print("Model Statistics:")
        print(f"   - UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"   - VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"   - Geo Modules: {sum(p.numel() for p in self.geo_modules.parameters()) / 1e6:.2f}M")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   - Total Trainable: {total_params / 1e6:.2f}M")
        print("="*70)

    def prepare_correspondence(self, ref_depth, K, ref_pose, target_pose, H, W):
        from geometry_utils import build_geometric_correspondence

        B = ref_depth.shape[0]
        device = ref_depth.device

        ref_depth = ref_depth.float()
        K = K.float()
        ref_pose = ref_pose.float()
        target_pose = target_pose.float()

        y_ref, x_ref = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        ref_coord_grid = torch.stack([x_ref, y_ref], dim=0)
        ref_coord_grid = ref_coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        correspondence_result = build_geometric_correspondence(
            ref_coord_grid, ref_depth, K, ref_pose, target_pose
        )

        geometric_correspondence = correspondence_result['correspondence'].float()
        validity_mask = correspondence_result['validity_mask'].float()

        return {
            'correspondence': geometric_correspondence,
            'validity_mask': validity_mask,
        }

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.geo_modules.eval()
        self.geo_modules.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)

        self.geo_modules.train()
        self.geo_modules.requires_grad_(True)


        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None, geometry_inputs=None):

        assert (prompt is None) != (prompt_tokens is None)
        assert (timesteps is None) != (self.timesteps is None)

        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        B, num_views, C, H, W = x.shape

        if geometry_inputs is None:
            self.set_geometry_context(None, None, None, enabled=False)

            x_flat = rearrange(x, 'b v c h w -> (b v) c h w')
            z_unet = self.vae.encode(x_flat).latent_dist.sample() * self.vae.config.scaling_factor
            all_skip_feats = self.vae.encoder.current_down_blocks
            num_views_total = num_views

            caption_enc_repeated = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views_total)
            model_pred = self.unet(
                z_unet, self.timesteps,
                encoder_hidden_states=caption_enc_repeated,
                cross_attention_kwargs={"num_views": num_views_total}
            ).sample
            z_denoised = self.sched.step(model_pred, self.timesteps, z_unet, return_dict=True).prev_sample

            self.vae.decoder.incoming_skip_acts = all_skip_feats
            output_image = self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
            output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views_total)

            self.clear_geometry_context()
            return output_image

        x_flat = rearrange(x, 'b v c h w -> (b v) c h w')
        z_flat = self.vae.encode(x_flat).latent_dist.sample() * self.vae.config.scaling_factor
        encoder_feats = self.vae.encoder.current_down_blocks

        z = rearrange(z_flat, '(b v) c h w -> b v c h w', v=num_views)
        Hz, Wz = z.shape[-2:]


        feats_by_view = []
        for v in range(num_views):
            feats_v = [f[v*B:(v+1)*B] for f in encoder_feats]
            feats_by_view.append(feats_v)


        with torch.cuda.amp.autocast(enabled=False):

            corr_result = self.prepare_correspondence(
                geometry_inputs['ref_depth'],
                geometry_inputs['K'],
                geometry_inputs['ref_pose'],
                geometry_inputs['target_pose'],
                H, W
            )

            correspondence = F.interpolate(
                corr_result['correspondence'],
                size=(Hz, Wz),
                mode='bilinear',
                align_corners=False
            ).float()
            correspondence[:, 0] *= (Wz / W)
            correspondence[:, 1] *= (Hz / H)

            validity_mask = F.interpolate(
                corr_result['validity_mask'],
                size=(Hz, Wz),
                mode='nearest',
            ).float()

        self.set_geometry_context(
            correspondence=correspondence,
            validity_mask=validity_mask,
            spatial_size=(Hz, Wz),
            enabled=True,
        )

        z_unet = rearrange(z, 'b v c h w -> (b v) c h w')
        num_views_total = num_views

        all_skip_feats = []
        for level_idx in range(len(encoder_feats)):
            level_feats = [feats_by_view[v][level_idx] for v in range(num_views)]
            all_skip_feats.append(torch.cat(level_feats, dim=0))

        caption_enc_repeated = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views_total)



        model_pred = self.unet(
            z_unet,
            self.timesteps,
            encoder_hidden_states=caption_enc_repeated,
            cross_attention_kwargs={"num_views": num_views_total}
        ).sample

        z_denoised = self.sched.step(model_pred, self.timesteps, z_unet, return_dict=True).prev_sample

        self.vae.decoder.incoming_skip_acts = all_skip_feats
        output_image = self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views_total)

        self.clear_geometry_context()

        return output_image


    def sample(
        self,
        target_image: Image.Image,
        reference_image: Image.Image,
        reference_depth: np.ndarray,
        K: np.ndarray,
        reference_pose: np.ndarray,
        target_pose: np.ndarray,
        prompt: str = "remove degradation",
    ) -> Image.Image:
        """

        Args:
            target_image: Rendered target-view image.
            reference_image: Reference-view image.
            reference_depth: Reference-view depth map, shaped [H, W].
            K: Camera intrinsics, shaped [3, 3].
            reference_pose: Reference camera-to-world pose, shaped [4, 4].
            target_pose: Target camera-to-world pose, shaped [4, 4].
            prompt: Text prompt.

        Returns:
            refined_image: PIL Image.
        """
        from torchvision import transforms

        original_size = target_image.size

        input_width, input_height = target_image.size
        new_width = input_width - input_width % 8
        new_height = input_height - input_height % 8

        T = transforms.Compose([
            transforms.Resize((new_height, new_width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        target_tensor = T(target_image).cuda()
        reference_tensor = T(reference_image).cuda()

        x = torch.stack([target_tensor, reference_tensor], dim=0).unsqueeze(0)

        reference_depth_tensor = torch.from_numpy(reference_depth).float().cuda()
        reference_depth_tensor = reference_depth_tensor.unsqueeze(0).unsqueeze(0)
        valid_mask_depth = (reference_depth_tensor > 0).float()
        reference_depth_tensor = F.interpolate(
            reference_depth_tensor,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )
        valid_mask_resized = F.interpolate(
            valid_mask_depth,
            size=(new_height, new_width),
            mode='nearest',
        )
        reference_depth_tensor = reference_depth_tensor * valid_mask_resized


        K_tensor = torch.from_numpy(K).float().cuda()
        scale_x = new_width / input_width
        scale_y = new_height / input_height
        K_scaled = K_tensor.clone()
        K_scaled[0, :] *= scale_x
        K_scaled[1, :] *= scale_y
        K_scaled = K_scaled.unsqueeze(0)

        reference_pose_tensor = torch.from_numpy(reference_pose).float().cuda().unsqueeze(0)
        target_pose_tensor = torch.from_numpy(target_pose).float().cuda().unsqueeze(0)

        geometry_inputs = {
            'ref_depth': reference_depth_tensor,
            'K': K_scaled,
            'ref_pose': reference_pose_tensor,
            'target_pose': target_pose_tensor,
        }


        torch.cuda.synchronize()
        tic = time.time()
        with torch.no_grad():
            output = self.forward(x, prompt=prompt, geometry_inputs=geometry_inputs)

        torch.cuda.synchronize()
        elapsed = time.time()-tic



        output_image = output[0, 0]
        output_image = output_image.cpu() * 0.5 + 0.5
        output_image = output_image.clamp(0, 1)

        output_pil = transforms.ToPILImage()(output_image)
        output_pil = output_pil.resize(original_size, Image.LANCZOS)

        return output_pil, elapsed




    def save_model(self, outf, optimizer):
        """Save a GeoQuery checkpoint."""
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}

        sd["state_dict_geo_modules"] = self.geo_modules.state_dict()

        sd["optimizer"] = optimizer.state_dict()

        torch.save(sd, outf)
        print(f"GeoQuery model saved to {outf}")
