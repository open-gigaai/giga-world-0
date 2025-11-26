import functools

import torch
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from giga_train import ModuleDict, Trainer
from peft import LoraConfig

from giga_models import GigaWorld0Transformer3DModel, LoRAPeftWrapper
from giga_models.nn import EDMLoss


class GigaWorld0Trainer(Trainer):
    """GigaWorld0 Trainer Class Inherits from base Trainer class, specialized
    for training 3D Transformer diffusion models.

    Supports video generation tasks using VAE encoder and EDM loss function.
    """

    def get_models(self, model_config):
        """Initialize and configure model components.

        Args:
            model_config: Model configuration object containing VAE and Transformer paths and training parameters.

        Returns:
            ModuleDict: Dictionary containing all models required for training.
        """
        model = dict()

        vae_dtype = model.get('vae_dtype', self.dtype)
        vae = AutoencoderKLWan.from_pretrained(model_config.vae_model_path)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        # Shape: (1, z_dim, 1, 1, 1) for broadcasting
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)

        transformer = GigaWorld0Transformer3DModel.from_pretrained(model_config.transformer_model_path)
        # Get training mode: 'full' (full training) or 'lora' (LoRA fine-tuning)
        self.train_mode = model_config.get('train_mode', 'full')
        if self.train_mode == 'lora':
            transformer.requires_grad_(False)
            lora_rank = model_config.lora_rank
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
                target_modules=['to_q.0', 'to_k.0', 'to_v.0', 'to_out.0'],
            )
            transformer.add_adapter(lora_config)
            transformer = LoRAPeftWrapper(transformer)
        model.update(transformer=transformer)

        self.edm_loss = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)

        model = ModuleDict(model)
        model.to(self.dtype)
        model.train()
        if self.train_mode == 'lora':
            model.load_state_dict_mode = 'each'
        # Convert to FP8 if mixed precision is enabled
        if self.mixed_precision == 'fp8':
            for model_name in model.keys():
                model[model_name].to_fp8(ignore_modules=model_config.fp8_ignore_modules)
        return model

    def forward_step(self, batch_dict):
        """Perform a single forward pass during training.

        Args:
            batch_dict: Dictionary containing batch data including images, prompts, and reference images.

        Returns:
            loss: Computed loss value for the batch.
        """
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        batch_size = images.shape[0]

        padding_mask = torch.zeros((batch_size, 1, images.shape[-2], images.shape[-1]), dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        latents = self.forward_vae(images)
        input_latents, timesteps = self.edm_loss.add_noise(latents)

        ref_images = batch_dict['ref_images']
        ref_masks = batch_dict['ref_masks'].to(self.dtype)
        ref_latents = self.forward_vae(ref_images)

        augment_sigma = torch.tensor([0.0001], device=ref_latents.device, dtype=latents.dtype)
        while len(augment_sigma.shape) < len(ref_latents.shape):
            augment_sigma = augment_sigma.unsqueeze(-1)

        input_latents = ref_masks * ref_latents + (1 - ref_masks) * input_latents
        input_masks = ref_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
        input_latents = torch.cat([input_latents, input_masks], dim=1)
        timesteps = timesteps.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
        t_conditioning = augment_sigma / (augment_sigma + 1)
        timesteps = ref_masks * t_conditioning + (1 - ref_masks) * timesteps

        input_latents = input_latents.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        prompt_embeds = prompt_embeds.to(self.dtype)
        if self.train_mode == 'lora':
            input_latents.requires_grad_(True)
        model_pred = transformer(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
        )
        denoised_latents = self.edm_loss.denoise(model_pred.float())
        if 'ref_images' in batch_dict:
            denoised_latents = ref_masks * ref_latents + (1 - ref_masks) * denoised_latents
        loss = self.edm_loss.compute_loss(denoised_latents)
        return loss

    def forward_vae(self, images):
        """Encode images to latent space using VAE.

        Args:
            images: Input images tensor with shape (batch, time, channels, height, width).

        Returns:
            latents: Encoded and normalized latent representations.
        """
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.sample()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents
