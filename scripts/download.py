import os

import tyro

from giga_models.utils import download_from_huggingface


def download(model_name: str, save_dir: str = './checkpoints/'):
    """Download the T5 text encoder and VAE models from HuggingFace and save
    them to the specified directory.

    Args:
        model_name: GigaWorld-0 model name
        save_dir: Directory to save the downloaded models.
    """
    if model_name == 'video_pretrain':
        transformer_model_path = download_from_huggingface(
            'open-gigaai/GigaWorld-0-Video-Pretrain-2b',
            local_dir=save_dir,
        )
    elif model_name == 'video_gr1':
        transformer_model_path = download_from_huggingface(
            'open-gigaai/GigaWorld-0-Video-GR1-2b',
            local_dir=save_dir,
        )
    else:
        assert False
    print(f'download transformer model to {transformer_model_path}')
    # Download the T5 text encoder model
    text_encoder_model_path = download_from_huggingface(
        'google-t5/t5-11b',
        local_dir=os.path.join(save_dir, 'text_encoder'),
    )
    print(f'download text_encoder model to {text_encoder_model_path}')
    # Download the VAE model
    vae_model_path = download_from_huggingface(
        'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        local_dir=save_dir,
        folders='vae',
    )
    print(f'download vae model to {vae_model_path}')


if __name__ == '__main__':
    tyro.cli(download)
