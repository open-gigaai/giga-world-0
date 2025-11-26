import os
from glob import glob
from typing import List

import tyro
from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from tqdm import tqdm

from giga_models.models.diffusion.giga_world_0 import T5TextEncoder
from giga_models.utils import download_from_huggingface


def pack_data(
    video_dir: str,
    save_dir: str,
    text_encoder_model_path: str | None = None,
    device: str = 'cuda',
):
    """Pack videos, prompts, and prompt embeddings into a dataset for training
    or evaluation.

    Args:
        video_dir: Directory containing .mp4 videos and corresponding .txt prompt files.
        save_dir: Directory to save the packed dataset.
        text_encoder_model_path: Path to T5 text encoder (download if None).
        device: Device for text encoder.
    """
    if text_encoder_model_path is None:
        text_encoder_model_path = download_from_huggingface('google-t5/t5-11b')
    # Load the T5 text encoder
    text_encoder = T5TextEncoder(text_encoder_model_path)
    text_encoder.to(device)
    # Find all video files
    video_paths: List[str] = glob(os.path.join(video_dir, '*.mp4'))
    # Writers for labels, videos, and prompt embeddings
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    video_writer = FileWriter(os.path.join(save_dir, 'videos'))
    prompt_writer = FileWriter(os.path.join(save_dir, 'prompts'))
    for idx in tqdm(range(len(video_paths))):
        # For each video, read the corresponding prompt
        anno_file = video_paths[idx].replace('.mp4', '.txt')
        prompt = open(anno_file, 'r').read().strip()
        # Encode the prompt to get embeddings
        prompt_embeds = text_encoder.encode_prompts(prompt)[0].cpu()
        label_dict = dict(data_index=idx, prompt=prompt)
        label_writer.write_dict(label_dict)
        video_writer.write_video(idx, video_paths[idx])
        prompt_writer.write_dict(idx, dict(prompt_embeds=prompt_embeds))
    # Finalize and close writers
    label_writer.write_config()
    video_writer.write_config()
    prompt_writer.write_config()
    label_writer.close()
    video_writer.close()
    prompt_writer.close()
    # Load datasets and combine into a single Dataset object
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
    prompt_dataset = load_dataset(os.path.join(save_dir, 'prompts'))
    dataset = Dataset([label_dataset, video_dataset, prompt_dataset])
    dataset.save(save_dir)


if __name__ == '__main__':
    tyro.cli(pack_data)
