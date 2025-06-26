from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import os

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    Skips unreadable or missing files. Stops when `num` valid images are collected.
    Shows progress with tqdm.
    """
    samples = []
    idx = 0
    collected = 0
    max_attempts = num * 2  # Avoid infinite loop
    pbar = tqdm(total=num, desc="Collecting valid images")

    while collected < num and idx < max_attempts:
        image_path = os.path.join(sample_dir, f"{idx:06d}.png")
        try:
            with Image.open(image_path) as sample_pil:
                sample_np = np.asarray(sample_pil).astype(np.uint8)
                samples.append(sample_np)
                collected += 1
                pbar.update(1)
        except (FileNotFoundError, UnidentifiedImageError):
            tqdm.write(f"Skipped: {image_path}")
        idx += 1

    pbar.close()

    if collected < num:
        raise RuntimeError(f"Only collected {collected} valid samples after checking {idx} files.")

    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

create_npz_from_sample_folder("/data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/samples/ResCaFalse-DDIMFalse-DiT-XL-2-size-256-vae-ema-cfg-1.5-seed-0-step-50-num-50000-attention-0.07-ToCa-ddim50-global-3-softweight-0.25", 30000)
create_npz_from_sample_folder("/data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/samples/ResCaFalse-DDIMTrue-DiT-XL-2-size-256-vae-ema-cfg-1.5-seed-0-step-50-num-50000-attention-0.07-ToCa-ddim50-global-3-softweight-0.25", 30000)
create_npz_from_sample_folder("/data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/samples/ResCaTrue-DDIMFalse-DiT-XL-2-size-256-vae-ema-cfg-1.5-seed-0-step-50-num-50000-attention-0.07-ToCa-ddim50-global-3-softweight-0.25", 30000)
create_npz_from_sample_folder("/data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/samples/ResCaTrue-DDIMTrue-DiT-XL-2-size-256-vae-ema-cfg-1.5-seed-0-step-50-num-50000-attention-0.07-ToCa-ddim50-global-3-softweight-0.25", 30000)




