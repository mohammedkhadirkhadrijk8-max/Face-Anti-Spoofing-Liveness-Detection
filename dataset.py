"""
dataset.py  –  Face Anti-Spoofing Dataset Loader
=================================================
• Loads images from a local dataset folder arranged as:
      root/
        real/   (or live/, bonafide/)
        spoof/  (or fake/, attack/, print/, replay/, mask/)
• Resizes every frame to 128×128.
• Extracts three feature groups per image:
      1. RGB tensor           (3, 128, 128)
      2. YCrCb channels       (3, 128, 128)
      3. LBP texture map      (1, 128, 128)
• If the local folder is missing / empty, a synthetic fallback generator
  creates random images so the full pipeline can still be smoke-tested.
"""

import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# ---------------------------------------------------------------------------
#  LBP (Local Binary Pattern) – pure-NumPy implementation
# ---------------------------------------------------------------------------

def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Compute a basic 8-neighbour LBP map for a grayscale image (H, W)."""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.float32)
    # 8 offsets (dy, dx) for the ring of radius 1
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0,  1),  (1,  1), (1,  0),
               (1, -1),  (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        # shifted view vs centre
        y_lo, y_hi = max(0, dy), min(h, h + dy)
        x_lo, x_hi = max(0, dx), min(w, w + dx)
        cy_lo, cy_hi = max(0, -dy), min(h, h - dy)
        cx_lo, cx_hi = max(0, -dx), min(w, w - dx)
        lbp[cy_lo:cy_hi, cx_lo:cx_hi] += \
            ((gray[y_lo:y_hi, x_lo:x_hi] >= gray[cy_lo:cy_hi, cx_lo:cx_hi])
             .astype(np.float32)) * (1 << bit)
    # normalise to [0, 1]
    lbp = lbp / 255.0
    return lbp


def rgb_to_ycrcb(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image (H, W, 3) to YCrCb float32 in [0, 1]."""
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cr =  128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    Cb =  128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    ycrcb = np.stack([Y, Cr, Cb], axis=-1) / 255.0
    return ycrcb.astype(np.float32)


# ---------------------------------------------------------------------------
#  Synthetic / Fallback image generator
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(root: str, n_real: int = 400, n_spoof: int = 400,
                                 img_size: int = 128) -> None:
    """Create a tiny synthetic dataset so the pipeline can be tested immediately."""
    os.makedirs(os.path.join(root, "real"),  exist_ok=True)
    os.makedirs(os.path.join(root, "spoof"), exist_ok=True)
    rng = random.Random(42)

    def _random_face(is_real: bool) -> Image.Image:
        """Return a crude synthetic 'face' image."""
        # Base colour differs for real vs spoof to give the classifier a signal
        if is_real:
            base = (rng.randint(180, 230), rng.randint(140, 180), rng.randint(110, 150))
        else:
            base = (rng.randint(100, 160), rng.randint(100, 160), rng.randint(100, 160))

        img = Image.new("RGB", (img_size, img_size), base)
        draw = ImageDraw.Draw(img)
        # Oval 'face'
        cx, cy = img_size // 2, img_size // 2
        r = rng.randint(30, 45)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(base[0] + 20, base[1] + 10, base[2] + 5))
        # Eyes
        for ex in [cx - 12, cx + 12]:
            draw.ellipse([ex - 3, cy - 8, ex + 3, cy - 2], fill=(40, 40, 40))
        # Mouth
        draw.arc([cx - 10, cy + 5, cx + 10, cy + 18], start=0, end=180,
                 fill=(180, 60, 60), width=2)

        if not is_real:
            # Add artefacts: blur + grid lines to simulate print attack
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.5)))
            draw = ImageDraw.Draw(img)
            for y in range(0, img_size, rng.randint(8, 16)):
                draw.line([(0, y), (img_size, y)],
                          fill=(base[0] - 30, base[1] - 30, base[2] - 30), width=1)
        return img

    for i in range(n_real):
        _random_face(True).save(os.path.join(root, "real", f"real_{i:04d}.png"))
    for i in range(n_spoof):
        _random_face(False).save(os.path.join(root, "spoof", f"spoof_{i:04d}.png"))
    print(f"[dataset] Generated synthetic dataset → {root}  "
          f"({n_real} real, {n_spoof} spoof)")


# ---------------------------------------------------------------------------
#  PyTorch Dataset
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

REAL_FOLDER_NAMES  = {"real", "live", "bonafide", "bona_fide", "genuine", "Real", "Live"}
SPOOF_FOLDER_NAMES = {"spoof", "fake", "attack", "print", "replay", "mask",
                      "Spoof", "Fake", "Attack", "Print", "Replay", "Mask",
                      "presentation_attack"}


class FaceAntiSpoofDataset(Dataset):
    """
    Returns a dict per sample:
        rgb     : (3, 128, 128) float32 tensor  – normalised RGB
        ycrcb   : (3, 128, 128) float32 tensor  – YCrCb channels
        lbp     : (1, 128, 128) float32 tensor  – LBP texture map
        label   : int  (0 = spoof, 1 = real)
    """

    def __init__(self, image_paths: list, labels: list, img_size: int = 128,
                 augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        # Augmentation transforms (applied to PIL image before conversion)
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ]) if augment else None

        # Normalisation matching ImageNet stats (used for the RGB branch)
        self.rgb_normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Load & resize
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Optional augmentation
        if self.aug_transform is not None:
            img_pil = self.aug_transform(img_pil)

        img_np = np.array(img_pil, dtype=np.uint8)  # (H, W, 3) RGB

        # --- RGB tensor ---
        rgb_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        rgb_tensor = self.rgb_normalize(rgb_tensor)

        # --- YCrCb tensor ---
        ycrcb_np = rgb_to_ycrcb(img_np)               # (H, W, 3) float32 [0,1]
        ycrcb_tensor = torch.from_numpy(ycrcb_np).permute(2, 0, 1)

        # --- LBP texture map ---
        gray = np.mean(img_np.astype(np.float32), axis=2)
        lbp_np = compute_lbp(gray.astype(np.uint8))    # (H, W)
        lbp_tensor = torch.from_numpy(lbp_np).unsqueeze(0)  # (1, H, W)

        return {
            "rgb":   rgb_tensor,
            "ycrcb": ycrcb_tensor,
            "lbp":   lbp_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
#  Dataset builder – discovers or generates data, returns DataLoaders
# ---------------------------------------------------------------------------

def _discover_images(root: str):
    """Walk the root and return (paths, labels)."""
    paths, labels = [], []
    for subdir in sorted(os.listdir(root)):
        full = os.path.join(root, subdir)
        if not os.path.isdir(full):
            continue
        if subdir in REAL_FOLDER_NAMES:
            lbl = 1
        elif subdir in SPOOF_FOLDER_NAMES:
            lbl = 0
        else:
            # Try to detect from deeper structure
            continue
        for fpath in sorted(glob.glob(os.path.join(full, "**", "*"), recursive=True)):
            if os.path.splitext(fpath)[1].lower() in _IMG_EXTS:
                paths.append(fpath)
                labels.append(lbl)
    return paths, labels


def build_dataloaders(
    dataset_root: str = r"C:\Users\20241CAI0023\Downloads\archive_extracted",
    fallback_root: str = None,
    img_size: int = 128,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Build train and test DataLoaders.

    If *dataset_root* does not exist or contains no images, a synthetic
    dataset is generated under *fallback_root* (defaults to ./synthetic_data).
    """
    if fallback_root is None:
        fallback_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "synthetic_data")

    paths, labels = [], []
    if os.path.isdir(dataset_root):
        paths, labels = _discover_images(dataset_root)

    if len(paths) == 0:
        print(f"[dataset] Local dataset not found at '{dataset_root}'. "
              "Generating synthetic fallback …")
        _generate_synthetic_dataset(fallback_root, n_real=400, n_spoof=400,
                                     img_size=img_size)
        paths, labels = _discover_images(fallback_root)

    assert len(paths) > 0, "No images found even after fallback generation!"
    print(f"[dataset] Total images discovered: {len(paths)}  "
          f"(Real={labels.count(1)}, Spoof={labels.count(0)})")

    # Deterministic shuffle & split
    combined = list(zip(paths, labels))
    random.Random(seed).shuffle(combined)
    paths, labels = zip(*combined)
    paths, labels = list(paths), list(labels)

    split_idx = int(len(paths) * train_ratio)
    train_paths, test_paths   = paths[:split_idx],  paths[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    train_ds = FaceAntiSpoofDataset(train_paths, train_labels, img_size, augment=True)
    test_ds  = FaceAntiSpoofDataset(test_paths,  test_labels,  img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[dataset] Train: {len(train_ds)}  |  Test: {len(test_ds)}")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
#  Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, test_loader = build_dataloaders(batch_size=16)
    batch = next(iter(train_loader))
    print("Batch keys :", list(batch.keys()))
    print("RGB shape  :", batch["rgb"].shape)
    print("YCrCb shape:", batch["ycrcb"].shape)
    print("LBP shape  :", batch["lbp"].shape)
    print("Labels     :", batch["label"].tolist())
