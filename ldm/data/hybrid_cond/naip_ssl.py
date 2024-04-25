from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py


class NAIPDataset(Dataset):
    """Dataset with naip patches and ssl embeddings"""

    def __init__(self, config=None):
        self.root = Path(config.get("root"))
        split = config.get("split")

        self.image_root = self.root / "naip_patches"

        self.p_uncond = config.get("p_uncond", 0)
        self.augment = config.get("augment", True)

        # load patches list

        patches = np.load(
            self.root / f"naip_ldm_metadata/patch_list.npz", allow_pickle=True
        )
        self.patches = patches[split]

        # load SSL features
        features = h5py.File(self.root / "naip_ldm_metadata/features.h5", "r")
        self.feat = features[f"feat_{split}"]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path = self.patches[idx]
        tile = np.array(Image.open(self.image_root / img_path))[..., :3]

        image = (tile / 127.5 - 1.0).astype(np.float32)

        # Random horizontal and vertical flips
        if self.augment:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()

        # load SSL features
        feat_patch = self.feat[idx]

        # replace patch level emb with all zeros
        if np.random.rand() < self.p_uncond:
            feat_patch = np.zeros_like(feat_patch)

        return {
            "image": image,
            "feat_patch": feat_patch,
            "human_label": "",
            "image_path": img_path,
        }
