from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py

# Condition with HIPT features

class TCGADataset(Dataset):

    def __init__(self, config=None):
        self.split = config.get("split")
        self.root = Path(config.get("root"))

        self.image_root = self.root / "pyramid"

        self.p_uncond = config.get("p_uncond", 0)

        self.mag = config.get("magnification", "5x")

        # load patches list

        self.patches = np.load(self.root / f"patch_list_{self.mag}.npy", allow_pickle=True)

        # Load metadata
        arr = np.load(self.root / f"train_test_split/{self.split}_all.npz", allow_pickle=True)
        self.indices = arr[f"indices_{self.split}_{self.mag}"]


        # load SSL features
        features = h5py.File(self.root / "ssl_features/features_hipt_ctr.h5", "r")
        self.feat = features[f'feat_{self.mag}']


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx_global = self.indices[idx]

        img_path = self.patches[idx_global]
        tile = np.array(Image.open(self.image_root / img_path))[..., :3]

        image = (tile / 127.5 - 1.0).astype(np.float32)

        # Random horizontal and vertical flips
        if self.split == "train":
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()


        # load 20x and 5x SSL features
        feat_patch = self.feat[idx_global]


        
        # replace patch level emb with all zeros
        if np.random.rand() < self.p_uncond:
            feat_patch = np.zeros_like(feat_patch)

        return {
            "image": image,
            "feat_patch": feat_patch,
            "human_label": "",
        }