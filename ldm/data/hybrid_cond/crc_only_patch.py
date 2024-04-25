from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py

class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        self.root = Path(config.get("root"))

        self.image_root = self.root

        self.p_uncond = config.get("p_uncond", 0)

        self.mag = config.get("magnification")

        # load patches list

        self.patches = np.load(self.root / f"patches_{self.mag}_all.npy", allow_pickle=True)


        # load SSL features
        features = h5py.File(self.root / "features.h5", "r")
        self.feat = features[f'feat_{self.mag}']


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):


        img_path = self.patches[idx]
        tile = np.array(Image.open(self.image_root / img_path))[..., :3]
        if tile.shape != (256, 256, 3):
            return self.__getitem__(idx + 1)

        image = (tile / 127.5 - 1.0).astype(np.float32)

        # Random horizontal and vertical flips
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
        }