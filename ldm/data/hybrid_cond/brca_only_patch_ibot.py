from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py

class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        self.split = config.get("split")
        self.root = Path(config.get("root"))
        self.aug = config.get('aug', True)

        self.image_root = self.root / "pyramid"

        self.p_uncond = config.get("p_uncond", 0)

        # load patches list

        self.patches = np.load(self.root / "patch_list_20x.npy", allow_pickle=True)

        # Load metadata
        arr = np.load(self.root / f"train_test_split/{self.split}.npz", allow_pickle=True)
        self.indices_20x = arr[f"indices_{self.split}_20x"]
        self.indices_5x = arr[f"indices_{self.split}_5x"]


        # load SSL features
        features = h5py.File(self.root / "ssl_features/features_ibot.h5", "r")
        self.feat_20x = features['feat_20x']

        self.return_5x_img = config.get("return_5x_img", False)


    def __len__(self):
        return len(self.indices_20x)

    def __getitem__(self, idx):

        idx_20x = self.indices_20x[idx]

        img_path = self.patches[idx_20x]
        tile = np.array(Image.open(self.image_root / img_path))

        image = (tile / 127.5 - 1.0).astype(np.float32)

        # Random horizontal and vertical flips
        if self.split == "train" and self.aug:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()

        # load 20x and 5x SSL features
        feat_20x = self.feat_20x[idx_20x].astype(np.float32)


        
        # replace patch and region level emb with all zeros
        if np.random.rand() < self.p_uncond:
            feat_20x = np.zeros_like(feat_20x)



        return {
            "image": image,
            "feat_patch": feat_20x,
            "human_label": "",
        }