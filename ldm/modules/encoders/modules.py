import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SSLEmbedder(nn.Module):
    """concatenate patch and region embeddings"""

    def __init__(
        self,
        patch_ssl_key="feat_20x",
        region_ssl_key="feat_5x",
        patch_embed_dim=384,
        region_embed_dim=768,
        normalize_ssl=False,
    ):
        super().__init__()
        self.patch_ssl_key = patch_ssl_key
        self.region_ssl_key = region_ssl_key

        # project patch/region embedder to txt dimension
        embed_dim = 512
        self.patch_fc = nn.Linear(patch_embed_dim, embed_dim)
        self.region_fc = nn.Linear(region_embed_dim, embed_dim)

        self.normalize_ssl = normalize_ssl

    def forward(self, batch):
        patch_embed = batch[self.patch_ssl_key]

        # use zero vec for region level embeddings if not provided
        if self.region_ssl_key in batch:
            region_embed = batch[self.region_ssl_key]
        else:
            region_embed = torch.zeros((patch_embed.shape[0], 768), device=patch_embed.device)
         
        
        if self.normalize_ssl:
            # normalize ssl embeddings (l2 norm)
            patch_embed = F.normalize(patch_embed, dim=-1)
            region_embed = F.normalize(region_embed, dim=-1)

        patch_embed_proj = self.patch_fc(patch_embed)
        region_embed_proj = self.region_fc(region_embed)

        ssl_embed_list = [patch_embed_proj, region_embed_proj]

        return torch.stack(ssl_embed_list, dim=1)


class PatchEmbedder(nn.Module):
    """only use patch level embeddings"""

    def __init__(
        self,
        patch_ssl_key="feat_patch",
        patch_embed_dim=768,
        target_dim=512,
        normalize_ssl=False,
    ):
        super().__init__()
        self.patch_ssl_key = patch_ssl_key

        # project patch/region embedder to txt dimension
        self.patch_fc = nn.Linear(patch_embed_dim, target_dim)
        self.normalize_ssl = normalize_ssl

    def forward(self, batch):
        patch_embed = batch[self.patch_ssl_key]

        if self.normalize_ssl:
            # normalize ssl embeddings (l2 norm)
            patch_embed = F.normalize(patch_embed, dim=-1)

        patch_embed_proj = self.patch_fc(patch_embed).unsqueeze(1)
        return patch_embed_proj
