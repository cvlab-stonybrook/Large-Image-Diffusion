import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from mydiff import GaussianDiffusion
import numpy as np


device = torch.device(0)


def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model(path, device, checkpoint):
    config = OmegaConf.load(list(path.glob("configs/*-project.yaml"))[-1])
    del config.model.params.first_stage_config.params["ckpt_path"]
    del config.model.params.unet_config.params["ckpt_path"]
    model = load_model_from_config(config, path / f"checkpoints/{checkpoint}", device)
    return model, config




def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def get_conditioning(
    model,
    i,
    j,
    embeddings,
    uncond=False,
    patch_size=64,
    embedding_spatial=(4, 4),
):
    """Find 4 nearest neighbors and extract their embeddings
    e1 - e2       e_top
     |    |  -->    |    -->  e_interp
    e3 - e4       e_bot
    """
    k1 = model.cond_stage_model.patch_ssl_key
    batch = {
        k1: torch.zeros((1, model.cond_stage_model.patch_fc.in_features)).to(device),
    }
    if hasattr(model.cond_stage_model, "region_ssl_key"):
        batch[model.cond_stage_model.region_ssl_key] = torch.zeros(
            (1, model.cond_stage_model.region_fc.in_features)
        ).to(device)
    if not uncond:
        # Coordinates of the 4 nearest neighbors
        i1, i2 = (i // patch_size,) * 2
        j1, j3 = (j // patch_size,) * 2
        i3, i4 = (i // patch_size + 1,) * 2
        j2, j4 = (j // patch_size + 1,) * 2

        # Pad embeddings
        emb_torch = torch.from_numpy(embeddings.copy()).view(embedding_spatial[0], embedding_spatial[1], -1)
        emb_torch = torch.nn.functional.pad(
            emb_torch.reshape(1, embedding_spatial[0], embedding_spatial[1], -1).permute([0, 3, 1, 2]),
            (0, 1, 0, 1),
            mode="replicate",
        )
        emb_torch = (
            emb_torch.permute([0, 2, 3, 1]).contiguous().reshape(embedding_spatial[0] + 1, embedding_spatial[1] + 1, -1)
        )
        # Extract embeddings
        e1 = emb_torch[i1, j1, :]
        e2 = emb_torch[i2, j2, :]
        e3 = emb_torch[i3, j3, :]
        e4 = emb_torch[i4, j4, :]
        # Compute distances
        t1 = (j / patch_size - j1) / (j2 - j1)
        t2 = (i / patch_size - i1) / (i3 - i1)

        # t1 = t1**2
        # t2 = t2**2

        e_top = slerp(t1, e1, e2)
        e_bot = slerp(t1, e3, e4)
        e_interp = slerp(t2, e_top, e_bot)
        batch[k1] = e_interp.unsqueeze(0).to(device)

    return model.get_learned_conditioning(batch)

def get_panorama(model, embedding, sliding_window_size=16):
    f = 4
    lt_sz = 64
    panorama_resolution = (4 * 256, 4 * 256)
    assert panorama_resolution[0] % 8 == 0 and panorama_resolution[1] % 8 == 0
    panorama_latent = torch.randn((1, 3, panorama_resolution[0] // f, panorama_resolution[1] // f)).float().to(device)

    # Diffusion parameters
    diffusion = GaussianDiffusion(T=1000, schedule=model.betas.cpu().numpy())
    t0 = 1000

    guidance = 3.0
    stride = 20
    ddim = 1
    start_t = t0
    steps = t0
    t_range = list(range(start_t, start_t - min(steps * stride, start_t), -stride))

    # Initialize
    atbar = torch.Tensor([diffusion.alphabar[t0 - 1]]).view(1, 1, 1, 1).to(device)
    epsilon = torch.randn_like(panorama_latent)
    x = torch.sqrt(atbar) * panorama_latent + torch.sqrt(1 - atbar) * epsilon

    # Run inference
    model.eval()

    img_cond_lis = []
    no_cond_lis = []
    for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
        for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
            # Prompt
            with torch.no_grad():
                img_cond = get_conditioning(model, j, k, embedding, uncond=False)
                no_cond = get_conditioning(model, j, k, embedding, uncond=True)

            img_cond_lis.append(img_cond)
            no_cond_lis.append(no_cond)
    img_cond_lis = torch.vstack(img_cond_lis)
    no_cond_lis = torch.vstack(no_cond_lis)
    batch_size = len(img_cond_lis)

    for j, t in enumerate(t_range):
        at = diffusion.alpha[t - 1]
        atbar = diffusion.alphabar[t - 1]

        t_cond = torch.tensor(batch_size * 2 * [t]).view(-1)

        # Denoise sliding window views
        with torch.no_grad():
            eps_map = torch.zeros_like(x)
            x0_map = torch.zeros_like(x)
            avg_map = torch.zeros_like(x)

            x_slice_lis = []
            indices_map = {}
            for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
                for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                    # Prompt

                    x_slice = x[:, :, j : j + lt_sz, k : k + lt_sz]

                    x_slice_lis.append(x_slice)
                    indices_map[j, k] = len(x_slice_lis) - 1

            x_slice_lis = torch.vstack(x_slice_lis)

            with torch.cuda.amp.autocast():
                combined_cond = torch.vstack([img_cond_lis, no_cond_lis])
                combined_x = torch.vstack([x_slice_lis] * 2)
                combined_out = model.model.diffusion_model(
                    combined_x,
                    t_cond.float().to(device),
                    context=combined_cond,
                )

                cond_out, uncond_out = torch.tensor_split(combined_out, 2)

            epsilon_combined = (1 + guidance) * cond_out - guidance * uncond_out
            x0_combined = (x_slice_lis / np.sqrt(atbar)) - (epsilon_combined * np.sqrt((1 - atbar) / atbar))

            for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
                for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                    idx = indices_map[j, k]
                    epsilon_slice = epsilon_combined[idx]

                    x0_slice = x0_combined[idx]

                    eps_map[:, :, j : j + lt_sz, k : k + lt_sz] += epsilon_slice
                    x0_map[:, :, j : j + lt_sz, k : k + lt_sz] += x0_slice
                    avg_map[:, :, j : j + lt_sz, k : k + lt_sz] += 1

            x0_pred = x0_map / avg_map
            epsilon = (x.float() - np.sqrt(atbar) * x0_pred) / np.sqrt(1 - atbar)
        # Perform step
        if t > stride:
            z = torch.randn_like(x)
            # z = torch.randn_like(x[0]).tile(x.shape[0],1,1,1)

            atbar_prev = diffusion.alphabar[t - 1 - stride]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)
        else:
            z = torch.zeros_like(x)
            atbar_prev = 1
            beta_tilde = 0

        with torch.no_grad():
            if ddim < 0:
                # DDPM
                x = (1 / np.sqrt(at)) * (x - ((1 - at) / np.sqrt(1 - atbar)) * epsilon) + np.sqrt(beta_tilde) * z
            else:
                # DDIM
                beta_tilde = beta_tilde * ddim
                x = (
                    np.sqrt(atbar_prev) * x0_pred
                    + np.sqrt(1 - atbar_prev - beta_tilde) * epsilon
                    + np.sqrt(beta_tilde) * z
                )
    return x



def vis_panorama_window_weighted(latent, model, sliding_window_size=16, sigma=0.8):
    f = 4
    lt_sz = 64
    out_img = torch.zeros((latent.shape[0], 3, 4 * latent.shape[2], 4 * latent.shape[3])).to(latent.device)
    avg_map = torch.zeros_like(out_img).to(latent.device)

    # Blending kernel that focuses at the center of each patch
    kernel = gaussian_kernel(size=f * lt_sz, sigma=sigma).to(device)

    for i in range(0, latent.shape[2] - lt_sz + 1, sliding_window_size):
        for j in range(0, latent.shape[3] - lt_sz + 1, sliding_window_size):
            with torch.no_grad():
                decoded = model.decode_first_stage(latent[:, :, i : i + lt_sz, j : j + lt_sz])
                out_img[:, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f] += decoded * kernel.view(1, 1, 256, 256)
                avg_map[:, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f] += torch.ones_like(
                    decoded
                ) * kernel.view(1, 1, 256, 256)

    out_img /= avg_map
    out_img = torch.clamp((out_img + 1) / 2.0, min=0.0, max=1.0)
    out_img = (out_img * 255).to(torch.uint8)
    return out_img.cpu().numpy().transpose([0, 2, 3, 1])


def gaussian_kernel(size=64, mu=0, sigma=1):
    x = torch.linspace(-1, 1, size)
    x = torch.stack((x.tile(size, 1), x.tile(size, 1).T), dim=0)

    d = torch.linalg.norm(x - mu, dim=0)
    x = torch.exp(-(d**2) / sigma**2)
    x = x / x.max()
    return x