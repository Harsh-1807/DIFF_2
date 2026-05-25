# -*- coding: utf-8 -*-
"""
Stage 2: Train CorrDiff UNet  (Flow-Matching Corrector  OR  CorrDiff Residual)
===============================================================================
DUAL-MODE TRAINER  ──  plug-and-play, 4 × H100, anti-overfit

SCIENTIFIC RIGOR VERSION + SPATIAL AUTO-TUNER + ANTI-SMOOTHNESS FIXES
───────────────────────────────────────────────────────────────────────
  * AUTO-TUNER     : Runs a pre-flight test to find the optimal LR and Spatial Weights.
  * QDM CALIBRATION : Fitted strictly on Training Subset (Zero Data Leakage).
  * TEMPORAL COND   : Dynamic spatial interpolation to guarantee [B, T, H, W] alignment.
  * ARCHITECTURE    : v11 Multi-Stem. d2m and var_map passed as separate 1-ch inputs.
  * LOSS            : Pure generative objective (Min-SNR + High-Freq + Sobel Grad).
  * ANTI-SMOOTHNESS : 10x LR on Zero-Gates, mu_train jitter, and high CFG Scale.
"""

import os
import math
import time
import argparse
import traceback
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    warnings.warn("albumentations not found. pip install albumentations opencv-python-headless")

from Dataset import UpscaleDataset
from Network import CorrDiffRegressor, UNet, FlowMatching, PhysicsGuide, QDM


# ══════════════════════════════════════════════════════════════════════════════
# 0.  MODE TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_MODE = os.environ.get("TRAIN_MODE", "corrdiff_residual")
assert TRAIN_MODE in ("flow_matching", "corrdiff_residual"), \
    f"Bad TRAIN_MODE={TRAIN_MODE!r}"

SIGMA_DATA = 0.2171  # Stage-1 residual std — do NOT change


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS
# ══════════════════════════════════════════════════════════════════════════════
RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
REG_CKPT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/SET1/checkpoints/regressor/regressor_best.pth"
CKPT_DIR = "checkpoints/"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
BATCH        = 16               # Push to 16-24 (Max out 80GB VRAM per GPU)
ACCUM_STEPS  = 4                # 16 * 4 GPUs * 2 = 128 Effective Batch (Ideal for EDM)
LR           = 2e-5            
MIN_LR       = LR * 0.05
EPOCHS       = 1500
PATIENCE     = 200
T_COND       = 3                
PRECIP_CH    = 0
BASE_CH      = 160
CHANNEL_MULT = (1, 2, 4, 4)     # Un-chokes the mesoscale feature extraction
NRB          = 3                # Deeper spatial processing before downsampling
DROPOUT      = 0.05
FM_STEPS     = 12
CFG_SCALE    = 3.0              # [ANTI-SMOOTHNESS] Bumped from 1.5 to 3.0
P_CFG_DROP   = 0.15
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995
N_ENS        = 2
REG_IN_CH    = 2            
REG_D2M_CH   = 1            
D2M_CH       = 1            
UNET_D2M_CH  = 1            
UNET_VAR_MAP_CH = 1         
TOPO_CH      = 3
GLOBAL_DIM   = 2
UNET_IN_CH   = 1 + 1 + T_COND   # noisy_res + mu + temporal_frames = 7
DS_FACTOR    = 4                


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EDM NOISE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=SIGMA_DATA, rho=7.0):
    sigma_max = 2.0 * sigma_data
    steps = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
    return (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

_EDM_CPU = build_edm_schedule(FM_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EMA
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)

    def apply_to(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})

    def restore(self, model, state):
        model.load_state_dict(state)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def _make_aug_pipeline():
    if not HAS_ALB:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, 
                           rotate_limit=8, border_mode=0, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.08, 
                                   contrast_limit=0.08, p=0.35),
        A.GaussNoise(var_limit=(1e-6, 3e-4), p=0.20),
    ], additional_targets={
        "topo": "image",
        "d2m": "image",
    })

_AUG_PIPELINE = _make_aug_pipeline()

def augment_sample(fp_t, topo_t, d2m_t, aug_prob=0.5):
    def _rederive_coarse(fp):
        return F.avg_pool2d(fp.unsqueeze(0),
                            kernel_size=DS_FACTOR, stride=DS_FACTOR).squeeze(0)

    if _AUG_PIPELINE is None or np.random.rand() > aug_prob:
        return fp_t, topo_t, _rederive_coarse(fp_t), d2m_t

    fp_np   = fp_t.squeeze(0).numpy().astype(np.float32)
    topo_np = topo_t.squeeze(0).numpy().astype(np.float32)
    d2m_np  = (d2m_t.squeeze(0).numpy().astype(np.float32)
               if d2m_t is not None else np.zeros_like(fp_np))

    result   = _AUG_PIPELINE(image=fp_np, topo=topo_np, d2m=d2m_np)
    fp_aug   = torch.from_numpy(result["image"]).unsqueeze(0)
    topo_aug = torch.from_numpy(result["topo"]).unsqueeze(0)
    d2m_aug  = (torch.from_numpy(result["d2m"]).unsqueeze(0)
                if d2m_t is not None else None)
    coarse_aug = _rederive_coarse(fp_aug)
    return fp_aug, topo_aug, coarse_aug, d2m_aug


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TOPO HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_slope_aspect(elev, global_elev_max=8600.0, global_slope_max=1.5):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                       device=elev.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                       device=elev.device).view(1, 1, 3, 3)
    e = elev.float()
    dx = F.conv2d(e, kx, padding=1)
    dy = F.conv2d(e, ky, padding=1)
    slope = torch.sqrt(dx**2 + dy**2 + 1e-8)
    aspect = torch.atan2(dy, dx)
    
    def global_norm(t, g_min, g_max):
        return 2 * (t - g_min) / (g_max - g_min + 1e-8) - 1

    return torch.cat([
        global_norm(e, 0.0, global_elev_max), 
        global_norm(slope, 0.0, global_slope_max), 
        aspect / math.pi 
    ], dim=1)

def expand_topo(topo_1ch):
    return torch.cat([compute_slope_aspect(topo_1ch[i:i+1])
                      for i in range(topo_1ch.shape[0])], dim=0)

def build_coarse_input(coarse, var_map):
    Hc, Wc = coarse.shape[-2], coarse.shape[-1]
    return torch.cat([coarse, F.adaptive_avg_pool2d(var_map, (Hc, Wc))], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TEMPORAL CONDITIONING 
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_cond(batch, dev, n_frames=T_COND):
    target_shape = batch["fine"].shape[-2:]

    if "tc_frames" in batch and batch["tc_frames"] is not None:
        tc = batch["tc_frames"].to(dev, non_blocking=True)
        if tc.dim() == 5 and tc.shape[2] == 1:
            tc = tc.squeeze(2)
        if tc.shape[1] >= n_frames:
            tc = tc[:, :n_frames]
        else:
            pad_len = n_frames - tc.shape[1]
            coarse = batch["coarse"].to(dev, non_blocking=True)
            if coarse.dim() == 4 and coarse.shape[1] > 1:
                coarse = coarse[:, 0:1] 
            pad = coarse.expand(-1, pad_len, -1, -1)
            tc = torch.cat([tc, pad], dim=1)

        if tc.shape[-2:] != target_shape:
            tc = F.interpolate(tc, size=target_shape, mode='bilinear', align_corners=False)
        return tc
    
    coarse = batch["coarse"].to(dev, non_blocking=True)
    if coarse.dim() == 4 and coarse.shape[1] > 1:
        coarse = coarse[:, 0:1]
    tc = coarse.expand(-1, n_frames, -1, -1)
    return F.interpolate(tc, size=target_shape, mode='bilinear', align_corners=False)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUXILIARY LOSSES & METRICS
# ══════════════════════════════════════════════════════════════════════════════
def hybrid_sigma_loss(pred, target, sigma, hf_weight=3.0, grad_weight=0.6):
    eps = 1e-6
    snr = (SIGMA_DATA ** 2) / (sigma ** 2 + eps)
    min_snr_weight = torch.clamp(snr, max=5.0)

    mae = torch.abs(pred.float() - target.float())
    weighted_mae = (mae * min_snr_weight).mean()

    pred_fft = torch.fft.rfft2(pred.float()).abs()
    targ_fft = torch.fft.rfft2(target.float()).abs()
    hf_loss = torch.abs(pred_fft[..., -pred_fft.shape[-1]//2:] - 
                        targ_fft[..., -targ_fft.shape[-1]//2:]).mean()

    def sobel_loss(x):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    grad_loss = torch.abs(sobel_loss(pred) - sobel_loss(target)).mean()
    return weighted_mae + (hf_weight * hf_loss) + (grad_weight * grad_loss)
    
def get_grad_loss(pred, target):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    def get_grads(x):
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    return F.l1_loss(get_grads(pred), get_grads(target))

@torch.no_grad()
def weighted_pcc(pred, target, lat_w=None):
    B = pred.shape[0]
    p = pred.float().view(B, -1); t = target.float().view(B, -1)
    if lat_w is not None:
        w = lat_w.view(1, -1).expand_as(p)
        w = w / w.sum(dim=1, keepdim=True)
        pm = (p * w).sum(dim=1, keepdim=True); tm = (t * w).sum(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        r = (pm_c * tm_c * w).sum(dim=1) / (
            torch.sqrt((pm_c**2 * w).sum(dim=1) * (tm_c**2 * w).sum(dim=1)) + 1e-8)
    else:
        pm = p.mean(dim=1, keepdim=True); tm = t.mean(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        p_l2 = torch.sqrt((pm_c**2).sum(dim=1))
        t_l2 = torch.sqrt((tm_c**2).sum(dim=1))
        r = (pm_c * tm_c).sum(dim=1) / (p_l2 * t_l2 + 1e-8)
    return r.mean().item()

@torch.no_grad()
def crps_ensemble(samples, target):
    N=samples.shape[0]
    mae=(samples-target.unsqueeze(0)).abs().mean(0)
    pair=(samples.unsqueeze(0)-samples.unsqueeze(1)).abs()
    return (mae-0.5/N/(N-1)*pair.sum([0,1])).mean().item()

@torch.no_grad()
def psd_tail_ratio(pred, target, hff=0.3):
    P=torch.fft.rfft2(pred.float()).abs(); T=torch.fft.rfft2(target.float()).abs()
    c=int((1-hff)*P.shape[-1])
    return (P[...,c:].mean()/(T[...,c:].mean()+1e-8)).item()

@torch.no_grad()
def fractions_skill_score(pred, target, threshold=0.5, window=5):
    p_bin = (pred > threshold).float()
    t_bin = (target > threshold).float()
    p_frac = F.avg_pool2d(p_bin, kernel_size=window, stride=1, padding=window//2)
    t_frac = F.avg_pool2d(t_bin, kernel_size=window, stride=1, padding=window//2)
    mse = ((p_frac - t_frac)**2).mean(dim=[-1, -2])
    ref = (p_frac**2).mean(dim=[-1, -2]) + (t_frac**2).mean(dim=[-1, -2])
    fss = 1.0 - mse / (ref + 1e-8)
    return fss.mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DDP
# ══════════════════════════════════════════════════════════════════════════════
def setup():
    rank=int(os.environ.get("RANK",0)); ws=int(os.environ.get("WORLD_SIZE",1))
    lr_=int(os.environ.get("LOCAL_RANK",0))
    if ws>1: dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   
    torch.backends.cudnn.allow_tf32 = True
    return rank,ws,lr_,torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")

def ar(t, ws):
    if ws>1 and dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.SUM); t/=ws
    return t


# ══════════════════════════════════════════════════════════════════════════════
# 10. REGRESSOR LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_regressor(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev)
    reg = CorrDiffRegressor(
        in_channels=ck.get("reg_in_channels", REG_IN_CH), out_channels=1,
        base_channels=64, channel_mult=(1,2,4), num_blocks=2,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        d2m_channels=ck.get("d2m_channels", REG_D2M_CH),   
        use_d2m=ck.get("use_d2m", True),
    ).to(dev)
    
    model_dict = reg.state_dict()
    pretrained_dict = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    
    matching_weights = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(matching_weights)
    reg.load_state_dict(model_dict, strict=False)
    
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)
    return reg

# ══════════════════════════════════════════════════════════════════════════════
# 11. SPATIAL AUTO-TUNER
# ══════════════════════════════════════════════════════════════════════════════
class SpatialAutoTuner:
    """
    Runs a rapid pre-flight search on a tiny subset of data.
    Evaluates strictly on PSD Tail Ratio and FSS to find the hyperparameters
    that produce the sharpest, most intense spatial structures.
    """
    def __init__(self, model_class, ds, dev, num_train=30, num_val=10):
        self.dev = dev
        self.trl = DataLoader(Subset(ds, range(0, num_train * BATCH)), BATCH, shuffle=True, drop_last=True)
        self.val = DataLoader(Subset(ds, range(num_train * BATCH, (num_train + num_val) * BATCH)), BATCH, shuffle=False)
        self.model_class = model_class
        
        # [ANTI-SMOOTHNESS] Cranked the baseline search grid to aggressively hunt for sharp outputs
        self.grid = [
            {"lr": 2e-4, "hf": 5.0, "grad": 2.0},  # Aggressive baseline
            {"lr": 5e-5, "hf": 8.0, "grad": 3.0},  # Heavy edge enforcement
            {"lr": 1e-4, "hf": 12.0, "grad": 4.0}, # Maximum sharpness penalty
        ]

    def run(self, regressor, edm_schedule):
        print("\n" + "═"*60)
        print("🚀 INITIATING SPATIAL AUTO-TUNER (Optimizing for Sharpness)")
        print("═"*60)
        
        best_score = -1.0
        best_params = self.grid[0]

        for idx, p in enumerate(self.grid):
            print(f"\n[Trial {idx+1}/{len(self.grid)}] LR: {p['lr']:.1e}, HF_wt: {p['hf']}, Grad_wt: {p['grad']}")
            temp_model = self.model_class().to(self.dev)
            opt = torch.optim.AdamW(temp_model.parameters(), lr=p['lr'])
            
            temp_model.train()
            for b in self.trl:
                opt.zero_grad(set_to_none=True)
                fp = b["fine"].to(self.dev, non_blocking=True)[:, PRECIP_CH:PRECIP_CH+1]
                xi_raw = b["coarse"].to(self.dev, non_blocking=True)
                tc = build_temporal_cond(b, self.dev)
                tp = expand_topo(b["topo"].to(self.dev, non_blocking=True))
                var_map = b["var_map"].to(self.dev, non_blocking=True)
                gf = torch.stack([b["doy"], b["hour"]], 1).float().to(self.dev, non_blocking=True)
                d2m = b["d2m"].to(self.dev, non_blocking=True) if "d2m" in b else None
                
                with torch.no_grad():
                    mu = regressor(build_coarse_input(xi_raw, var_map), topo=tp, global_features=gf, d2m=d2m)
                
                residual = fp - mu
                sigma_idx = torch.randint(0, len(edm_schedule), (fp.shape[0],))
                sigma_t = edm_schedule[sigma_idx].to(self.dev).view(-1,1,1,1)
                x_t = residual + sigma_t * torch.randn_like(residual)
                c_in = 1. / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                c_out = sigma_t * SIGMA_DATA / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                c_skip = SIGMA_DATA**2 / (sigma_t**2 + SIGMA_DATA**2)
                
                with autocast(device_type=self.dev.type):
                    D_pred = temp_model(c_in * torch.cat([x_t, mu, tc], dim=1), (sigma_t.log()/4).view(-1), 
                                        topo=tp, global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
                    x0_pred = c_skip * x_t[:, :1] + c_out * D_pred
                    loss = hybrid_sigma_loss(x0_pred, residual, sigma_t, hf_weight=p['hf'], grad_weight=p['grad']) + (1.5 * get_grad_loss(x0_pred, residual))
                
                loss.backward()
                opt.step()

            temp_model.eval()
            psd_sum, fss_sum = 0.0, 0.0
            with torch.no_grad():
                for b in self.val:
                    fp = b["fine"].to(self.dev, non_blocking=True)[:, PRECIP_CH:PRECIP_CH+1]
                    xi_raw = b["coarse"].to(self.dev, non_blocking=True)
                    tc = build_temporal_cond(b, self.dev)
                    tp = expand_topo(b["topo"].to(self.dev, non_blocking=True))
                    var_map = b["var_map"].to(self.dev, non_blocking=True)
                    gf = torch.stack([b["doy"], b["hour"]], 1).float().to(self.dev, non_blocking=True)
                    d2m = b["d2m"].to(self.dev, non_blocking=True) if "d2m" in b else None
                    
                    mu = regressor(build_coarse_input(xi_raw, var_map), topo=tp, global_features=gf, d2m=d2m)
                    residual = fp - mu
                    
                    sigma_t = torch.full((fp.shape[0],), SIGMA_DATA, device=self.dev).view(-1,1,1,1)
                    x_t = residual + sigma_t * torch.randn_like(residual)
                    c_in = 1. / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_out = sigma_t * SIGMA_DATA / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_skip = SIGMA_DATA**2 / (sigma_t**2 + SIGMA_DATA**2)
                    
                    with autocast(device_type=self.dev.type):
                        D_pred = temp_model(c_in * torch.cat([x_t, mu, tc], dim=1), (sigma_t.log()/4).view(-1), 
                                            topo=tp, global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
                        x0_pred = c_skip * x_t[:, :1] + c_out * D_pred
                    
                    final_pred = x0_pred + mu
                    final_pred = PhysicsGuide.apply(final_pred, xi_raw, enforce_mass=True, enforce_dry=True)
                    psd_sum += psd_tail_ratio(final_pred, fp)
                    fss_sum += fractions_skill_score(final_pred, fp)
            
            if math.isnan(psd_sum) or math.isnan(fss_sum):
                score = -1.0
            else:
                score = (psd_sum + (fss_sum * 10)) 
            
            print(f"--> Sharpness Score: {score:.4f} (PSD: {psd_sum/max(1, len(self.val)):.3f}, FSS: {fss_sum/max(1, len(self.val)):.3f})")
            
            if score > best_score:
                best_score = score
                best_params = p
                
            del temp_model, opt
            torch.cuda.empty_cache()
                
        print("═"*60)
        print(f"🏆 AUTO-TUNE COMPLETE. Winning Params: {best_params}")
        print("═"*60 + "\n")
        return best_params


# ══════════════════════════════════════════════════════════════════════════════
# 12. DDIM SAMPLER
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ddim_sample(raw_model, mu, tc, tp, gf, d2m, var_map, edm_schedule, dev):
    B = mu.shape[0]
    x_t = torch.randn_like(mu) * SIGMA_DATA
    sigmas = edm_schedule.to(dev)

    for i, sigma_cur in enumerate(sigmas):
        s_cur  = sigma_cur.view(1, 1, 1, 1)
        c_in   = 1. / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_out  = s_cur * SIGMA_DATA / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_skip = SIGMA_DATA**2 / (s_cur**2 + SIGMA_DATA**2)
        c_n    = (sigma_cur.log() / 4).expand(B)
        x_in   = torch.cat([x_t, mu, tc], dim=1)
        D_pred = raw_model(c_in * x_in, c_n, topo=tp,
                           global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
        x0_hat = c_skip * x_t[:, :1] + c_out * D_pred
        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1].view(1, 1, 1, 1)
            x_t = x0_hat + sigma_next * (x_t - x0_hat) / s_cur.clamp(min=1e-8)
        else:
            x_t = x0_hat
    return x_t


# ══════════════════════════════════════════════════════════════════════════════
# 13. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    rank,ws,lr_,dev=setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    SAVE    =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_best.pth")
    LATEST  =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_latest.pth")
    QDM_PATH=os.path.join(CKPT_DIR,f"qdm_{TRAIN_MODE}.pth")

    edm_schedule=_EDM_CPU.clone()

    ds =UpscaleDataset(RF_PATH,ORO_PATH,d2m_file=D2M_PATH,
                       split="train",normalize=True,device="cpu")
                       
    n = len(ds)
    tst = int(0.20 * n)  # First 20% for Testing
    trn = int(0.70 * n)  # Next 70% for Training

    _loader_kwargs = dict(
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,   
        prefetch_factor=4,         
        drop_last=True,
    )
    
    # ── TESTING (First 20% - Vaulted) ──
    test = DataLoader(Subset(ds, range(0, tst)), BATCH, shuffle=False,
                      num_workers=8, pin_memory=True,
                      persistent_workers=True, prefetch_factor=2)

    # ── TRAINING (Next 70%) ──
    trl = DataLoader(Subset(ds, range(tst, tst + trn)), BATCH, shuffle=True, **_loader_kwargs)
    
    # ── VALIDATION (Last 10%) ──
    val = DataLoader(Subset(ds, range(tst + trn, n)), BATCH, shuffle=False,
                     num_workers=8, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

    reg=load_regressor(REG_CKPT,dev)
    if rank==0: print(f"[Stage-1] Regressor loaded (frozen)")

    # ── AUTO-TUNER EXECUTION (DDP Safe) ───────────────────────────────
    tuned_lr, tuned_hf, tuned_grad = LR, 3.0, 0.6
    
    if TRAIN_MODE == "corrdiff_residual":
        if rank == 0:
            def unet_maker():
                return UNet(
                    in_channels=UNET_IN_CH, out_channels=1, base_channels=BASE_CH,
                    channel_mult=CHANNEL_MULT, num_res_blocks=NRB, dropout=DROPOUT,
                    global_dim=GLOBAL_DIM, use_bottleneck_attention=True,
                    topo_channels=TOPO_CH, use_d2m=True, d2m_channels=UNET_D2M_CH,         
                    use_var_map=True, var_map_channels=UNET_VAR_MAP_CH, temporal_frames=T_COND,
                )
            tuner = SpatialAutoTuner(unet_maker, ds, dev)
            best_params = tuner.run(reg, edm_schedule)
            param_tensor = torch.tensor([best_params['lr'], best_params['hf'], best_params['grad']], 
                                        dtype=torch.float32, device=dev)
        else:
            param_tensor = torch.zeros(3, dtype=torch.float32, device=dev)

        if ws > 1 and dist.is_initialized():
            dist.barrier()
            dist.broadcast(param_tensor, src=0)

        tuned_lr   = 2e-4
        tuned_hf   = 5.0
        tuned_grad = 1.0
    # ──────────────────────────────────────────────────────────────────

    if TRAIN_MODE=="flow_matching":
        fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=CFG_SCALE); loss_label="L_flow  "
    else:
        fm=None; loss_label="L_denoise"

    model=UNet(
        in_channels=UNET_IN_CH, out_channels=1, base_channels=BASE_CH,
        channel_mult=CHANNEL_MULT, num_res_blocks=NRB, dropout=DROPOUT,
        global_dim=GLOBAL_DIM, use_bottleneck_attention=True,
        topo_channels=TOPO_CH, 
        use_d2m=True,
        d2m_channels=UNET_D2M_CH,         
        use_var_map=True,                  
        var_map_channels=UNET_VAR_MAP_CH,  
        temporal_frames=T_COND,
    ).to(dev)

    if ws>1:
            model=nn.parallel.DistributedDataParallel(
            model,device_ids=[lr_],find_unused_parameters=False)
    raw=model.module if ws>1 else model

    # 2. COMPILE SECOND
    model = torch.compile(model)

    ema   = EMA(raw, decay=EMA_DECAY)

    # ── [ANTI-SMOOTHNESS] FORCE GATES TO WAKE UP (10x LR) ──
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]
    base_params = [p for n, p in model.named_parameters() if 'gate' not in n]

    opt = AdamW([
        {'params': base_params, 'lr': tuned_lr},
        {'params': gate_params, 'lr': tuned_lr * 10.0}  # 10x LR multiplier for zero-gates
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    scaler=GradScaler(device=dev.type)
    sched =CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=1,eta_min=MIN_LR)

    start=0; best_val_loss=float('inf'); no_improve=0

    if os.path.exists(LATEST):
        print(f"[RESUME] Loading model weights from {LATEST}...")
        ck = torch.load(LATEST, map_location=dev)
        
        # 1. Get current model state
        model_dict = raw.state_dict()
        
        # 2. Normalize checkpoint keys
        pretrained_dict = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
        
        # 3. Filter: Only keep weights where the name exists AND the shape matches
        matching_weights = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and model_dict[k].shape == v.shape
        }
        
        # 4. Update and load model weights
        model_dict.update(matching_weights)
        raw.load_state_dict(model_dict)
        
        print(f"Successfully loaded {len(matching_weights)}/{len(model_dict)} layers.")

        start = ck.get("epoch", 0) + 1
        best_val_loss = ck.get("val_loss", float('inf'))
        no_improve = ck.get("no_improve", 0)
        
        # --- EMA FIX APPLIED HERE ---
        if "ema_shadow" in ck:
            old_ema = {k: v.to(dev) for k, v in ck["ema_shadow"].items()}
            # Apply the exact same shape-filter to the EMA dict
            for k, v in old_ema.items():
                if k in ema.shadow and ema.shadow[k].shape == v.shape:
                    ema.shadow[k] = v
            
        if rank == 0: 
            print(f"[RESUME] Architecture changed. Starting with fresh optimizer state.")
            print(f"[RESUME] ep={start}  best_val_loss={best_val_loss:.4f}")
            
    if rank==0:
        np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL]  UNet {np_/1e6:.2f}M  mode={TRAIN_MODE}  sigma_data={SIGMA_DATA}")
        print(f"[COND]   UNET_D2M_CH={UNET_D2M_CH}  UNET_VAR_MAP_CH={UNET_VAR_MAP_CH}  T_COND={T_COND}")
        print(f"[OPTIM]  LR={tuned_lr:.1e}  HF_WT={tuned_hf:.1f}  GRAD_WT={tuned_grad:.1f}  ACCUM={ACCUM_STEPS}")
        print(f"[PERF]   cudnn.benchmark=True  tf32=True  persistent_workers=True  prefetch_factor=2")
        hdr=(f"{'Ep':>5}|{loss_label:>10}|{'ValLoss':>8}|{'wPCC':>7}|{'CRPS':>8}|{'PSD_r':>7}|{'FSS':>6}|{'LR':>9}")
        print(hdr); print("-" * len(hdr))

    lat_w=None

    for ep in range(start, EPOCHS):
        model.train()
        t0=time.time(); sum_ml=nb=0.
        opt.zero_grad(set_to_none=True)
        _opt_steps=0   

        for step, b in enumerate(trl, 1):
            try:
                fp       = b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                topo_1ch = b["topo"].to(dev, non_blocking=True)
                xi_raw   = b["coarse"].to(dev, non_blocking=True)
                d2m      = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                var_map  = b["var_map"].to(dev, non_blocking=True)
                gf       = torch.stack([b["doy"],b["hour"]],1).float().to(dev, non_blocking=True)
                
                tc = build_temporal_cond(b, dev)

                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-1); topo_1ch = topo_1ch.flip(-1)
                    if d2m is not None: d2m = d2m.flip(-1)
                    var_map = var_map.flip(-1)
                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-2); topo_1ch = topo_1ch.flip(-2)
                    if d2m is not None: d2m = d2m.flip(-2)
                    var_map = var_map.flip(-2)

                xi_raw = F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)
                tp  = expand_topo(topo_1ch)
                xi  = build_coarse_input(xi_raw, var_map)

                with torch.no_grad():
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)

                # ── [ANTI-SMOOTHNESS] JITTER MU DURING TRAINING ──
                # Degrade mu slightly so the UNet is forced to do the heavy lifting
                mu_train = mu + torch.randn_like(mu) * 0.05 * mu.std()

                residual = fp - mu_train
                cfg_drop = (torch.rand(fp.shape[0], device=dev) < P_CFG_DROP)

                if TRAIN_MODE == "flow_matching":
                    x_t, t_vec, v_star = fm.get_train_sample(residual)
                    x_in    = torch.cat([x_t, mu_train, tc], dim=1) # Use mu_train
                    with autocast(device_type=dev.type):
                        v_pred   = model(x_in, t_vec, topo=tp, global_features=gf,
                                         cfg_drop=cfg_drop, d2m=d2m, var_map=var_map, T=T_COND)
                        loss     = fm.loss(v_pred, v_star) / ACCUM_STEPS
                else:
                    idx     = torch.randint(0, len(edm_schedule), (fp.shape[0],))
                    sigma_t = edm_schedule[idx].to(dev).view(-1,1,1,1)
                    eps     = torch.randn_like(residual)
                    x_t     = residual + sigma_t*eps
                    c_in    = 1./torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_out   = sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_skip  = SIGMA_DATA**2/(sigma_t**2 + SIGMA_DATA**2)
                    c_n     = (sigma_t.log()/4).view(fp.shape[0])
                    x_in    = torch.cat([x_t, mu_train, tc], dim=1) # Use mu_train
                    with autocast(device_type=dev.type):
                        D_pred  = model(c_in*x_in, c_n, topo=tp, global_features=gf,
                                        cfg_drop=cfg_drop, d2m=d2m, var_map=var_map, T=T_COND)
                        x0_pred = c_skip*x_t[:,:1] + c_out*D_pred
                        grad_penalty = get_grad_loss(x0_pred, residual)
                        loss = (hybrid_sigma_loss(x0_pred, residual, sigma_t, hf_weight=tuned_hf, grad_weight=tuned_grad) + (1.5 * grad_penalty)) / ACCUM_STEPS

                scaler.scale(loss).backward()

                if step % ACCUM_STEPS == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if not torch.isfinite(loss):
                        print("Detected NaN loss, skipping batch")
                        opt.zero_grad(set_to_none=True)
                        continue 
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    ema.update(raw)   
                    _opt_steps += 1

                with torch.no_grad():
                    sum_ml+=loss.item()*ACCUM_STEPS; nb+=1

            except Exception:
                if rank==0: traceback.print_exc()
                raise

        sched.step()
        te_ml=ar(torch.tensor(sum_ml/max(nb,1),device=dev),ws).item()
        lr_now=opt.param_groups[0]["lr"]

        if ep % 10 == 0 or ep == EPOCHS - 1:
            # ── VALIDATION (EMA weights - Using clean mu) ───────────────────────
            live_state=deepcopy(raw.state_dict()); ema.apply_to(raw); model.eval()
            v_loss=torch.tensor(0.,device=dev); vn=torch.tensor(0,device=dev,dtype=torch.long)
            pcc_sum=crps_sum=psd_sum=fss_sum=0.; vb=0
            
            with torch.no_grad():
                for b in val:
                    fp      = b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                    tp      = expand_topo(b["topo"].to(dev, non_blocking=True))
                    gf      = torch.stack([b["doy"],b["hour"]],1).float().to(dev, non_blocking=True)
                    xi_raw  = b["coarse"].to(dev, non_blocking=True)
                    var_map = b["var_map"].to(dev, non_blocking=True)
                    d2m     = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                    tc      = build_temporal_cond(b, dev)
                    xi      = build_coarse_input(xi_raw, var_map)

                    with autocast(device_type=dev.type):
                        mu = reg(xi, topo=tp, global_features=gf, d2m=d2m) # No jitter for inference!

                    residual=fp-mu

                    if TRAIN_MODE=="flow_matching":
                        x_t,t_vec,v_star=fm.get_train_sample(residual)
                        x_in=torch.cat([x_t,mu,tc],dim=1)
                        with autocast(device_type=dev.type):
                            v_pred=model(x_in,t_vec,topo=tp,global_features=gf,
                                         d2m=d2m,var_map=var_map,T=T_COND)
                            l_val=fm.loss(v_pred,v_star)
                    else:
                        idx=torch.randint(0,len(edm_schedule),(fp.shape[0],))
                        sigma_t=edm_schedule[idx].to(dev).view(-1,1,1,1)
                        eps=torch.randn_like(residual); x_t=residual+sigma_t*eps
                        c_in=1./torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                        c_out=sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                        c_skip=SIGMA_DATA**2/(sigma_t**2+SIGMA_DATA**2)
                        c_n=(sigma_t.log()/4).view(fp.shape[0])
                        x_in=torch.cat([x_t,mu,tc],dim=1)
                        with autocast(device_type=dev.type):
                            D_pred=model(c_in*x_in,c_n,topo=tp,global_features=gf,
                                         d2m=d2m,var_map=var_map,T=T_COND)
                            x0_pred=c_skip*x_t[:,:1]+c_out*D_pred
                            l_val=hybrid_sigma_loss(x0_pred, residual, sigma_t, hf_weight=tuned_hf, grad_weight=tuned_grad)

                    v_loss+=l_val*fp.shape[0]; vn+=fp.shape[0]

                    samples=[]
                    for _ in range(N_ENS):
                        if TRAIN_MODE=="flow_matching":
                            x_cond=torch.cat([mu,tc],dim=1)
                            s=fm.sample(raw,x_cond,topo=tp,global_features=gf,
                                        d2m=d2m,var_map=var_map,cfg_scale=CFG_SCALE,T=T_COND)+mu
                        else:
                            s=ddim_sample(raw,mu,tc,tp,gf,d2m,var_map,edm_schedule,dev)+mu
                        s=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                        samples.append(s)

                    samples_t=torch.stack(samples); best_s=samples_t.mean(0)
                    pcc_sum +=weighted_pcc(best_s,fp,lat_w)
                    crps_sum+=crps_ensemble(samples_t,fp)
                    psd_sum +=psd_tail_ratio(best_s,fp)
                    fss_sum +=fractions_skill_score(best_s,fp,threshold=0.5,window=5)
                    vb+=1

            ema.restore(raw,live_state); model.train()

            if ws>1 and dist.is_initialized():
                dist.barrier()
                for t__ in [v_loss,vn]: dist.all_reduce(t__,op=dist.ReduceOp.SUM)
                dist.barrier()

            vw=(v_loss/vn.clamp(1)).item()
            wpcc=pcc_sum/max(vb,1); crps_v=crps_sum/max(vb,1)
            psd_r=psd_sum/max(vb,1); fss_v=fss_sum/max(vb,1)
            el=time.time()-t0

            if rank==0:
                star=" ★" if vw < best_val_loss else ""
                print(f"{ep:>5}|{te_ml:>10.5f}|{vw:>8.4f}|{wpcc:>7.4f}|{crps_v:>8.4f}"
                      f"|{psd_r:>7.3f}|{fss_v:>6.3f}|{lr_now:>9.2e}"
                      f"  [{el:.0f}s]{star}")

                ck_base={
                    "epoch":ep,"model_state_dict":raw.state_dict(),
                    "optimizer_state_dict":opt.state_dict(),"ema_shadow":ema.shadow,
                    "val_loss":vw,"wpcc":wpcc,"crps":crps_v,"psd_tail_ratio":psd_r,
                    "no_improve":no_improve,"train_mode":TRAIN_MODE,"sigma_data":SIGMA_DATA,
                    "unet_in_channels":UNET_IN_CH,"t_cond":T_COND,
                    "d2m_channels":UNET_D2M_CH,"var_map_channels":UNET_VAR_MAP_CH,
                    "reg_d2m_channels":REG_D2M_CH,
                }
                torch.save(ck_base, LATEST)
                
                # ── SCIENTIFIC QDM FIT (ZERO DATA LEAKAGE) ────────────────────────
                # Fits explicitly on `trl` subset. Data is walled off from `test`.
                if (ep + 1) % 15 == 0 or vw < best_val_loss:
                    print("  [QDM] Fitting on training subset to avoid data leakage...")
                    _qdm_latest = QDM(n_quantiles=500)
                    _all_pred, _all_obs = [], []
                    _live2 = deepcopy(raw.state_dict())
                    ema.apply_to(raw); model.eval()
                    with torch.no_grad():
                        for _i, _b in enumerate(trl):
                            if _i >= 50: break 
                            _fp     = _b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                            _tp     = expand_topo(_b["topo"].to(dev, non_blocking=True))
                            _gf     = torch.stack([_b["doy"],_b["hour"]],1).float().to(dev, non_blocking=True)
                            _xi_raw = _b["coarse"].to(dev, non_blocking=True)
                            _vm     = _b["var_map"].to(dev, non_blocking=True)
                            _d2m    = _b["d2m"].to(dev, non_blocking=True) if "d2m" in _b else None
                            _tc     = build_temporal_cond(_b, dev)
                            _xi     = build_coarse_input(_xi_raw, _vm)
                            _mu     = reg(_xi, topo=_tp, global_features=_gf, d2m=_d2m)
                            if fm is not None:
                                _xc = torch.cat([_mu, _tc], dim=1)
                                _s  = fm.sample(raw, _xc, topo=_tp, global_features=_gf,
                                                d2m=_d2m, var_map=_vm, cfg_scale=CFG_SCALE, T=T_COND) + _mu
                            else:
                                _s  = ddim_sample(raw, _mu, _tc, _tp, _gf, _d2m, _vm, edm_schedule, dev) + _mu
                            _s = PhysicsGuide.apply(_s, _xi_raw, enforce_mass=True, enforce_dry=True)
                            _all_pred.append(_s.cpu()); _all_obs.append(_fp.cpu())
                    ema.restore(raw, _live2); model.train()
                    _qdm_latest.fit(torch.cat(_all_pred), torch.cat(_all_obs))
                    _qdm_latest.save(QDM_PATH)

                if vw < best_val_loss:
                    best_val_loss=vw; no_improve=0
                    torch.save(ck_base, SAVE)
                    if (ep+1)%50==0:
                        torch.save(ck_base, os.path.join(
                            CKPT_DIR,f"unet_{TRAIN_MODE}_ep{ep+1:04d}_loss{vw:.4f}.pth"))
                else:
                    no_improve+=5

            if no_improve>=PATIENCE:
                if rank==0: print(f"\n⚠  Early stop ep={ep+1}  best_val_loss={best_val_loss:.4f}")
                break
        else:
            el = time.time() - t0
            if rank == 0:
                print(f"{ep:>5}|{te_ml:>10.5f}|{'--':>8}|{'--':>7}|{'--':>8}"
                      f"|{'--':>7}|{'--':>6}|{lr_now:>9.2e}"
                      f"  [{el:.0f}s]")

    if ws>1 and dist.is_initialized(): dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# 14. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class CorrDiffInference:
    def __init__(self, reg_ckpt, unet_ckpt, qdm_ckpt, device,
                 cfg_scale=CFG_SCALE, n_ens=8):
        self.dev=device; self.cfg_scale=cfg_scale; self.n_ens=n_ens
        self.reg=load_regressor(reg_ckpt,device)
        ck=torch.load(unet_ckpt,map_location=device)
        tc=ck.get("t_cond",T_COND); self.tc=tc
        self.train_mode=ck.get("train_mode","flow_matching")
        self.fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=cfg_scale)
        self.edm_sched=build_edm_schedule(FM_STEPS,sigma_data=ck.get("sigma_data",SIGMA_DATA))
        self.unet=UNet(
            in_channels=ck.get("unet_in_channels",UNET_IN_CH),out_channels=1,
            base_channels=BASE_CH,channel_mult=CHANNEL_MULT,num_res_blocks=NRB,dropout=0.,
            global_dim=GLOBAL_DIM,topo_channels=TOPO_CH,use_d2m=True,
            d2m_channels=ck.get("d2m_channels",UNET_D2M_CH),
            use_var_map=True,
            var_map_channels=ck.get("var_map_channels",UNET_VAR_MAP_CH),
            temporal_frames=tc,
        ).to(device)
        state=({k:v.to(device) for k,v in ck["ema_shadow"].items()} if "ema_shadow" in ck
               else {k.replace("module.",""):v for k,v in ck["model_state_dict"].items()})
        self.unet.load_state_dict(state); self.unet.eval()
        self.qdm=QDM.load(qdm_ckpt) if qdm_ckpt and os.path.exists(qdm_ckpt) else None

    @torch.no_grad()
    def predict(self, coarse, var_map, topo, d2m, doy, hour, tc_frames=None):
        dev=self.dev
        coarse=coarse.to(dev); var_map=var_map.to(dev)
        topo=topo.to(dev); d2m=d2m.to(dev)
        gf=torch.stack([doy.to(dev),hour.to(dev)],dim=1).float()
        xi=build_coarse_input(coarse,var_map); tp=expand_topo(topo)
        
        target_shape = [coarse.shape[-2]*4, coarse.shape[-1]*4]
        if tc_frames is None:
            tc_frames=coarse.expand(-1,self.tc,-1,-1).to(dev)
            tc_frames = F.interpolate(tc_frames, size=target_shape, mode='bilinear', align_corners=False)
        else:
            tc_frames=tc_frames.to(dev)
            if tc_frames.dim() == 5 and tc_frames.shape[2] == 1:
                tc_frames = tc_frames.squeeze(2)
            if tc_frames.shape[-2:] != target_shape:
                tc_frames = F.interpolate(tc_frames, size=target_shape, mode='bilinear', align_corners=False)

        mu=self.reg(xi,topo=tp,global_features=gf,d2m=d2m)

        samples=[]
        for _ in range(self.n_ens):
            if self.train_mode=="flow_matching":
                x_cond=torch.cat([mu,tc_frames],dim=1)
                s=self.fm.sample(self.unet,x_cond,topo=tp,global_features=gf,
                                 d2m=d2m,var_map=var_map,cfg_scale=self.cfg_scale,T=self.tc)+mu
            else:
                s=ddim_sample(self.unet,mu,tc_frames,tp,gf,d2m,var_map,
                              self.edm_sched,dev)+mu
            s=PhysicsGuide.apply(s,coarse,enforce_mass=True,enforce_dry=True)
            if self.qdm is not None: s=self.qdm.apply(s)
            samples.append(s)
        samples=torch.stack(samples)
        return {"mean":samples.mean(0),"std":samples.std(0),"samples":samples,"mu":mu}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",    default=None,choices=["flow_matching","corrdiff_residual"])
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience",type=int,   default=None)
    args=parser.parse_args()
    if args.mode     is not None and "TRAIN_MODE" not in os.environ: TRAIN_MODE=args.mode
    if args.epochs   is not None: EPOCHS=args.epochs
    if args.batch    is not None: BATCH=args.batch
    if args.lr       is not None: LR=args.lr
    if args.dropout  is not None: DROPOUT=args.dropout
    if args.patience is not None: PATIENCE=args.patience
    train()
