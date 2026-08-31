"""FloodDualSeg: frozen DINOv3 (RGB) + frozen DOFA (4-band) dual encoder,
ADAC adapters at multiple depths, PPA detail stem, multi-depth fusion,
fine-detail decoder. Input: raw reflectance (B, 4, H, W); output: water logits
(B, 1, H, W).
"""

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import DinoV3Encoder, DofaEncoder
from .clay_encoder import ClayEncoder
from .prithvi_encoder import Prithvi2Encoder
from .terramind_seg_encoder import TerramindEncoder
from .croma_encoder import CromaEncoder
from .scalemae_encoder import ScaleMaeEncoder
from .ssl4eo_dino_encoder import SSL4EODinoEncoder
from .fusion import MultiDepthFusion, GatedMultiDepthFusion, CrossAttnStack
from .heads import BranchAuxHead, disagreement_map
from .ppa import DetailStem
from .decoder import FineDecoder
from .decoders_v2 import UPerNetDecoder, LinkNetDecoder, DPTDecoder


class FloodDualSeg(nn.Module):
    def __init__(
        self,
        dino_name: str = "vit_large_patch16_dinov3.lvd1689m",
        dino_indices=(5, 11, 17, 23),
        dofa_indices=(2, 5, 8, 11),
        fusion_dim: int = 256,
        use_dino: bool = True,
        use_dofa: bool = True,
        adapter=True,            # True / False / list e.g. ["adac","ppad"]
        norm_mode: str = "dataset",
        clay_norm_mode: str = None,  # overrides norm_mode for Clay branch only
        lora_rank: int = 0,      # >0: LoRA on attn qkv of both backbones
        earth_adapter: bool = False,  # in-block Mixture-of-Frequency adapters
        ea_bottleneck: int = 32,
        ea_rho: float = 0.25,
        grad_ckpt: bool = False,  # checkpoint backbone blocks (saves VRAM)
        use_stem: bool = True,   # PPA DetailStem (required by decoder="fine")
        decoder: str = "fine",   # "fine" | "upernet" | "linknet" | "dpt"
        swir_aux_head: bool = False,  # train-time MNDWI regression head
                                      # (SWIR distillation; unused at inference)
        spectral: str = "dofa",  # "dofa" or "clay" (use_dofa gates either)
        clay_indices=(5, 11, 17, 23),
        clay_downsample: int = 1,  # 1 = full-res (patch-8 grid, finer than DINOv3)
        clay_gsd: float = 3.0,
        clay_wavelengths=None,    # list of band wavelengths in µm; None = PS defaults
        clay_band_mean=None,      # per-band mean for Clay norm; None = PS defaults
        clay_band_std=None,       # per-band std  for Clay norm; None = PS defaults
        dino_rgb_indices=(2, 1, 0),  # which input channels feed DINOv3 as R,G,B
        dofa_upsample: int = 1,  # 2 = fine-grid DOFA (input 2x -> 1/8 stride)
        dofa_size: str = "base",  # "base" (ViT-B) or "large" (ViT-L)
        croma_size: int = 256,   # CROMA fixed input grid (patch-8 -> size/8 tokens)
        fusion_opts: dict = None,  # disagreement-aware fusion ladder (see CLAUDE.md)
    ):
        super().__init__()
        assert use_dino or use_dofa
        n = len(dino_indices)
        self.use_dino, self.use_dofa = use_dino, use_dofa
        ea_kw = dict(earth_adapter=earth_adapter, ea_bottleneck=ea_bottleneck,
                     ea_rho=ea_rho, grad_ckpt=grad_ckpt)
        self.dino = DinoV3Encoder(dino_name, dino_indices, adapter,
                                  norm_mode=norm_mode, lora_rank=lora_rank,
                                  rgb_indices=dino_rgb_indices,
                                  **ea_kw) if use_dino else None
        if use_dofa:
            if spectral == "clay":
                self.dofa = ClayEncoder(indices=clay_indices, adapter=adapter,
                                        norm_mode=clay_norm_mode or norm_mode,
                                        downsample=clay_downsample,
                                        gsd=clay_gsd, lora_rank=lora_rank,
                                        wavelengths=clay_wavelengths,
                                        band_mean=clay_band_mean,
                                        band_std=clay_band_std,
                                        **ea_kw)
            elif spectral == "prithvi":
                self.dofa = Prithvi2Encoder(
                    indices=tuple(dofa_indices), norm_mode=norm_mode)
            elif spectral == "terramind":
                self.dofa = TerramindEncoder(
                    indices=tuple(dofa_indices), norm_mode=norm_mode)
            elif spectral == "croma":
                self.dofa = CromaEncoder(
                    indices=tuple(dofa_indices), norm_mode=norm_mode,
                    size=croma_size)
            elif spectral == "scalemae":
                self.dofa = ScaleMaeEncoder(
                    indices=tuple(dofa_indices), norm_mode=norm_mode)
            elif spectral == "ssl4eo_dino":
                self.dofa = SSL4EODinoEncoder(
                    indices=tuple(dofa_indices), norm_mode=norm_mode)
            else:
                self.dofa = DofaEncoder(indices=dofa_indices,
                                        adapter=adapter,
                                        norm_mode=norm_mode,
                                        upsample=dofa_upsample,
                                        lora_rank=lora_rank,
                                        size=dofa_size)
            assert len(self.dofa.indices) == n
        else:
            self.dofa = None

        # ---- disagreement-aware fusion ladder flags (default v0 = off) ----
        fo = dict(fusion_opts or {})
        self.aux_heads_on = bool(fo.get("aux_heads", False))
        self.disagreement_on = bool(fo.get("disagreement", False))
        self.cross_attn_on = bool(fo.get("cross_attn", False))
        self.gated_fusion_on = bool(fo.get("gated_fusion", False))
        self.aux_loss_weight = float(fo.get("aux_loss_weight", 0.3))
        self.disagree_grid = tuple(fo.get("disagree_grid", (32, 32)))
        dual = use_dino and use_dofa
        if not dual:  # ladder only defined for the dual-encoder model
            assert not (self.aux_heads_on or self.cross_attn_on
                        or self.gated_fusion_on), \
                "fusion_opts ladder requires use_dino and use_dofa"
        # dependency chain: gated -> disagreement -> aux_heads
        if self.gated_fusion_on:
            assert self.disagreement_on, "gated_fusion requires disagreement"
        if self.disagreement_on:
            assert self.aux_heads_on, "disagreement requires aux_heads"
        # whether forward must run the aux/disagreement path (also at inference,
        # since gated fusion consumes D)
        self.aux_on = self.aux_heads_on

        if dual:
            if self.aux_heads_on:
                self.aux_head_a = BranchAuxHead(self.dino.dim)
                self.aux_head_b = BranchAuxHead(self.dofa.dim)
            self.cross_attn = (CrossAttnStack(self.dino.dim, self.dofa.dim, n)
                               if self.cross_attn_on else None)
            if self.gated_fusion_on:
                self.fusion = GatedMultiDepthFusion(
                    self.dino.dim, self.dofa.dim, n, fusion_dim,
                    spectral_bias=float(fo.get("spectral_bias", 0.5)))
            else:
                self.fusion = MultiDepthFusion(
                    self.dino.dim, self.dofa.dim, n, fusion_dim)
        else:  # single-branch ablation: project to fusion_dim
            dim_in = self.dino.dim if use_dino else self.dofa.dim
            self.fusion = None
            self.proj = nn.ModuleList([
                nn.Sequential(nn.Conv2d(dim_in, fusion_dim, 1),
                              nn.GroupNorm(8, fusion_dim), nn.GELU())
                for _ in range(n)
            ])
        self.decoder_kind = decoder
        if decoder == "fine":
            assert use_stem, "FineDecoder needs the PPA stem skips"
            self.stem = DetailStem(in_ch=4)
            self.decoder = FineDecoder(fusion_dim, n)
        else:
            assert not use_stem, "upernet/linknet/dpt decoders are stem-free"
            self.stem = None
            self.decoder = {"upernet": UPerNetDecoder,
                            "linknet": LinkNetDecoder,
                            "dpt": DPTDecoder}[decoder](fusion_dim)

        self.swir_head = nn.Sequential(
            nn.Conv2d(fusion_dim, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Tanh(),   # MNDWI in [-1, 1]
        ) if swir_aux_head else None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        extras = {}
        if self.fusion is None:                       # single-branch ablation
            feats = self.dino(x) if self.use_dino else self.dofa(x)
            fused = [p(f) for p, f in zip(self.proj, feats)]
        else:
            feats_a, feats_b = self.dino(x), self.dofa(x)
            D = None
            if self.aux_on:
                # PRE-FUSION aux predictions (constraint: prediction space, and
                # measured before any co-attention / fusion).
                aux_a = self.aux_head_a(feats_a[-1])   # (B,1,32,32)
                aux_b = self.aux_head_b(feats_b[-1])   # (B,1,64,64)
                if self.disagreement_on:
                    D, H = disagreement_map(aux_a, aux_b, self.disagree_grid)
                    # D is a label-free OOD SIGNAL into the gate, not a grad path:
                    # detach so the aux heads are trained ONLY by their own deep
                    # supervision (keeps disagreement emergent + avoids the JSD
                    # log-gradient blow-up that caused NaN at lr 5e-4).
                    D, H = D.detach(), H.detach()
                    extras["D"], extras["H"] = D, H
                hw = x.shape[-2:]
                extras["aux_dino"] = F.interpolate(
                    aux_a, size=hw, mode="bilinear", align_corners=False)
                extras["aux_clay"] = F.interpolate(
                    aux_b, size=hw, mode="bilinear", align_corners=False)
            if self.cross_attn is not None:
                feats_a, feats_b = self.cross_attn(feats_a, feats_b)
            if self.gated_fusion_on:
                fused = self.fusion(feats_a, feats_b, D)
            else:
                fused = self.fusion(feats_a, feats_b)
        if self.decoder_kind == "fine":
            f2, f4 = self.stem(x)
            logits = self.decoder(fused, f2, f4)
        else:
            logits = self.decoder(fused, x.shape[-2:])
        if return_aux:
            extras["swir"] = (self.swir_head(fused[-1])
                              if self.swir_head is not None else None)
            return logits, extras
        return logits
