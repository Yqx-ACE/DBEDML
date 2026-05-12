import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import numbers
import math
from typing import Optional
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
        
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
    
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias, dilation=1)
        self.win_size = 4
        self.modulator = nn.Parameter(torch.ones(self.win_size, self.win_size, dim*2))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        h1, w1 = h//self.win_size, w//self.win_size
        x = self.project_in(x)
        x = self.dwconv(x)
        x_win = rearrange(x, 'b c (wsh h1) (wsw w1) -> b h1 w1 wsh wsw c', wsh=self.win_size, wsw=self.win_size)
        x_win = x_win * self.modulator
        x = rearrange(x_win, 'b h1 w1 wsh wsw c -> b c (wsh h1) (wsw w1)', wsh=self.win_size, wsw=self.win_size, h1=h1, w1=w1)
        x1, x2 = x.chunk(2, dim=1) 
        x = x1 * x2
        x = self.project_out(x)
        return x


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias, dilation=1)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.project_out(x)
        return x
        
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=103, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.kernel = 3
        self.padding = 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=(self.kernel, self.kernel), padding=(self.padding, self.padding)),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.proj(x)
        return x
        
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        return self.body(x)

class Intra_VSSM(nn.Module):
    def __init__(self, dim, vssm_expansion_factor, bias):
        super(Intra_VSSM, self).__init__()
        hidden = int(dim * vssm_expansion_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.conv_input = nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=bias)
        self.fuse_out = nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=bias)
        self.mamba = Mamba(d_model=hidden // 2)
        
    def forward_core(self, x):
        B, C, H, W = x.size()
        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)

        if H == W:
            feature = torch.cat((feature_h, feature_v), dim=0)
            scan_output = self.mamba(feature)
            scan_output = torch.chunk(scan_output, 2, dim=0)
            scan_output_h = scan_output[0]
            scan_output_v = scan_output[1]
        else:
            scan_output_h = self.mamba(feature_h)
            scan_output_v = self.mamba(feature_v)

        scan_output_h = scan_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
        scan_output_v = scan_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
        scan_output = self.fuse_out(torch.cat((scan_output_h, scan_output_v), dim=1))
        return scan_output

    def forward(self, x):
        x = self.proj_in(x)
        x, x_ = self.dwconv(x).chunk(2, dim=1)
        x = self.forward_core(x)
        x = F.silu(x_) * x
        x = self.proj_out(x)
        return x

class Inter_VSSM(nn.Module):
    def __init__(self, dim, vssm_expansion_factor, bias):
        super(Inter_VSSM, self).__init__()
        hidden = int(dim * vssm_expansion_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_input = nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=bias)
        self.fuse_out = nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=bias)
        self.mamba = Mamba(d_model=hidden // 2)
        self.sigmoid = nn.Sigmoid()

    def forward_core(self, x):
        B, C, H, W = x.size()
        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = x_input[0].permute(0, 2, 1, 3).contiguous()
        feature_h_score = self.avg_pool(feature_h)
        feature_h_score = feature_h_score.view(B, H, -1)
        feature_v = x_input[1].permute(0, 3, 1, 2).contiguous()
        feature_v_score = self.avg_pool(feature_v)
        feature_v_score = feature_v_score.view(B, W, -1)

        if H == W:
            feature_score = torch.cat((feature_h_score, feature_v_score), dim=0)
            scan_score = self.mamba(feature_score)
            scan_score = torch.chunk(scan_score, 2, dim=0)
            scan_score_h = scan_score[0]
            scan_score_v = scan_score[1]
        else:
            scan_score_h = self.mamba(feature_h_score)  
            scan_score_v = self.mamba(feature_v_score)
            
        scan_score_h = self.sigmoid(scan_score_h)
        scan_score_v = self.sigmoid(scan_score_v)
        feature_h = feature_h * scan_score_h[:,:,:,None]
        feature_v = feature_v * scan_score_v[:,:,:,None]
        feature_h = feature_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
        feature_v = feature_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
        output = self.fuse_out(torch.cat((feature_h, feature_v), dim=1))
        return output

    def forward(self, x):
        x = self.proj_in(x)
        x, x_ = self.dwconv(x).chunk(2, dim=1)
        x = self.forward_core(x)
        x = F.silu(x_) * x
        x = self.proj_out(x)
        return x

class Strip_VSSB(nn.Module):
    def __init__(self, dim, vssm_expansion_factor, ffn_expansion_factor, bias=False, ssm=False, LayerNorm_type='WithBias'):
        super(Strip_VSSB, self).__init__()
        self.ssm = ssm
        if self.ssm:
            self.norm1_ssm = LayerNorm(dim, LayerNorm_type)
            self.norm2_ssm = LayerNorm(dim, LayerNorm_type)
            self.intra = Intra_VSSM(dim, vssm_expansion_factor, bias)
            self.inter = Inter_VSSM(dim, vssm_expansion_factor, bias)
        self.norm1_ffn = LayerNorm(dim, LayerNorm_type)
        self.norm2_ffn = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = GDFN(dim, ffn_expansion_factor, bias)
        self.ffn2 = GDFN(dim, ffn_expansion_factor, bias)
    
    def forward(self, x):
        if self.ssm:
            x = x + self.intra(self.norm1_ssm(x))
        #x = x + self.ffn1(self.norm1_ffn(x))
        if self.ssm:
            x = x + self.inter(self.norm2_ssm(x))
        #x = x + self.ffn2(self.norm2_ffn(x))
        
        return x

class CLFF(nn.Module):
    def __init__(self, dim, dim_n1, dim_n2, bias=False):
        super(CLFF, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_n1 = nn.Conv2d(dim_n1, dim, kernel_size=1, bias=bias)
        self.conv_n2 = nn.Conv2d(dim_n2, dim, kernel_size=1, bias=bias)
        self.fuse_out1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, n1, n2):
        x_ = self.conv(x)
        n1_ = self.conv_n1(n1)
        n2_ = self.conv_n2(n2)
        n1_ = F.interpolate(n1_, size=x_.shape[2:], mode='bilinear', align_corners=False)
        n2_ = F.interpolate(n2_, size=x_.shape[2:], mode='bilinear', align_corners=False)
        kl_n1 = F.kl_div(self.log_sigmoid(n1_), self.log_sigmoid(x_), log_target=True)
        kl_n2 = F.kl_div(self.log_sigmoid(n2_), self.log_sigmoid(x_), log_target=True)
        g1 = self.sigmoid(kl_n1)
        g2 = self.sigmoid(kl_n2)
        x = self.fuse_out1(torch.cat((x_, g1 * n1_ + g2 * n2_), dim=1))
        return x
class AttentionPooling(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn_weights = self.attention(x)
        weighted = x * attn_weights
        output = weighted.sum(dim=(2, 3))
        return output

class XYScanNet(nn.Module):
    def __init__(self, 
        params,
        inp_channels=103, 
        out_channels=3, 
        dim=64,  
        num_blocks=[1, 1, 1],
        vssm_expansion_factor=1,
        ffn_expansion_factor=1,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super(XYScanNet, self).__init__()
        inp_channels = params.spectral_size
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            Strip_VSSB(dim=dim, vssm_expansion_factor=vssm_expansion_factor, ffn_expansion_factor=ffn_expansion_factor, bias=bias, ssm=True, LayerNorm_type=LayerNorm_type) 
            for i in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            Strip_VSSB(dim=int(dim*2**1), vssm_expansion_factor=vssm_expansion_factor, ffn_expansion_factor=ffn_expansion_factor, bias=bias, ssm=True, LayerNorm_type=LayerNorm_type) 
            for i in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            Strip_VSSB(dim=int(dim*2**2), vssm_expansion_factor=vssm_expansion_factor, ffn_expansion_factor=ffn_expansion_factor, bias=bias, ssm=True, LayerNorm_type=LayerNorm_type) 
            for i in range(num_blocks[2])
        ])
        self.clff_level1 = CLFF(int(dim*2**0), dim_n1=int(dim*2**1), dim_n2=(dim*2**2), bias=bias)
        self.output = nn.Conv2d(int(dim*2**0), params.spectral_size, kernel_size=3, stride=1, padding=1, bias=bias)
        self.final_fea = nn.Sequential(
            nn.Conv2d(params.spectral_size, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.final_emd = AttentionPooling(dim)
        self.dropout = nn.Dropout(0.1)
        self.upsample = nn.Upsample(
            size=(5, 5),
            mode='bilinear',
            align_corners=False
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1
        )
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1_2 = F.interpolate(out_enc_level1, scale_factor=0.5)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2_1 = F.interpolate(out_enc_level2, scale_factor=2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3_2 = F.interpolate(out_enc_level3, scale_factor=2)
        out_enc_level3_1 = F.interpolate(out_enc_level3_2, scale_factor=2)
        out_enc_level1 = self.clff_level1(out_enc_level1, out_enc_level2_1, out_enc_level3_1)
        out_dec_level1 = self.output(out_enc_level1) + inp_img
        out_final_fea = self.final_fea(out_dec_level1)
        out_final_emd = self.final_emd(out_final_fea)
        return out_final_emd, out_final_fea