import math
from typing import Union, Callable, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from torch import nn, Tensor
from einops import rearrange
from PIL import Image
from modules.processing import StableDiffusionProcessing, slerp as Slerp

from scripts.sdhook import (
    SDHook,
    each_unet_attn_layers,
    each_unet_transformers,
    each_unet_resblock
)


class Upscaler:
    
    def __init__(self, mode: str, aa: bool):
        mode = {
            'nearest': 'nearest-exact',
            'bilinear': 'bilinear',
            'bicubic': 'bicubic',
        }.get(mode.lower(), mode)
        self.mode = mode
        self.aa = bool(aa)
    
    @property
    def name(self):
        s = self.mode
        if self.aa: s += '-aa'
        return s
    
    def __call__(self, x: Tensor, scale: float = 2.0):
        return F.interpolate(x, scale_factor=scale, mode=self.mode, antialias=self.aa)


class Downscaler:
    
    def __init__(self, mode: str, aa: bool):
        self._name = mode.lower()
        intp, mode = {
            'nearest': (F.interpolate, 'nearest-exact'),
            'bilinear': (F.interpolate, 'bilinear'),
            'bicubic': (F.interpolate, 'bicubic'),
            'area': (F.interpolate, 'area'),
            'pooling max': (F.max_pool2d, ''),
            'pooling avg': (F.avg_pool2d, ''),
        }[mode.lower()]
        self.intp = intp
        self.mode = mode
        self.aa = bool(aa)
    
    @property
    def name(self):
        s = self._name
        if self.aa: s += '-aa'
        return s
    
    def __call__(self, x: Tensor, scale: float = 2.0):
        if scale <= 1:
            scale = float(scale)
            scale_inv = 1 / scale
        else:
            scale_inv = float(scale)
            scale = 1 / scale_inv
        assert scale <= 1
        assert 1 <= scale_inv
        
        kwargs = {}
        if len(self.mode) != 0:
            kwargs['scale_factor'] = scale
            kwargs['mode'] = self.mode
            kwargs['antialias'] = self.aa
        else:
            kwargs['kernel_size'] = int(scale_inv)
        return self.intp(x, **kwargs)


def lerp(v0, v1, t):
    return torch.lerp(v0, v1, t)

def slerp(v0, v1, t):
    v = Slerp(t, v0, v1)
    if torch.any(torch.isnan(v)).item():
        v = lerp(v0, v1, t)
    return v

class Hooker(SDHook):
    
    def __init__(
        self,
        enabled: bool,
        multiply: int,
        weight: float,
        layers: Union[list,None],
        apply_to: List[str],
        start_steps: int,
        max_steps: int,
        up_fn: Callable[[Tensor,float], Tensor],
        down_fn: Callable[[Tensor,float], Tensor],
        intp: str,
        x: float,
        y: float,
        force_float: bool,
        mask_image: Union[Image.Image,None],
    ):
        super().__init__(enabled)
        self.multiply = int(multiply)
        self.weight = float(weight)
        self.layers = layers
        self.apply_to = apply_to
        self.start_steps = int(start_steps)
        self.max_steps = int(max_steps)
        self.up = up_fn
        self.down = down_fn
        self.x0 = x
        self.y0 = y
        self.force_float = force_float
        self.mask_image = mask_image
        
        if intp == 'lerp':
            self.intp = lerp
        elif intp == 'slerp':
            self.intp = slerp
        else:
            raise ValueError(f'invalid interpolation method: {intp}')
        
        if not (1 <= self.multiply and (self.multiply & (self.multiply - 1) == 0)):
            raise ValueError(f'multiplier must be power of 2, but not: {self.multiply}')
        
        if mask_image is not None:
            if mask_image.mode != 'L':
                raise ValueError(f'the mode of mask image is: {mask_image.mode}')
    
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        step = 0
        
        def hook_step_pre(*args, **kwargs):
            nonlocal step
            step += 1
        
        self.hook_layer_pre(unet, hook_step_pre)
        
        start_step = self.start_steps
        max_steps = self.max_steps
        M = self.multiply
        
        def create_pre_hook(name: str, ctx: dict):
            def pre_hook(module: nn.Module, inputs: list):
                ctx['skipped'] = True
                
                if step < start_step or max_steps < step:
                    return
                
                x, *rest = inputs
                dim = x.dim()
                if dim == 3:
                    # attension
                    bi, ni, chi = x.shape
                    wi, hi, Ni = self.get_size(p, ni)
                    x = rearrange(x, 'b (h w) c -> b c h w', w=wi)
                    if len(rest) != 0:
                        # x. attn.
                        rest[0] = torch.concat((rest[0], rest[0]), dim=0)
                elif dim == 4:
                    # resblock, transformer
                    bi, chi, hi, wi = x.shape
                    if 0 < len(rest):
                        t_emb = rest[0] # t_emb (for resblock) or context (for transformer)
                        rest[0] = torch.concat((t_emb, t_emb), dim=0)
                    else:
                        # `out` layer
                        pass
                else:
                    return
                
                # extract
                w, h = wi // M, hi // M
                if w == 0 or h == 0:
                    # input latent is too small to apply
                    return
                
                s0, t0 = int(wi * self.x0), int(hi * self.y0)
                s1, t1 = s0 + w, t0 + h
                if wi < s1:
                    s1 = wi
                    s0 = s1 - w
                if hi < t1:
                    t1 = hi
                    t0 = t1 - h
                
                if s0 < 0 or t0 < 0:
                    raise ValueError(f'LLuL failed to process: s=({s0},{s1}), t=({t0},{t1})')
                
                x1 = x[:, :, t0:t1, s0:s1]
                
                # upscaling
                x1 = self.up(x1, M)
                if x1.shape[-1] < x.shape[-1] or x1.shape[-2] < x.shape[-2]:
                    dx = x.shape[-1] - x1.shape[-1]
                    dx1 = dx // 2
                    dx2 = dx - dx1
                    dy = x.shape[-2] - x1.shape[-2]
                    dy1 = dy // 2
                    dy2 = dy - dy1
                    x1 = F.pad(x1, (dx1, dx2, dy1, dy2), 'replicate')
                
                x = torch.concat((x, x1), dim=0)
                if dim == 3:
                    x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
                
                #print('I', tuple(inputs[0].shape), tuple(x.shape))
                ctx['skipped'] = False
                return x, *rest
            return pre_hook
        
        def create_post_hook(name: str, ctx: dict):
            def post_hook(module: nn.Module, inputs: list, output: Tensor):
                if step < start_step or max_steps < step:
                    return
                
                if ctx['skipped']:
                    return
                
                x = output
                dim = x.dim()
                if dim == 3:
                    bo, no, cho = x.shape
                    wo, ho, No = self.get_size(p, no)
                    x = rearrange(x, 'b (h w) c -> b c h w', w=wo)
                elif dim == 4:
                    bo, cho, ho, wo = x.shape
                else:
                    return
                
                assert bo % 2 == 0
                x, x1 = x[:bo//2], x[bo//2:]
                
                # downscaling
                x1 = self.down(x1, M)
                
                # embed
                w, h = x1.shape[-1], x1.shape[-2]
                s0, t0 = int(wo * self.x0), int(ho * self.y0)
                s1, t1 = s0 + w, t0 + h
                if wo < s1:
                    s1 = wo
                    s0 = s1 - w
                if ho < t1:
                    t1 = ho
                    t0 = t1 - h
                
                if s0 < 0 or t0 < 0:
                    raise ValueError(f'LLuL failed to process: s=({s0},{s1}), t=({t0},{t1})')
                
                x[:, :, t0:t1, s0:s1] = self.interpolate(x[:, :, t0:t1, s0:s1], x1, self.weight)
                
                if dim == 3:
                    x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
                
                #print('O', tuple(inputs[0].shape), tuple(x.shape))
                return x
            return post_hook
        
        def create_hook(name: str, **kwargs):
            ctx = dict()
            ctx.update(kwargs)
            return (
                create_pre_hook(name, ctx),
                create_post_hook(name, ctx)
            )
        
        def wrap_for_xattn(pre, post):
            def f(module: nn.Module, o: Callable, *args, **kwargs):
                inputs = list(args) + list(kwargs.values())
                inputs_ = pre(module, inputs)
                if inputs_ is not None:
                    inputs = inputs_
                output = o(*inputs)
                output_ = post(module, inputs, output)
                if output_ is not None:
                    output = output_
                return output
            return f
        
        # 
        # process each attention layers
        # 
        for name, attn in each_unet_attn_layers(unet):
            if self.layers is not None:
                if not any(layer in name for layer in self.layers):
                    continue
            
            q_in = attn.to_q.in_features
            k_in = attn.to_k.in_features
            if q_in == k_in:
                # self-attention
                if 's. attn.' in self.apply_to:
                    pre, post = create_hook(name)
                    self.hook_layer_pre(attn, pre)
                    self.hook_layer(attn, post)
            else:
                # cross-attention
                if 'x. attn.' in self.apply_to:
                    pre, post = create_hook(name)
                    self.hook_forward(attn, wrap_for_xattn(pre, post))
        
        # 
        # process Resblocks
        # 
        for name, res in each_unet_resblock(unet):
            if 'resblock' not in self.apply_to:
                continue
            
            if self.layers is not None:
                if not any(layer in name for layer in self.layers):
                    continue
            
            pre, post = create_hook(name)
            self.hook_layer_pre(res, pre)
            self.hook_layer(res, post)
        
        # 
        # process Transformers (including s/x-attn)
        # 
        for name, res in each_unet_transformers(unet):
            if 'transformer' not in self.apply_to:
                continue
            
            if self.layers is not None:
                if not any(layer in name for layer in self.layers):
                    continue
            
            pre, post = create_hook(name)
            self.hook_layer_pre(res, pre)
            self.hook_layer(res, post)
        
        # 
        # process OUT
        # 
        if 'out' in self.apply_to:
            out = unet.out
            pre, post = create_hook('out')
            self.hook_layer_pre(out, pre)
            self.hook_layer(out, post)
    
    def get_size(self, p: StableDiffusionProcessing, n: int):
        # n := p.width / N * p.height / N
        wh = p.width * p.height
        N2 = wh // n
        N = int(math.sqrt(N2))
        assert N*N == N2, f'N={N}, N2={N2}'
        assert p.width % N == 0, f'width={p.width}, N={N}'
        assert p.height % N == 0, f'height={p.height}, N={N}'
        w, h = p.width // N, p.height // N
        assert w * h == n, f'w={w}, h={h}, N={N}, n={n}'
        return w, h, N
    
    def interpolate(self, v1: Tensor, v2: Tensor, t: float):
        dtype = v1.dtype
        if self.force_float:
            v1 = v1.float()
            v2 = v2.float()
        
        if self.mask_image is None:
            v = self.intp(v1, v2, t)
        else:
            to_w, to_h = v1.shape[-1], v1.shape[-2]
            resized_image = self.mask_image.resize((to_w, to_h), Image.BILINEAR)
            mask = torchvision.transforms.functional.to_tensor(resized_image).to(device=v1.device, dtype=v1.dtype)
            mask.unsqueeze_(0) # (C,H,W) -> (B,C,H,W)
            mask.mul_(t)
            v = self.intp(v1, v2, mask)
        
        if self.force_float:
            v = v.to(dtype)
        
        return v
