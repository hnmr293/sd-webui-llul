import math
from typing import Union, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
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
    
    def __call__(self, x: Tensor):
        return F.interpolate(x, scale_factor=2, mode=self.mode, antialias=self.aa)


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
    
    def __call__(self, x: Tensor):
        kwargs = {}
        if len(self.mode) != 0:
            kwargs['scale_factor'] = 0.5
            kwargs['mode'] = self.mode
            kwargs['antialias'] = self.aa
        else:
            kwargs['kernel_size'] = 2
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
        weight: float,
        layers: Union[list,None],
        apply_to: List[str],
        start_steps: int,
        max_steps: int,
        up_fn: Callable[[Tensor], Tensor],
        down_fn: Callable[[Tensor], Tensor],
        intp: str,
        x: float,
        y: float,
    ):
        super().__init__(enabled)
        self.weight = float(weight)
        self.layers = layers
        self.apply_to = apply_to
        self.start_steps = int(start_steps)
        self.max_steps = int(max_steps)
        self.up = up_fn
        self.down = down_fn
        self.x0 = x
        self.y0 = y
        
        if intp == 'lerp':
            self.intp = lerp
        elif intp == 'slerp':
            self.intp = slerp
        else:
            raise ValueError(f'invalid interpolation method: {intp}')
    
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        step = 0
        
        def hook_step_pre(*args, **kwargs):
            nonlocal step
            step += 1
        
        self.hook_layer_pre(unet, hook_step_pre)
        
        start_step = self.start_steps
        max_steps = self.max_steps
        
        def create_pre_hook(name: str):
            def pre_hook(module: nn.Module, inputs: list):
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
                    t_emb = rest[0] # t_emb (for resblock) or context (for transformer)
                    #print(name, t_emb.shape)
                    rest[0] = torch.concat((t_emb, t_emb), dim=0)
                else:
                    return
                
                # extract
                w, h = wi // 2, hi // 2
                assert w % 2 == 0
                assert h % 2 == 0
                mw, mh = int(wi * self.x0), int(hi * self.y0)
                x1 = x[:, :, mh:mh+h, mw:mw+w]
                
                # upscaling
                x1 = self.up(x1)
                
                x = torch.concat((x, x1), dim=0)
                if dim == 3:
                    x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
                
                #print('I', tuple(inputs[0].shape), tuple(x.shape))
                return x, *rest
            return pre_hook
        
        def create_post_hook(name: str):
            def post_hook(module: nn.Module, inputs: list, output: Tensor):
                if step < start_step or max_steps < step:
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
                x1 = self.down(x1)
                
                # embed
                assert wo % 2 == 0, ho % 2 == 0
                w, h = wo // 2, ho // 2
                assert w % 2 == 0, h % 2 == 0
                
                mw, mh = int(wo * self.x0), int(ho * self.y0)
                x[:, :, mh:mh+h, mw:mw+w] = self.intp(x[:, :, mh:mh+h, mw:mw+w], x1, self.weight)
                
                if dim == 3:
                    x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
                
                #print('O', tuple(inputs[0].shape), tuple(x.shape))
                return x
            return post_hook
        
        def create_hook(name: str):
            return (
                create_pre_hook(name),
                create_post_hook(name)
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
