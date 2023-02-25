from typing import Union, List

import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import scripts

from scripts.llul_hooker import Hooker, Upscaler, Downscaler
from scripts.llul_xyz import init_xyz

NAME = 'LLuL'

class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.last_hooker: Union[Hooker,None] = None

    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        mode = 'img2img' if is_img2img else 'txt2img'
        id = lambda x: f'{NAME.lower()}-{mode}-{x}'
        
        with gr.Group():
            with gr.Accordion(NAME, open=False):
                enabled = gr.Checkbox(label='Enable', value=False)
                weight = gr.Slider(minimum=-1, maximum=2, value=1, step=0.01, label='Weight')
                
                understand = gr.Checkbox(label='I know what I am doing.', value=False)
                with gr.Column(visible=False) as g:
                    layers = gr.Textbox(label='Layers', value='OUT')
                    apply_to = gr.CheckboxGroup(choices=['Resblock', 'Transformer', 'S. Attn.', 'X. Attn.'], value=['X. Attn.'], label='Apply to')
                    start_steps = gr.Slider(minimum=1, maximum=300, value=5, step=1, label='Start steps')
                    max_steps = gr.Slider(minimum=0, maximum=300, value=0, step=1, label='Max steps')
                    with gr.Row():
                        up = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic'], value='Bilinear', label='Upscaling')
                        up_aa = gr.Checkbox(value=False, label='Enable AA for Upscaling.')
                    with gr.Row():
                        down = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic', 'Area', 'Pooling Max', 'Pooling Avg'], value='Pooling Max', label='Downscaling')
                        down_aa = gr.Checkbox(value=False, label='Enable AA for Downscaling.')
                    intp = gr.Radio(choices=['Lerp', 'SLerp'], value='Lerp', label='interpolation method')
                
                understand.change(
                    lambda b: { g: gr.update(visible=b) },
                    inputs=[understand],
                    outputs=[
                        g  # type: ignore
                    ]
                )
        
        return [
            enabled,
            weight,
            understand,
            layers,
            apply_to,
            start_steps,
            max_steps,
            up,
            up_aa,
            down,
            down_aa,
            intp,
        ]
    
    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        weight: float,
        understand: bool,
        layers: str,
        apply_to: Union[List[str],str],
        start_steps: Union[int,float],
        max_steps: Union[int,float],
        up: str,
        up_aa: bool,
        down: str,
        down_aa: bool,
        intp: str,
    ):
        if self.last_hooker is not None:
            self.last_hooker.__exit__(None, None, None)
            self.last_hooker = None
        
        if not enabled:
            return
        
        if p.width < 128 or p.height < 127:
            raise ValueError(f'too small to LLuL: {p.width}x{p.height}; expected >=128x128')
        
        weight = float(weight)
        
        if understand:
            lays = (
                None if len(layers) == 0 else
                [x.strip() for x in layers.split(',')]
            )
            if isinstance(apply_to, str):
                apply_to = [x.strip() for x in apply_to.split(',')]
            apply_to = [x.lower() for x in apply_to]
            start_steps = max(1, int(start_steps))
            max_steps = max(1, [p.steps, int(max_steps)][1 <= max_steps])
            up_fn = Upscaler(up, up_aa)
            down_fn = Downscaler(down, down_aa)
            intp = intp.lower()
        else:
            lays = ['OUT']
            apply_to = ['s. attn.', 'x. attn.']
            start_steps = 5
            max_steps = p.steps
            up_fn = Upscaler('bilinear', aa=False)
            down_fn = Downscaler('pooling max', aa=False)
            intp = 'lerp'
        
        self.last_hooker = Hooker(
            enabled=True,
            weight=weight,
            layers=lays,
            apply_to=apply_to,
            start_steps=start_steps,
            max_steps=max_steps,
            up_fn=up_fn,
            down_fn=down_fn,
            intp=intp,
        )
        
        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'{NAME} Enabled': enabled,
            f'{NAME} Weight': weight,
            f'{NAME} Layers': lays,
            f'{NAME} Apply to': apply_to,
            f'{NAME} Start steps': start_steps,
            f'{NAME} Max steps': max_steps,
            f'{NAME} Upscaler': up_fn.name,
            f'{NAME} Downscaler': down_fn.name,
            f'{NAME} Interpolation': intp,
        })

init_xyz(Script)
