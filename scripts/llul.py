import tempfile
from typing import Union, List, Callable

from PIL import Image, ImageDraw, ImageChops
import gradio as gr

from modules.processing import StableDiffusionProcessing, Processed
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
        js = lambda s: f'globalThis["{id(s)}"]'
        
        with gr.Group():
            with gr.Accordion(NAME, open=False, elem_id=id('accordion')):
                enabled = gr.Checkbox(label='Enabled', value=False)
                with gr.Row():
                    weight = gr.Slider(minimum=-1, maximum=2, value=0.15, step=0.01, label='Weight')
                    multiply = gr.Slider(value=1, minimum=1, maximum=5, step=1, label='Multiplication (2^N)', elem_id=id('m'))
                gr.HTML(elem_id=id('container'))
                add_area_image = gr.Checkbox(value=True, label='Add the effective area to output images.')
                
                with gr.Row():
                    use_mask = gr.Checkbox(value=False, label='Enable mask which scales the weight (black = 0.0, white = 1.0)')
                    mask = gr.File(interactive=True, label='Upload mask image', elem_id=id('mask'))
                
                force_float = gr.Checkbox(label='Force convert half to float on interpolation (for some platforms)', value=False)
                understand = gr.Checkbox(label='I know what I am doing.', value=False)
                with gr.Column(visible=False) as g:
                    layers = gr.Textbox(label='Layers', value='OUT')
                    apply_to = gr.CheckboxGroup(choices=['Resblock', 'Transformer', 'S. Attn.', 'X. Attn.', 'OUT'], value=['OUT'], label='Apply to')
                    start_steps = gr.Slider(minimum=1, maximum=300, value=5, step=1, label='Start steps')
                    max_steps = gr.Slider(minimum=0, maximum=300, value=0, step=1, label='Max steps')
                    with gr.Row():
                        up = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic'], value='Bilinear', label='Upscaling')
                        up_aa = gr.Checkbox(value=False, label='Enable AA for Upscaling.')
                    with gr.Row():
                        down = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic', 'Area', 'Pooling Max', 'Pooling Avg'], value='Bilinear', label='Downscaling')
                        down_aa = gr.Checkbox(value=False, label='Enable AA for Downscaling.')
                    intp = gr.Radio(choices=['Lerp', 'SLerp'], value='Lerp', label='interpolation method')
                
                understand.change(
                    lambda b: { g: gr.update(visible=b) },
                    inputs=[understand],
                    outputs=[
                        g  # type: ignore
                    ]
                )
        
                with gr.Row(visible=False):
                    sink = gr.HTML(value='') # to suppress error in javascript
                    x = js2py('x', id, js, sink)
                    y = js2py('y', id, js, sink)
                
        return [
            enabled,
            multiply,
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
            x,
            y,
            force_float,
            use_mask,
            mask,
            add_area_image,
        ]
    
    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        multiply: Union[int,float],
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
        x: Union[str,None] = None,
        y: Union[str,None] = None,
        force_float = False,
        use_mask: bool = False,
        mask: Union[tempfile._TemporaryFileWrapper,None] = None,
        add_area_image: bool = True, # for postprocess
    ):
        if self.last_hooker is not None:
            self.last_hooker.__exit__(None, None, None)
            self.last_hooker = None
        
        if not enabled:
            return
        
        if p.width < 128 or p.height < 128:
            raise ValueError(f'Image size is too small to LLuL: {p.width}x{p.height}; expected >=128x128.')
        
        multiply = 2 ** int(max(multiply, 0))
        weight = float(weight)
        if x is None or len(x) == 0:
            x = str((p.width - p.width // multiply) // 2)
        if y is None or len(y) == 0:
            y = str((p.height - p.height // multiply) // 2)
        
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
            apply_to = ['out']
            start_steps = 5
            max_steps = int(p.steps)
            up_fn = Upscaler('bilinear', aa=False)
            down_fn = Downscaler('bilinear', aa=False)
            intp = 'lerp'
        
        xf = float(x)
        yf = float(y)
        
        mask_image = None
        if use_mask and mask is not None:
            # Can I read from passed tempfile._TemporaryFileWrapper???
            mask_image = Image.open(mask.name).convert('L')
            intp = 'lerp'
        
        self.last_hooker = Hooker(
            enabled=True,
            multiply=int(multiply),
            weight=weight,
            layers=lays,
            apply_to=apply_to,
            start_steps=start_steps,
            max_steps=max_steps,
            up_fn=up_fn,
            down_fn=down_fn,
            intp=intp,
            x=xf/p.width,
            y=yf/p.height,
            force_float=force_float,
            mask_image=mask_image,
        )
        
        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'{NAME} Enabled': enabled,
            f'{NAME} Multiply': multiply,
            f'{NAME} Weight': weight,
            f'{NAME} Layers': lays,
            f'{NAME} Apply to': apply_to,
            f'{NAME} Start steps': start_steps,
            f'{NAME} Max steps': max_steps,
            f'{NAME} Upscaler': up_fn.name,
            f'{NAME} Downscaler': down_fn.name,
            f'{NAME} Interpolation': intp,
            f'{NAME} x': x,
            f'{NAME} y': y,
        })
    
    def postprocess(
        self,
        p: StableDiffusionProcessing,
        proc: Processed,
        enabled: bool,
        multiply: Union[int,float],
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
        x: Union[str,None] = None,
        y: Union[str,None] = None,
        force_float = False,
        use_mask: bool = False,
        mask: Union[tempfile._TemporaryFileWrapper,None] = None,
        add_area_image: bool = True,
    ):
        if not enabled:
            return
        
        multiply = 2 ** int(max(multiply, 0))
        if x is None or len(x) == 0:
            x = str((p.width - p.width // multiply) // 2)
        if y is None or len(y) == 0:
            y = str((p.height - p.height // multiply) // 2)
        
        xi0 = int(x)
        yi0 = int(y)
        xi1 = xi0 + p.width // multiply
        yi1 = yi0 + p.height // multiply
        area = Image.new(mode='RGB', size=(p.width, p.height), color=(128,128,128))
        draw = ImageDraw.Draw(area)
        draw.rectangle((xi0, yi0, xi1, yi1), fill=(255,255,255))
        
        for image_index in range(len(proc.images)):
            is_grid = image_index < proc.index_of_first_image
            if is_grid:
                continue
            
            image = proc.images[image_index]
            area_image = ImageChops.multiply(image, area)
            
            i = image_index - proc.index_of_first_image
            proc.images.append(area_image)
            proc.all_prompts.append(proc.all_prompts[i])
            proc.all_negative_prompts.append(proc.all_negative_prompts[i])
            proc.all_seeds.append(proc.all_seeds[i])
            proc.all_subseeds.append(proc.all_subseeds[i])
            proc.infotexts.append(proc.infotexts[image_index])
        

def js2py(
    name: str,
    id: Callable[[str], str],
    js: Callable[[str], str],
    sink: gr.components.IOComponent,
):
    v_set = gr.Button(elem_id=id(f'{name}_set'))
    v = gr.Textbox(elem_id=id(name))
    v_sink = gr.Textbox()
    v_set.click(fn=None, _js=js(name), outputs=[v, v_sink])
    v_sink.change(fn=None, _js=js(f'{name}_after'), outputs=[sink])    
    return v


init_xyz(Script)
