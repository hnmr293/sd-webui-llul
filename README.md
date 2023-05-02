# LLuL - Local Latent upscaLer

![cover](./images/cover.jpg)

https://user-images.githubusercontent.com/120772120/221390831-9fbccdf8-5898-4515-b988-d6733e8af3f1.mp4

## What is this?

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which lets you to upscale latents locally.

See above image. This is all what this extension does.

## Usage

1. Select `Enabled` checkbox.
2. Move gray box where you want to apply upscaling.
3. Generate image.

## Examples

![sample 2](./images/sample1.jpg)

![sample 3](./images/sample2.jpg)

![sample 4](./images/sample3.jpg)

![sample 5](./images/sample4.jpg)

- Weight 0.00 -> 0.20 Animation

https://user-images.githubusercontent.com/120772120/221390834-7e2c1a1a-d7a6-46b0-8949-83c4d5839c33.mp4

## Mask

The mask is now available. 

Each pixel values of the mask are treated as the scale factors of the interpolation weight. White is 1.0 (enabled), black is 0.0 (disabled) and gray reduces the weights.

![mask sample](./images/mask_effect.jpg)

## How it works

![description of process](./images/desc.png)
