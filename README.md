# deep_learning_video_tools

![tensorflow](https://img.shields.io/badge/tensorflow-v2.2.0-green.svg?style=plastic)

Some tools to modify video content based on well established Deep Learning algorithms for segmentation, inpainting, 3d modelling...

Still work in progress, but some preliminary results available:

**Preliminary results of object removal ([original video](http://img.youtube.com/vi/3FXUw98rrUY/0.jpg))** 

[![IMAGE ALT TEXT](http://img.youtube.com/vi/JlLbS8DpMwE/0.jpg)](http://www.youtube.com/watch?v=JlLbS8DpMwE" "Inpainted video")

## Installation

### Dependencies

```bash
pip .... (TODO)
```

#### generative_inpainting

On the root folder:
```bash
git clone https://github.com/nestormh/neuralgym
```

## About neuralgym

neuralgym repository had to be included in this one to be compatible with the latest version of tensorflow. 
I avoided created a fork since just some parts of the code were upgraded and could be confusing. Original code can be 
checked here: [neuralgym](https://github.com/JiahuiYu/neuralgym).

Changes:
- Some functions have been renamed / reconfigured for the newer tensorflow version.
- Checkpoints provided were incompatible with the actual code, so some variable names had to be changed.

## Models

Models included at the moment are required for image inpainting. You can use these available at:
 
### Segmentation

- Models are directly retrieved from tensorflow repositories.
 
### Image inpainting

Check [generative_inpainting](https://github.com/nestormh/generative_inpainting) repository for more information.

- [Places2](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO?usp=sharing) 
- [CelebA-HQ](https://drive.google.com/drive/folders/1uvcDgMer-4hgWlm6_G9xjvEQGP8neW15?usp=sharing) 

