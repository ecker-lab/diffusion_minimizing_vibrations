# Minimizing Structural Vibrations via Guided Diffusion Design Optimization

This repository accompanies our paper presented at the ICLR 2024 Workshop on AI4DifferentialEquations in Science (Link will follow soon) and enables to generate novel plate designs to minimize vibration-energy based on deep learning.

Structural vibrations are a source of unwanted noise in everyday products like cars, trains or airplanes. For example, the motor of a car causes the chassis to vibrate, which radiates sound into the interior of the car and is eventually perceived by the passenger as noise. Because of this, engineers try to minimize the amount of vibration to reduce noise. This work introduces a method for reducing vibrations by optimally placing beadings (indentations) in plate-like structures with a guided diffusion design optimization approach.
Our approach integrates a diffusion model for realistic design generation and the gradient information from a surrogate model trained to predict the vibration patterns of a design to guide the design towards low-vibration energy. Results demonstrate that our method generates plates with lower vibration energy than any sample within the training dataset. To enhance broader applicability, further development is needed in incorporating constraints in the outcome plate design.


## Code

In this repository, we provide code to generate novel plate designs to minimize vibration-energy.
For a quick demo of the capabilities of our method without any manual setup, we prepared a notebook:

<a href="https://colab.research.google.com/github/JanvDelden/diffusion_minimizing_vibrations/blob/main/Example_notebook_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Here, you are able to specify the loss function used to guide the denoising process. This enables designing plates to have low vibration energy in any frequency range. Please note, that the regression model used in this notebook has been trained on slightly more data than the one used to generate results in the paper, leading to slight differences in the generations.

## API

This codebase builds upon the https://github.com/ecker-lab/Learning_Vibrating_Plates repo to train surrogate regression models for diffusion guidance and requires the repo and environment to be set up.
In addition, we implement diffusion model training and guided diffusion plate design optimization in this repository. To actually evaluate generated plate designs, additional numerical solvers are required, that can not be provided in this codebase.

To install this repository, make sure the environment defined in the Learning Vibrating Plates repository is installed and then call from within this repository:


``
pip install .
``


To train the diffusion model:

``
python diffusion_plate_optim/train_diffusion.py --save_dir /path/to/save_directory
python diffusion_plate_optim/train_diffusion.py --save_dir /home/nimdelde/scratch/experiments/generative_plates/debug
``

To generate new plate designs:

``
python
diffusion_plate_optim/generation_call.py --dir /path/to/save_directory \
--diffusion_path /path/to/diffusion_model \
--regression_path /path/to/guidance_regression_model \
--batch_size 4 \
--do_range_loss
``

## Citation
```
@inproceedings{delden2024minimizing,
  title={Minimizing Structural Vibrations via Guided Diffusion Design Optimization},
  author={van Delden, Jan and Schultz, Julius and Blech, Christopher and Langer, Sabine C and L{\"u}ddecke, Timo},
  booktitle={ICLR 2024 Workshop on AI4DifferentialEquations In Science},
  url={https://openreview.net/forum?id=z4dcQodnoo}
  year={2024}
}
```
