from setuptools import setup, find_packages
setup(
    name='diffusion_plate_optim',
    packages=["diffusion_plate_optim"],
    install_requires=[
    'diffusers==0.11.1',
    'accelerate==0.24.1'
]
)