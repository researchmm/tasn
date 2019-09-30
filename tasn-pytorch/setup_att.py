from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='att_grid_generator',
    ext_modules=[
        CUDAExtension('att_grid_generator_cuda', [
            'src/att_grid_generator_cuda.cpp',
            'src/att_grid_generator_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
