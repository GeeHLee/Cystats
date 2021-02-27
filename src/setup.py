from setuptools import setup, Extension

add_modules = Extension(name='Cystats', 
                        sources=["models.cpp"], 
                        include_dirs=['/home/liheng/miniconda3/envs/py37/lib/python3.7/site-packages/pybind11/include', 
                                    '/usr/include', 
                                    "/usr/include/eigen3/"])
setup(ext_modules=[add_modules])