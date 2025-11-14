from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    # This ensures wheel is marked as "non-pure" (has binary files)
    def has_ext_modules(self):
        return True

setup(
    name="plugin_trt_ep",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # include MANIFEST.in contents
    package_data={
        "plugin_trt_ep": ["libs/*.dll"],  # include DLLs inside the wheel
    },
    distclass=BinaryDistribution,
    description="Example package including DLLs",
    author="ORT",
    python_requires=">=3.8",
)
