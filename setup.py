#!/usr/bin/env python
import warnings
from setuptools import setup, find_packages


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup_config = dict(
        name="hand_embodiment",
        version="0.2.0",
        maintainer="Alexander Fabisch",
        maintainer_email="alexander.fabisch@dfki.de",
        description="Embodiment mapping for robotic hands from human hand motions",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="None",
        packages=find_packages(),
        package_data={"hand_embodiment": ["model/*"]},
        install_requires=["numpy", "scipy", "pytransform3d", "open3d",
                          "pyyaml", "tqdm", "numba", "pandas"],
        extras_require={
            "test": ["pytest", "pytest-cov"]}
        )

    try:
        from Cython.Build import cythonize
        import numpy
        cython_config = dict(
            ext_modules=cythonize("hand_embodiment/mano_fast.pyx"),
            zip_safe=False,
            compiler_directives={"language_level": "3"},
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-Wno-cpp", "-Wno-unused-function"]
        )
        setup_config.update(cython_config)
    except ImportError:
        warnings.warn("Cython or NumPy is not available. "
                      "Install it if you want the fast MANO model.")

    setup(**setup_config)
