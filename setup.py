#!/usr/bin/env python
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
                          "pyyaml", "tqdm", "jax", "jaxtransform3d",
                          "pandas"],
        extras_require={
            "test": ["pytest", "pytest-cov"],
            "doc": ["sphinx"]}
        )

    setup(**setup_config)
