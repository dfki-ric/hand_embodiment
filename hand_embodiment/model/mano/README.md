# Prepare MANO hand model

1. Download MANO model from [here](https://mano.is.tue.mpg.de/) and unzip it.
2. For the single preparation-step, please install following dependencies. (All dependencies are available via pip and conda)

- chumpy==0.70
- scipy==1.7.3
- pytransform3d

3. Run `python prepare_mano.py`, you will get the converted MANO model that is compatible with this project.
