# Prepare MANO hand model

1. Download MANO model from [here](https://mano.is.tue.mpg.de/) and unzip it.
2. For the single preparation-step, please install following dependencies. (All dependencies are available via pip and conda)

- chumpy
- scipy

3. Run `python prepare_mano.py`, you will get the converted MANO model that is compatible with this project.
    - **Warning**: The downloaded pickle files of the MANO model uses np.bool, which is removed since numpy 1.24.
Running prepare_mano.py with a numpy version >= 1.24 will result in an error.
    - Please install a lower version of numpy, e.g. **1.23.5** or refer to the numpy version listed in requirements.txt, where the exact versions of the depedencies are listed, in the main folder.
Alternatively, contact one of the maintainer to receive the converted MANO model under the licensing terms of the MANO publishers.
