# Hand Embodiment

Embodiment mapping for robotic hands from human hand motions

## Installation

Install [mocap](https://git.hb.dfki.de/dfki-interaction/mocap) and then
install this library:

```bash
# note that you need access rights to the Mia URDF:
# https://github.com/aprilprojecteu/mia_hand_description
git clone git@git.hb.dfki.de:dfki-interaction/experimental/hand_embodiment.git --recursive
cd hand_embodiment
pip install -e .
```

## Examples

![MoCap to MANO](doc/source/_static/mocap_to_mano.png)

Motion capture data fitted to the MANO hand model.
