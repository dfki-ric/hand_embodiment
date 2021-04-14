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

## Idea

The general idea of this software package is to use the MANO hand model of
the mocap library to represent human hand configurations and then transfer
the state of the MANO model to robotic hands. This allows us to quickly
change the motion capture approach because we have an independent
representation of the hand's state. Furthermore, we can easily change
the target system because we just need to implement the mapping from
MANO to the target hand.

The currently implemented motion capture approaches are:
* marker-based motion capture with the Qualisys system

The currently implemented target systems are:
* Mia hand from Prensilia
* Shadow hand (not yet, TODO)

## Data

Some examples need motion capture data. Ask me about it.

## Examples

![MoCap to MANO](doc/source/_static/mocap_to_mano.png)

Motion capture data fitted to the MANO hand model.
