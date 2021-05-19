# Hand Embodiment

![Overview](doc/source/_static/overview.svg)

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
* Dexterous Hand from Shadow

## Data

Some examples need motion capture data. Ask me about it.

## Examples

![MoCap to MANO](doc/source/_static/mocap_to_mano.png)

Motion capture data fitted to the MANO hand model.

![MANO to Mia](doc/source/_static/mia_embodiment.png)

Transfer of the MANO state to the Mia hand.

## Integrating a New Robotic Hand

Each target hand needs a configuration. These are typically stored in
`hand_embodiment.target_configurations`, but you can define them in your
own python script or module.

A configuration includes information about the kinematic setup of the hand:

* name of finger tip frames
* joints per finger
* base frame of the hand
* pose of base frame with respect to MANO base frame
* path to URDF
* virtual joints (e. g. coupling of joints)

The following scripts can be used to investigate a hand configuration:

* `bin/kinematics_diagram.py` - generates a kinematic diagram (graph) of the
  hand
* `bin/vis_extended_hand_model.py` - visualizes finger tip positions in the
  visual model of the hand, if you have to define additional finger tips this
  is a good tool to verify the result
* `bin/gui_robot_embodiment.py` - with this tool you can (1) find an
  appropriate pose of the hand in MANO's base and (2) interactively verify
  that the embodiment mapping finds appropriate solutions to mimic the MANO
  model
