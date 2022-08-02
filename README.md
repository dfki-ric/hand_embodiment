# Hand Embodiment

![Overview](doc/source/_static/overview.png)

Embodiment mapping for robotic hands from human hand motions

## Idea

The general idea of this software package is to use the
[MANO hand model](https://mano.is.tue.mpg.de/) to represent human
hand configurations and then transfer the state of the MANO
model to robotic hands. This allows us to quickly change the motion capture
approach because we have an independent representation of the hand's state.
Furthermore, we can easily change the target system because we just need to
configure the mapping from MANO to the target hand.

The currently implemented motion capture approaches are:
* marker-based motion capture with the Qualisys system

The currently implemented target systems are:
* Mia hand from Prensilia
* Dexterous Hand from Shadow
* Robotiq 2F-140 gripper
* BarrettHand

## Paper

This is the implementation of the paper

Alexander Fabisch, Manuela Uliano, Dennis Marschner, Melvin Laux,
Johannes Brust, Marco Controzzi:
**A Modular Approach to the Embodiment of Hand Motions from Human Demonstrations**,
https://arxiv.org/abs/2203.02778

## Examples

Scripts are located at [bin/](bin/).

### Motion Capture Data Fitted to MANO

See [bin/vis_markers_to_mano_trajectory.py](bin/vis_markers_to_mano_trajectory.py)

<table>
<tr>
<td><img src="doc/source/_static/figure_record_1.png" height="200px" /></td>
<td><img src="doc/source/_static/figure_record_2.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/figure_record_3.png" height="200px" /></td>
<td><img src="doc/source/_static/figure_record_4.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/figure_record_5.png" height="200px" /></td>
<td><img src="doc/source/_static/figure_record_6.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/figure_record_7.png" height="200px" /></td>
<td><img src="doc/source/_static/figure_record_8.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/figure_record_9.png" height="200px" /></td>
<td><img src="doc/source/_static/figure_record_10.png" height="200px" /></td>
</tr>
</table>

### Interactive Transfer of MANO State to Robotic Hands

See [bin/gui_robot_embodiment.py](bin/gui_robot_embodiment.py)

<table>
<tr>
<td><img src="doc/source/_static/embodiment_interactive_mia_1.png" height="200px" /></td>
<td><img src="doc/source/_static/embodiment_interactive_mia_2.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/embodiment_interactive_mia_3.png" height="200px" /></td>
<td><img src="doc/source/_static/embodiment_interactive_shadow_1.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/embodiment_interactive_shadow_2.png" height="200px" /></td>
<td><img src="doc/source/_static/embodiment_interactive_shadow_3.png" height="200px" /></td>
</tr>
<tr>
<td><img src="doc/source/_static/embodiment_interactive_shadow_4.png" height="200px" /></td>
<td><img src="doc/source/_static/embodiment_interactive_shadow_5.png" height="200px" /></td>
</tr>
</table>

### Example Script with Test Data

You can run an example with test data from the main directory with

```bash
python bin/vis_markers_to_robot.py shadow --demo-file test/data/recording.tsv --mocap-config examples/config/markers/20210826_april.yaml --mano-config examples/config/mano/20210610_april.yaml --record-mapping-config examples/config/record_mapping/20211105_april.yaml --pillow --show-mano
```

or

```bash
python bin/vis_markers_to_robot.py mia --demo-file test/data/recording.tsv --mocap-config examples/config/markers/20210826_april.yaml --mano-config examples/config/mano/20210610_april.yaml --record-mapping-config examples/config/record_mapping/20211105_april.yaml --pillow --mia-thumb-adducted --show-mano
```

## Merge Policy

If you want to contribute to this repository, open a pull request on GitHub
with your suggested changes. Pushing to the main branch is forbidden. Note
that there might be no active maintenance and no support since this software
is only intended as an implementation supporting the corresponding paper.

## License

While this source code is released under BSD-3-clause license, it also contains
models of robotic hands that have been released under different licenses:

* Mia hand: [BSD-3-clause license](hand_embodiment/model/mia_hand_ros_pkgs/mia_hand_description/LICENSE)
* Shadow dexterous hand: [GPL v2.0](hand_embodiment/model/sr_common/LICENSE)
* Robotiq 2F-140: [BSD-2-clause license](hand_embodiment/model/robotiq_2f_140_gripper_visualization/LICENSE)

## Funding

This library has been developed initially at the
[Robotics Innovation Center](https://robotik.dfki-bremen.de/en/startpage.html)
of the German Research Center for Artificial Intelligence (DFKI GmbH) in
Bremen. At this phase the work was supported through a grant from the European
Commission (870142).

<img src="doc/source/_static/DFKI_Logo.jpg" height="100px" />
