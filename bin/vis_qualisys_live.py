# Install dependency:
# python -m pip install qtm


import asyncio
import numpy as np
import qtm
import pytransform3d.visualizer as pv
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.config import load_mano_config
from mocap.visualization import scatter


class OnPacket:
    def __init__(self):
        self.fig = pv.figure()
        self.window_open = True

        return  # TODO remove
        # TODO define artists
        mano2hand_markers, betas = load_mano_config(
            "examples/config/april_test_mano.yaml")
        self.mbrm = MarkerBasedRecordMapping(
            left=False, mano2hand_markers=mano2hand_markers,
            shape_parameters=betas, verbose=1)
        self.fig.add_geometry(self.mbrm.hand_state_.hand_mesh)
        # TODO measure marker size
        self.markers = scatter(self.fig, np.zeros((7, 3)), s=0.005)

    def __del__(self):
        self.fig.show()

    def __call__(self, packet):
        """Callback function that is called everytime a data packet arrives from QTM."""
        print("Framenumber: {}".format(packet.framenumber))

        timecode = packet.get_time_code()
        print(timecode)

        header, markers = packet.get_3d_markers_no_label()
        print("Component info: {}".format(header))
        for marker in markers:
            print("\t", marker)  # x, y, z, id

        header, markers = packet.get_3d_markers()
        print("Component info: {}".format(header))
        for marker in markers:
            print("\t", marker)  # x, y, z

        return  # TODO remove

        # TODO
        #components = packet.get_skeletons()
        # for the skeleton: https://github.com/qualisys/qualisys_python_sdk/blob/master/qtm/packet.py#L124

        # TODO update all marker positions
        self.markers.set_data(np.zeros((7, 3)))
        for geometry in self.markers.geometries:
            self.fig.update_geometry(geometry)

        # TODO Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.
        hand_markers = [np.zeros(3), np.zeros(3), np.zeros(3)]
        # TODO Positions of markers on fingers.
        finger_markers = {
            "thumb": np.zeros(3),
            "index": np.zeros(3),
            "middle": np.zeros(3),
            "ring": np.zeros(3)
        }
        self.mbrm.estimate(hand_markers, finger_markers)

        # https://github.com/AlexanderFabisch/pytransform3d_examples/blob/master/bin/async_visualizer/async_visualizer.py
        self.fig.update_geometry(self.mbrm.hand_state_.hand_mesh)
        self.window_open = self.fig.visualizer.poll_events()
        if not self.window_open:
            exit(0)  # TODO any better way?
        self.fig.visualizer.update_renderer()

        # TODO embodiment mapping and visualize URDF


async def setup(frequency=None):
    """ Main function """
    connection = await qtm.connect("127.0.0.1")
    if connection is None:
        return

    if frequency is None:
        frames = "allframes"
    else:
        frames = "frequency:%d" % frequency

    components = ["3dnolabels"]
    #'2d', '2dlin', '3d', '3dres', '3dnolabels', '3dnolabelsres', 'analog',
    #'analogsingle', 'force', 'forcesingle', '6d', '6dres', '6deuler',
    #'6deulerres', 'gazevector', 'eyetracker', 'image', 'timecode', 'skeleton',
    #'skeleton:global'

    await connection.stream_frames(
        frames=frames, components=components, on_packet=OnPacket())


if __name__ == "__main__":
    asyncio.ensure_future(setup(frequency=50))
    asyncio.get_event_loop().run_forever()
