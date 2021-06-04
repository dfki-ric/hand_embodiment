# Install dependency:
# python -m pip install qtm


import asyncio
import qtm
import pytransform3d.visualizer as pv


class OnPacket:
    def __init__(self):
        self.fig = pv.figure()
        self.window_open = True
        # TODO define artists

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

        # TODO
        #components = packet.get_skeletons()
        # for the skeleton: https://github.com/qualisys/qualisys_python_sdk/blob/master/qtm/packet.py#L124

        # https://github.com/AlexanderFabisch/pytransform3d_examples/blob/master/bin/async_visualizer/async_visualizer.py
        # TODO define meshes to update
        #drawn_artists = []
        #for a in drawn_artists:
        #    for geometry in a.geometries:
        #        self.fig.update_geometry(geometry)
        #self.window_open = self.fig.visualizer.poll_events()
        #if not self.window_open:
        #    exit(0)  # TODO any better way?
        #self.fig.visualizer.update_renderer()


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
