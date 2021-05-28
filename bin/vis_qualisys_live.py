# Install dependency:
# python -m pip install qtm


import asyncio
import qtm


def on_packet(packet):
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

    # TODO update visualizer


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
        frames=frames, components=components, on_packet=on_packet)


if __name__ == "__main__":
    asyncio.ensure_future(setup(frequency=50))
    asyncio.get_event_loop().run_forever()
