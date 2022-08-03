"""Write data streamed from Qualisys MoCap system to file."""
# Install dependency:
# python -m pip install qtm


import asyncio
import numpy as np
import qtm
import json


class OnPacket:
    def __init__(self, verbose=1):
        self.verbose = verbose

        # label order from AIM model
        self.labels = [
            "hand_top",
            "hand_left",
            "hand_right",
            "thumb_tip",
            "thumb_middle",
            "little_tip",
            "little_middle",
            "ring_tip",
            "ring_middle",
            "middle_middle",
            "middle_tip",
            "index_tip",
            "index_middle",
        ]

    def __call__(self, packet):
        """Callback function that is called everytime a data packet arrives from QTM."""
        print("Framenumber: {}".format(packet.framenumber))

        header, markers = packet.get_3d_markers()
        print("Component info: {}".format(header))
        result = {}
        for i, label, marker in zip(range(len(markers)), self.labels, markers):
            if self.verbose:
                print(f"{marker.x:.1f} {marker.y:.1f} {marker.z:.1f} - {label}")
            result[label] = (marker.x / 1000.0, marker.y / 1000.0, marker.z / 1000.0)
        with open("comm.json", "w") as f:
            json.dump(result, f)


async def setup(ip, frequency=None):
    """ Main function """
    connection = await qtm.connect(ip)
    if connection is None:
        return

    if frequency is None:
        frames = "allframes"
    else:
        frames = "frequency:%d" % frequency

    components = ["3d"]
    #'2d', '2dlin', '3d', '3dres', '3dnolabels', '3dnolabelsres', 'analog',
    #'analogsingle', 'force', 'forcesingle', '6d', '6dres', '6deuler',
    #'6deulerres', 'gazevector', 'eyetracker', 'image', 'timecode', 'skeleton',
    #'skeleton:global'

    await connection.stream_frames(
        frames=frames, components=components, on_packet=OnPacket())


if __name__ == "__main__":
    import sys
    ip = sys.argv[-1]
    asyncio.ensure_future(setup(ip, frequency=20))
    asyncio.get_event_loop().run_forever()
