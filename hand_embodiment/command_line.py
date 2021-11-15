def add_playback_control_arguments(parser):
    """Add arguments for playback control to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--start-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")


def add_configuration_arguments(parser):
    """Add configuration arguments to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--record-mapping-config", type=str, default=None,
        help="Record mapping configuration file.")


def add_animation_arguments(parser):
    """Add arguments to control animation to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds before starting the animation.")
    parser.add_argument(
        "--insole", action="store_true", help="Visualize insole mesh.")
    parser.add_argument(
        "--pillow", action="store_true", help="Visualize pillow.")
    parser.add_argument(
        "--electronic", action="store_true",
        help="Visualize electronic components.")
    parser.add_argument(
        "--passport", action="store_true", help="Visualize open passport.")
