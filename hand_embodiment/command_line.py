"""Common options for command line scripts."""
from .target_configurations import TARGET_CONFIG
from .vis_utils import ARTISTS
from .mocap_objects import MOCAP_OBJECTS


def add_hand_argument(parser):
    """Add argument for target hand selection to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "hand", choices=TARGET_CONFIG.keys(),
        help=f"Name of the robotic hand (target system).")


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
    parser.add_argument(
        "--robot-config", type=str, default=None,
        help="Target system configuration file.")


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
        "--show-expected-markers", action="store_true",
        help="Show expected markers at MANO mesh.")
    add_object_visualization_arguments(parser)


def add_object_visualization_arguments(parser):
    """Add arguments to show objects to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--insole", action="store_true", help="Visualize insole mesh.")
    parser.add_argument(
        "--pillow", action="store_true", help="Visualize small pillow.")
    parser.add_argument(
        "--pillow-big", action="store_true", help="Visualize big pillow.")
    parser.add_argument(
        "--osai-case", action="store_true", help="Visualize OSAI case.")
    parser.add_argument(
        "--electronic", action="store_true",
        help="Visualize electronic components.")
    parser.add_argument(
        "--passport", action="store_true", help="Visualize open passport.")
    parser.add_argument(
        "--passport-closed", action="store_true",
        help="Visualize closed passport.")


def add_frame_transform_arguments(parser):
    """Add arguments to transform into objects frames to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--base-frame", type=str, default=None,
        choices=list(MOCAP_OBJECTS.keys()),
        help="Compute object-relative end-effector coordinates with respect "
             "to this object.")

    # deprecated arguments
    parser.add_argument(
        "--insole-hack", action="store_true",
        help="Insole-relative end-effector coordinates.")
    parser.add_argument(
        "--pillow-hack", action="store_true",
        help="Pillow-relative end-effector coordinates.")
    parser.add_argument(
        "--pillow-big-hack", action="store_true",
        help="Pillow-relative end-effector coordinates.")
    parser.add_argument(
        "--osai-case-hack", action="store_true",
        help="OSAI-case-relative end-effector coordinates.")
    parser.add_argument(
        "--electronic-object-hack", action="store_true",
        help="Electronic-object-relative end-effector coordinates.")
    parser.add_argument(
        "--electronic-target-hack", action="store_true",
        help="Electronic-target-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-hack", action="store_true",
        help="Passport-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-closed-hack", action="store_true",
        help="Passport-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-box-hack", action="store_true",
        help="Passport-box-relative end-effector coordinates.")


def add_artist_argument(parser):
    """Add argument for artist selection to command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command line parser
    """
    parser.add_argument(
        "--artist", choices=ARTISTS.keys(),
        help=f"Name of pytransform3d artist.")
