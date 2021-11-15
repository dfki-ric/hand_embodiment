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
