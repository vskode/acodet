import argparse

import tensorflow as tf
import bacpipe


parser = argparse.ArgumentParser(description="Test basic bacpipe functionality.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda", type=str.lower,
                    help="Device to use for PyTorch and Tensorflow.")
parser.add_argument("--models", nargs="*", default=None, help=f"A space-separated list of models to test.")
parser.add_argument("--dashboard", "--dash", action="store_true", help="Show the dashboard at the end of the test.")
args = parser.parse_args()

bacpipe.settings.device = args.device
bacpipe.config.overwrite = True
bacpipe.config.dashboard = args.dashboard
if args.models is None:
    args.models = ["birdnet", "perch_bird"]
bacpipe.config.models = args.models
bacpipe.config.evaluation_task = ["classification"]
bacpipe.play()
