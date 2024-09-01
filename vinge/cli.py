import argparse
import os
import requests
import shutil
import sys

import vinge.config as config
import vinge.utils as utils


def main() -> int:
    parser = argparse.ArgumentParser(description="Link financial datasets on noisy names")
    parser.add_argument("task", help="[ configure | link ]", type=str)
    parser.add_argument("--candidate-links", help="Number of candidates per lhs to identify with embeddings", type=int)
    parser.add_argument("--final-links", help="Number of final matches per lhs", type=int)
    parser.add_argument("--left", help="Left table path", type=str)
    parser.add_argument("--right", help="Right table path", type=str)
    parser.add_argument("--output", help="Output directory for results", type=str)

    args = parser.parse_args()

    if args.task == "configure":
        return run_configuration()

    parser.print_help()

    return 1


def run_configuration() -> int:
    """
    Set up initial configuration
    """
    if not os.path.exists(config.VINGE_DIR):
        os.makedirs(config.VINGE_DIR)

    wts = os.path.join(config.VINGE_DIR, "mistral.gguf")
    if not os.path.exists(wts):
        utils.fetch_file(config.MISTRAL_WEIGHTS, wts)

    key = input("Enter OpenAI API key:")
    utils.set_openai_key(key)

    return 0


if __name__ == "__main__":
    sys.exit(main())

