import os
from argparse import ArgumentParser
from datetime import datetime

from plotly.io import write_image

now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def cmd_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--tuning",
        default=None,
        choices=["grid", "bayes"],
        help="Run specified tuning method, choices are 'bayes' for BayesianSearch or 'grid' for GridSearch. Set params in code directly.",
    )
    return parser.parse_args()


def save_txt(data, name):
    path = os.path.join(os.environ["LOGS"], f"{now}_{name}.txt")
    with open(path, "w") as file:
        file.write(str(data))


def save_image(fig, title):
    path = os.path.join(os.environ["IMAGES"], f"{now}_{title}.svg")
    write_image(fig, path)
