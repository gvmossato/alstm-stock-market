import json
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import plotly.io as py


def cmd_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--tunning",
        default=False,
        action="store_true",
        help="Run BayesianSearch, require `param_grid` to be set in `./alstm_stock_market/run.py`",
    )
    return parser.parse_args()


def save_dict(path, data):
    np.savetxt(path, [json.dumps(data)], fmt="%s")


def now():
    return datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def plot_candlestick(data, title="Gráfico de Velas", xlabel="Tempo", ylabel="Preço"):
    plot_data = [
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
        )
    ]

    layout = go.Layout(title=title, xaxis={"title": xlabel}, yaxis={"title": ylabel})

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()

    py.write_image(fig, os.path.join(os.environ["IMAGES"], f"{now()}_{title}.svg"))


def plot_lines(x, Y, legends, title="Gráfico de Linhas", xlabel="x", ylabel="y"):
    plot_data = []

    for y, legend in zip(Y, legends):
        plot_data.append(go.Scatter(x=x, y=y, mode="lines", name=legend))

    layout = go.Layout(title=title, xaxis={"title": xlabel}, yaxis={"title": ylabel})

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()

    py.write_image(fig, os.path.join(os.environ["IMAGES"], f"{now()}_{title}.svg"))
