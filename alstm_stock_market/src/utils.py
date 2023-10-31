import json
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.io import write_image


def cmd_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--tuning",
        default=False,
        action="store_true",
        help="Run BayesianSearch, require `param_grid` to be set in `./alstm_stock_market/run.py`",
    )
    return parser.parse_args()


def save_dict(path, data):
    with open(path, 'w') as file:
        file.write(str(data))

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

    write_image(fig, os.path.join(os.environ["IMAGES"], f"{now()}_{title}.svg"))


def plot_lines(
    x, Y, legends, legend_pos="br", title="Gráfico de Linhas", xlabel="x", ylabel="y"
):
    legend_coordinates = {
        "tl": {"x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
        "tr": {"x": 1, "y": 1, "xanchor": "right", "yanchor": "top"},
        "bl": {"x": 0, "y": 0, "xanchor": "left", "yanchor": "bottom"},
        "br": {"x": 1, "y": 0, "xanchor": "right", "yanchor": "bottom"},
    }

    plot_data = []
    for y, legend in zip(Y, legends):
        plot_data.append(go.Scatter(x=x, y=y, mode="lines", name=legend))

    layout = go.Layout(
        title=title,
        xaxis={"title": xlabel},
        yaxis={"title": ylabel},
        legend=legend_coordinates[legend_pos],
        margin=dict(l=10, r=10, b=50, t=50),
    )

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()

    write_image(fig, os.path.join(os.environ["IMAGES"], f"{now()}_{title}.svg"))
