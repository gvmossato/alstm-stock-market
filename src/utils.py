import plotly.graph_objects as go
import plotly.io as py


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


def plot_lines(lines, title="Gráfico de Linhas", xlabel="x", ylabel="y", save=True):
    plot_data = []

    for line in lines:
        plot_data.append(
            go.Scatter(x=line["x"], y=line["y"], mode="lines", name=line["label"])
        )

    layout = go.Layout(title=title, xaxis={"title": xlabel}, yaxis={"title": ylabel})

    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()

    if save:
        py.write_image(fig, f"./images/{title}.svg")
