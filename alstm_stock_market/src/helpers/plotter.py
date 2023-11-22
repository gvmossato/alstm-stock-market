import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import alstm_stock_market.src.model.params as p
from alstm_stock_market.src.helpers.utils import reverse_normalize, save_image


class Plotter:
    def __init__(self, save=True):
        self.save = save

    def _gen_base_layout(self, title="", xlabel="", ylabel="", legend_pos="br"):
        legend_coordinates = {
            "tl": {"x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
            "tr": {"x": 1, "y": 1, "xanchor": "right", "yanchor": "top"},
            "bl": {"x": 0, "y": 0, "xanchor": "left", "yanchor": "bottom"},
            "br": {"x": 1, "y": 0, "xanchor": "right", "yanchor": "bottom"},
            "mr": {"x": 1, "y": 0.5, "xanchor": "right", "yanchor": "middle"},
            "ml": {"x": 0, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
        }
        return go.Layout(
            title=title,
            xaxis={"title": xlabel},
            yaxis={"title": ylabel},
            legend=legend_coordinates[legend_pos],
            margin=dict(l=10, r=10, b=50, t=50),
        )

    def _plot_lines(
        self,
        x,
        Y,
        legends,
        legend_pos="br",
        title="Gráfico de Linhas",
        xlabel="x",
        ylabel="y",
        return_only=False,
    ):
        plot_data = []
        for y, legend in zip(Y, legends):
            plot_data.append(go.Scatter(x=x, y=y, mode="lines", name=legend))

        layout = self._gen_base_layout(title, xlabel, ylabel, legend_pos)

        fig = go.Figure(data=plot_data, layout=layout)

        if return_only:
            return fig
        if self.save:
            save_image(fig, title)
        fig.show()

    def _plot_true_false_heatmap(
        self,
        x,
        y,
        Z,
        title="Mapa de calor",
        xlabel="x",
        ylabel="y",
        return_only=False,
    ):
        true_data = np.array([[Z[0, 0], np.nan], [np.nan, Z[1, 1]]])
        false_data = np.array([[np.nan, Z[0, 1]], [Z[1, 0], np.nan]])

        green_colorscale = [[0.0, "rgba(0, 80, 0, 0.5)"], [1.0, "rgba(0, 80, 0, 1)"]]
        red_colorscale = [[0.0, "rgba(130, 0, 0, 0.5)"], [1.0, "rgba(130, 0, 0, 1)"]]

        heatmap_true = go.Heatmap(
            x=x,
            y=y,
            z=true_data,
            colorscale=green_colorscale,
            showscale=False,
        )
        heatmap_false = go.Heatmap(
            x=x,
            y=y,
            z=false_data,
            colorscale=red_colorscale,
            showscale=False,
        )

        total = np.sum(Z)
        annotations = []
        for i, row in enumerate(Z):
            for j, value in enumerate(row):
                text = f"{value / total * 100:.1f}% ({value}/{total})"
                annotations.append(
                    {
                        "showarrow": False,
                        "text": text,
                        "x": x[j],
                        "y": y[i],
                        "xref": "x",
                        "yref": "y",
                        "font": dict(color="white", size=20),
                    }
                )

        layout = self._gen_base_layout(title, xlabel, ylabel)
        layout["annotations"] = annotations
        fig = go.Figure(data=[heatmap_true, heatmap_false], layout=layout)

        if return_only:
            return fig
        if self.save:
            save_image(fig, title)
        fig.show()

    def _plot_histograms(
        self,
        x,
        legends,
        legend_pos="tl",
        title="Histograma",
        xlabel="Classes",
        ylabel="Ocorrências",
        return_only=False,
    ):
        plot_data = []
        for data, name in zip(x, legends):
            plot_data.append(go.Histogram(x=data, name=name))

        layout = self._gen_base_layout(title, xlabel, ylabel, legend_pos)
        fig = go.Figure(data=plot_data, layout=layout)
        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.5)

        if return_only:
            return fig
        if self.save:
            save_image(fig, title)
        fig.show()

    def _plot_bars(
        self,
        x,
        Y,
        legends,
        legend_pos="tr",
        title="Gráfico de Barras",
        xlabel="x",
        ylabel="y",
        colors=[],
        return_only=False,
    ):
        plot_data = []
        for i, (y, legend) in enumerate(zip(Y, legends)):
            bar = go.Bar(x=x, y=y, name=legend)
            if colors:
                bar.update(marker_color=colors[i])
            plot_data.append(bar)

        layout = self._gen_base_layout(title, xlabel, ylabel, legend_pos)
        fig = go.Figure(data=plot_data, layout=layout)

        if return_only:
            return fig
        if self.save:
            save_image(fig, title)
        fig.show()

    def wavelet_results(self, pre):
        self._plot_lines(
            x=pre.dates,
            Y=[pre.data["Close"], pre.data_transformed["Close"]],
            legends=["Original", "Reconstrução"],
            title="Resultados da redução de ruído utilizando transformada wavelet",
            xlabel="Data",
            ylabel="Preço",
        )

    def wavelet_results_detail(self, pre, start="2009-01-01", end="2009-08-01"):
        dates_mask = (start <= pre.dates) & (pre.dates <= end)

        self._plot_lines(
            x=pre.dates[dates_mask],
            Y=[
                pre.data[dates_mask]["Close"],
                pre.data_transformed[dates_mask]["Close"],
            ],
            legends=["Original", "Reconstrução"],
            title="Recorte da redução de ruído utilizando transformada wavelet",
            xlabel="Data",
            ylabel="Preço",
        )

    def learning_curve(self, model):
        self._plot_lines(
            x=np.arange(1, p.epochs + 1, 1),
            Y=[model.fitted.history["loss"], model.fitted.history["val_loss"]],
            legends=["Treino", "Validação"],
            legend_pos="tr",
            title="Curva de aprendizado do modelo com os dados de treino e validação",
            xlabel="Epoch",
            ylabel="Custo",
        )

    def prediction_train(self, pre, pred):
        self._plot_lines(
            x=pre.dates_train,
            Y=[
                pre.label_train,
                reverse_normalize(pred, pre.target_norm_mean, pre.target_norm_std),
            ],
            legends=["Real", "Predição"],
            title="Resultado da predição nos dados de treino",
            xlabel="Data",
            ylabel="Preço",
        )

    def prediction_valdn(self, pre, pred):
        self._plot_lines(
            x=pre.dates_valdn,
            Y=[
                pre.label_valdn,
                reverse_normalize(pred, pre.target_norm_mean, pre.target_norm_std),
            ],
            legends=["Real", "Predição"],
            legend_pos="tr",
            title="Resultado da predição nos dados de validação",
            xlabel="Data",
            ylabel="Preço",
        )

    def prediction_test(self, pre, pred):
        self._plot_lines(
            x=pre.dates_test,
            Y=[
                pre.label_test,
                reverse_normalize(pred, pre.target_norm_mean, pre.target_norm_std),
            ],
            legends=["Real", "Predição"],
            title="Resultado da predição nos dados de teste",
            xlabel="Data",
            ylabel="Preço",
        )

    def returns_trend_distribution(self, y_return_trend, y_pred_return_trend):
        y_labels = np.where(y_return_trend, "Subida", "Queda")
        y_pred_labels = np.where(y_pred_return_trend, "Subida", "Queda")

        self._plot_histograms(
            x=[y_labels, y_pred_labels],
            legends=["Real", "Predição"],
            title="Distribuição binária das tendências de retorno",
            xlabel="Tendência",
        )

    def confusion_matrix(self, matrix):
        self._plot_true_false_heatmap(
            x=["Previu subida", "Previu queda"],
            y=["Índice subiu", "Índice caiu"],
            Z=np.reshape(list(matrix.values()), (2, 2)),
            title="Matriz de confusão para a tendência do índice frente à predição",
            xlabel="Modelo",
            ylabel="S&P 500",
        )

    def cumulative_return(self, pre, returns):
        self._plot_lines(
            x=pre.dates_test,
            Y=[returns["y"] * 100, returns["y_pred"] * 100],
            legends=["Real", "Predição"],
            title="Retorno acumulado no período para a predição e para o índice",
            xlabel="Data",
            ylabel="Retorno Acumulado (%)",
        )

    def cumulative_return_spread(self, pre, returns):
        spread = (returns["y"] - returns["y_pred"]) * 100
        mean_series = np.full(spread.shape, np.mean(spread))

        self._plot_lines(
            x=pre.dates_test,
            Y=[spread, mean_series],
            legends=["Spread", "Média"],
            title="Diferença entre o retorno acumulado do índice e da predição",
            xlabel="Data",
            ylabel="Spread (%)",
        )

    def strategy(self, results, name):
        bankroll = results["bankroll"]
        bets = results["bets"]
        gains_losses = results["gains_losses"]
        rounds = np.arange(1, len(bets) + 1, 1)

        bankroll_fig = self._plot_lines(
            x=rounds,
            Y=[bankroll],
            legends=[""],
            title="Caixa",
            xlabel="Rodada",
            ylabel="Valor (R$)",
            return_only=True,
        )

        bets_fig = self._plot_bars(
            x=rounds,
            Y=[bankroll, bets],
            legends=["Caixa", "Aposta"],
            title="Apostas",
            xlabel="Rodada",
            ylabel="Valor (R$)",
            colors=[
                ["#636EFA"] * rounds[-1],  # Default blue
                ["#FFA200"] * rounds[-1],  # Orange
            ],
            return_only=True,
        )

        gains_losses_fig = self._plot_bars(
            x=rounds,
            Y=[gains_losses],
            legends=[""],
            title="Ganhos & Perdas",
            xlabel="Rodada",
            ylabel="Valor (R$)",
            colors=[["green" if v > 0 else "red" for v in gains_losses]],
            return_only=True,
        )

        title = f"Desempenho financeiro utilizando a estratégia {name}"
        sub_figs = [bankroll_fig, bets_fig, gains_losses_fig]

        layout = self._gen_base_layout(
            title=title,
            legend_pos="mr",
        )

        fig = make_subplots(
            figure=go.Figure(layout=layout),
            rows=len(sub_figs),
            cols=1,
            subplot_titles=("Caixa", "Apostas", "Ganhos & Perdas"),
            x_title="Rodada",
            vertical_spacing=0.06,
            shared_xaxes=True,
        )

        for i, plot in enumerate(sub_figs):
            for trace in plot.data:
                trace.showlegend = False
                if plot == bets_fig:
                    trace.showlegend = True
                fig.add_trace(trace, row=i + 1, col=1)

        fig.update_layout(showlegend=True, barmode="overlay")
        fig.update_yaxes(title_text="Valor (R$)", col=1)

        if self.save:
            save_image(fig, title)
        fig.show()
