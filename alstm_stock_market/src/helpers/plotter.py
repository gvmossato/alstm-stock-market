import numpy as np
import plotly.graph_objects as go

import alstm_stock_market.src.model.params as p
from alstm_stock_market.src.helpers.utils import save_image


class Plotter:
    def __init__(self, save=True):
        self.save = save

    def _gen_base_layout(self, title, xlabel, ylabel, legend_pos="br"):
        legend_coordinates = {
            "tl": {"x": 0, "y": 1, "xanchor": "left", "yanchor": "top"},
            "tr": {"x": 1, "y": 1, "xanchor": "right", "yanchor": "top"},
            "bl": {"x": 0, "y": 0, "xanchor": "left", "yanchor": "bottom"},
            "br": {"x": 1, "y": 0, "xanchor": "right", "yanchor": "bottom"},
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
    ):
        plot_data = []
        for y, legend in zip(Y, legends):
            plot_data.append(go.Scatter(x=x, y=y, mode="lines", name=legend))

        layout = self._gen_base_layout(title, xlabel, ylabel, legend_pos)

        fig = go.Figure(data=plot_data, layout=layout)
        fig.show()

        if self.save:
            save_image(fig, title)

    def _plot_heatmap(self, x, y, Z, title="Mapa de calor", xlabel="x", ylabel="y"):
        total = np.sum(Z)
        annotations = []
        for i, row in enumerate(Z):
            for j, value in enumerate(row):
                percentage = (value / total) * 100
                text = f"{percentage:.1f}% ({value}/{total})"
                annotations.append(
                    dict(
                        showarrow=False,
                        text=text,
                        x=x[j],
                        y=y[i],
                        xref="x",
                        yref="y",
                        font=dict(color="white", size=20),
                    )
                )

        plot_data = go.Heatmap(x=x, y=y, z=Z, hoverongaps=False, colorscale="Viridis")
        layout = self._gen_base_layout(title, xlabel, ylabel)
        layout["annotations"] = annotations

        fig = go.Figure(data=plot_data, layout=layout)
        fig.show()

        if self.save:
            save_image(fig, title)

    def _plot_histograms(
        self,
        x,
        legends,
        legend_pos="tl",
        title="Histograma",
        xlabel="Classes",
        ylabel="Ocorrências",
    ):
        plot_data = []
        for data, name in zip(x, legends):
            plot_data.append(go.Histogram(x=data, name=name))

        layout = self._gen_base_layout(title, xlabel, ylabel, legend_pos)
        fig = go.Figure(data=plot_data, layout=layout)
        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.5)
        fig.show()

        if self.save:
            save_image(fig, title)

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
            Y=[pre.label_train, pre.reverse_normalize(pred, p.target)],
            legends=["Real", "Predição"],
            title="Resultado da predição nos dados de treino",
            xlabel="Data",
            ylabel="Preço",
        )

    def prediction_validation(self, pre, pred):
        self._plot_lines(
            x=pre.dates_validation,
            Y=[pre.label_validation, pre.reverse_normalize(pred, p.target)],
            legends=["Real", "Predição"],
            legend_pos="tr",
            title="Resultado da predição nos dados de validação",
            xlabel="Data",
            ylabel="Preço",
        )

    def prediction_test(self, pre, pred):
        self._plot_lines(
            x=pre.dates_test,
            Y=[pre.label_test, pre.reverse_normalize(pred, p.target)],
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
            title="Distribuição Binária das Tendências de Retorno",
            xlabel="Tendência",
        )

    def confusion_matrix(self, matrix):
        self._plot_heatmap(
            x=["Previu subida", "Previu queda"],
            y=["Índice subiu", "Índice caiu"],
            Z=np.reshape(list(matrix.values()), (2, 2)),
            title="Matriz de confusão para a direção do índice frente a predição",
            xlabel="Modelo",
            ylabel="S&P 500",
        )

    def cumulative_return(self, pre, returns):
        self._plot_lines(
            x=pre.dates_test,
            Y=[returns["y"]*100, returns["y_pred"]*100],
            legends=["Real", "Predição"],
            title="Retorno acumulado no período para a predição e para o índice",
            xlabel="Data",
            ylabel="Retorno Acumulado (%)",
        )

    def cumulative_return_spread(self, pre, returns):
        self._plot_lines(
            x=pre.dates_test,
            Y=[(returns["y"] - returns["y_pred"])*100],
            legends=["Spread"],
            title="Diferença entre o retorno acumulado do índice e da predição",
            xlabel="Data",
            ylabel="Spread (%)",
        )
