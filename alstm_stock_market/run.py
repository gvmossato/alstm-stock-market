import alstm_stock_market.src.params as p
from alstm_stock_market.src.data import Preprocessor
from alstm_stock_market.src.model import Model
from alstm_stock_market.src.utils import plot_lines

pre = Preprocessor()
pre.get_data(p.ticker, p.start, p.end, p.target)
pre.denoise_data(p.wavelet, p.wavelet_mode, p.levels, p.shrink_coeffs, p.threshold_mode)
pre.normalize_data()
pre.split_data(p.train_size, p.time_step)

plot_lines(
    x=pre.dates,
    Y=[pre.data["Close"], pre.data_transformed["Close"]],
    legends=["Original", "Reconstrução"],
    title="Resultados da redução de ruído utilizando transformada wavelet",
    xlabel="Data",
    ylabel="Preço",
)

plot_lines(
    x=pre.dates[('2009-01-01'<= pre.dates) & (pre.dates <= '2009-08-01')],
    Y=[
        pre.data[('2009-01-01'<= pre.data.index) & (pre.data.index <= '2009-08-01')]["Close"],
        pre.data_transformed[('2009-01-01'<= pre.data.index) & (pre.data.index <= '2009-08-01')]["Close"],
    ],
    legends=["Original", "Reconstrução"],
    title="Recorte da redução de ruído utilizando transformada wavelet",
    xlabel="Data",
    ylabel="Preço",
)

model = Model()

if False:
    param_grid = {
        "model__learning_rate": [0.001, 0.01, 0.1],
        "model__hidden_state_size": [10, 20, 50, 100],
        "batch_size": [64, 128, 256, 512],
    }

    model.grid_search_optimization(
        pre.X_train,
        pre.y_train,
        param_grid,
    )

else:
    model.fit(
        pre.X_train,
        pre.y_train,
        pre.X_validation,
        pre.y_validation,
    )

    pred_train = model.predict(pre.X_train)
    pred_validation = model.predict(pre.X_validation)
    pred_test = model.predict(pre.X_test)

    plot_lines(
        x=pre.dates_train,
        Y=[pre.label_train, pre.reverse_normalize(pred_train, p.target)],
        legends=["Real", "Predição"],
        title="Resultado da predição nos dados de treino",
        xlabel="Data",
        ylabel="Preço",
    )

    plot_lines(
        x=pre.dates_validation,
        Y=[pre.label_validation, pre.reverse_normalize(pred_validation, p.target)],
        legends=["Real", "Predição"],
        title="Resultado da predição nos dados de validação",
        xlabel="Data",
        ylabel="Preço",
    )

    plot_lines(
        x=pre.dates_test,
        Y=[pre.label_test, pre.reverse_normalize(pred_test, p.target)],
        legends=["Real", "Predição"],
        title="Resultado da predição nos dados de teste",
        xlabel="Data",
        ylabel="Preço",
    )

    metrics = model.evaluate(
        pre.reverse_normalize(pre.y_test, p.target),
        pre.reverse_normalize(pred_test, p.target),
    )

    print("\nAvaliação dos Resultados:")
    print("Raiz do Erro Quadrático Médio:", metrics["rmse"])
    print("Erro Absoluto Médio:", metrics["mae"])
    print("R²:", metrics["r2"])
    print("Tracking Error:", metrics["te"])
