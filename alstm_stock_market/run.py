import yfinance as yf
from skopt.space import Categorical, Real

import alstm_stock_market.src.manager.strategies as st
import alstm_stock_market.src.model.params as p
from alstm_stock_market.src.helpers.plotter import Plotter
from alstm_stock_market.src.helpers.utils import cmd_args
from alstm_stock_market.src.manager.manager import Manager
from alstm_stock_market.src.model.evaluator import Evaluator
from alstm_stock_market.src.model.model import Model
from alstm_stock_market.src.model.preprocessor import Preprocessor


def main():
    data = yf.download(p.ticker, start=p.start, end=p.end)

    pre = Preprocessor(data)
    pre.run()

    args = cmd_args()
    model = Model(load_weights=args.load_weights)

    if args.tuning:
        param_grid = {  # Grid search only
            "model__learning_rate": [0.001, 0.01, 0.1],
            "model__hidden_state_size": [10, 20, 50, 100],
            "batch_size": [64, 128, 256, 512],
        }
        param_space = {  # Bayesian search only
            "model__learning_rate": Real(0.0001, 0.01, prior="log-uniform"),
            "model__dropout_rate": Real(0, 0.3),
            "batch_size": Categorical([64, 128, 256, 1024]),
        }
        best = model.tune(
            args.tuning,
            pre.X_train,
            pre.y_train,
            param_grid,
            param_space,
        )
        print(
            f"Best score of {best['score']} obtained with parameters {best['params']}.",
            "Overall results logs saved.",
        )
        return

    plot = Plotter()

    plot.wavelet_results(pre)
    plot.wavelet_results_detail(pre)

    model.fit(
        pre.X_train,
        pre.y_train,
        pre.X_validation,
        pre.y_validation,
    )

    plot.learning_curve(model)

    pred_train = model.predict(pre.X_train)
    pred_validation = model.predict(pre.X_validation)
    pred_test = model.predict(pre.X_test)

    plot.prediction_train(pre, pred_train)
    plot.prediction_validation(pre, pred_validation)
    plot.prediction_test(pre, pred_test)

    evaluator = Evaluator(
        pre.y_test,
        pred_test,
        pre.target_normalization_mean,
        pre.target_normalization_std,
    )
    evaluator.run()

    print("\nAvaliação dos Resultados:")
    print("Raiz do Erro Quadrático Médio:", evaluator.metrics["rmse"])
    print("Erro Absoluto Médio:", evaluator.metrics["mae"])
    print("R-quadrado:", evaluator.metrics["r2"])
    print("Tracking Error:", evaluator.metrics["te"])

    plot.returns_trend_distribution(evaluator.y_trend, evaluator.y_pred_trend)
    plot.confusion_matrix(evaluator.confusion_matrix)
    plot.cumulative_return(pre, evaluator.cumulative_return)
    plot.cumulative_return_spread(pre, evaluator.cumulative_return)

    initial_cash = 1000
    initital_bet = 100

    manager = Manager(
        initial_cash, initital_bet, evaluator.y_pred_trend, evaluator.y_return
    )

    strategies = {
        "Martingale": st.Martingale(initital_bet),
        "Paroli": st.Paroli(initital_bet),
        "D'Alembert": st.DAlembert(initital_bet),
        "Apostas Fixas": st.Fixed(initital_bet),
        "Apostas Proporcionais (10%)": st.Proportional(initial_cash, proportion=0.10),
        "Apostas Proporcionais (25%)": st.Proportional(initial_cash, proportion=0.25),
        "Apostas Proporcionais (50%)": st.Proportional(initial_cash, proportion=0.50),
        "Apostas Proporcionais (75%)": st.Proportional(initial_cash, proportion=0.75),
        "Apostas Proporcionais (90%)": st.Proportional(initial_cash, proportion=0.90),
    }

    for name, strategy in strategies.items():
        manager = Manager(
            initial_cash,
            strategy.initial_bet,
            evaluator.y_pred_trend,
            evaluator.y_return,
        )
        results = manager.run(strategy)
        plot.strategy(results, name)


if __name__ == "__main__":
    main()
