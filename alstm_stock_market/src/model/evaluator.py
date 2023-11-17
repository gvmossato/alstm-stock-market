import numpy as np

from alstm_stock_market.src.helpers.utils import save_txt


class Evaluator:
    def __init__(self, y, y_pred, normalization_mean=None, normalization_std=None):
        self.mean = normalization_mean
        self.std = normalization_std
        self.y = self._reverse_normalize(y)
        self.y_pred = self._reverse_normalize(y_pred)
        self.y_return = self._return(self.y)
        self.y_pred_return = self._return(self.y_pred)

    def _reverse_normalize(self, data):
        return self.std * data + self.mean if (self.mean and self.std) else data

    def _return(self, price):
        return np.diff(price) / price[:-1]

    def _calc_metrics(self):
        n = len(self.y)
        y_resids = self.y - self.y_pred

        def rmse():
            return np.sqrt(np.sum(y_resids**2) / n)

        def mae():
            return np.sum(np.abs(y_resids)) / n

        def r2():
            return 1 - np.sum(y_resids**2) / np.sum((self.y - np.mean(self.y)) ** 2)

        def te():
            return np.sqrt(
                np.sum((self.y_return - self.y_pred_return) ** 2) / ((n - 1) - 1)
            )

        self.metrics = {
            "rmse": rmse(),
            "mae": mae(),
            "r2": r2(),
            "te": te(),
        }
        save_txt(self.metrics, "metrics")

    def _calc_confusion_matrix(self):
        self.y_trend = self.y_return >= 0
        self.y_pred_trend = self.y_pred_return >= 0

        self.confusion_matrix = (
            {  # These indexes must match the ones in the confusion matrix plot
                "TP": np.logical_and(self.y_trend, self.y_pred_trend).sum(),  # (0,0)
                "FN": np.logical_and(self.y_trend, ~self.y_pred_trend).sum(),  # (0,1)
                "FP": np.logical_and(~self.y_trend, self.y_pred_trend).sum(),  # (1,0)
                "TN": np.logical_and(~self.y_trend, ~self.y_pred_trend).sum(),  # (1,1)
            }
        )

    def _calc_cumulative_return(self):
        self.cumulative_return = {
            "y": np.cumprod(self.y_return + 1) - 1,
            "y_pred": np.cumprod(self.y_pred_return + 1) - 1,
        }

    def run(self):
        self._calc_metrics()
        self._calc_confusion_matrix()
        self._calc_cumulative_return()
