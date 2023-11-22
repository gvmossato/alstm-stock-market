import numpy as np
import pandas as pd
import pywt

import alstm_stock_market.src.model.params as p


class Preprocessor:
    def __init__(self, data, sets_sizes):
        self.data = self._trim(data.dropna())
        self.dates = self.data.index
        self.target_col_idx = list(self.data.columns).index(p.target)
        self.sets_sizes = sets_sizes

    def _trim(self, data):
        if len(data) <= p.time_step:
            return data

        # Final array must have N full sequences of p.time_step
        # days, plus one more day for the last prediction
        trim_len = (len(data) - 1) % p.time_step
        print(
            f"AVISO: {trim_len} dia(s) removido(s)"
            f"para criar sequências de {p.time_step} dias"
        )
        return data[trim_len:]

    def _denoise(self):
        def calc_universal_threshold(finnest_coeffs):
            sigma = np.std(finnest_coeffs)
            n = len(finnest_coeffs)
            return sigma * np.sqrt(2 * np.log(n))

        self.data_transformed = pd.DataFrame(
            index=self.data.index,
            columns=self.data.columns,
        )

        for col in self.data.columns:
            coeffs = pywt.wavedec(
                self.data[col],
                p.wavelet,
                p.wavelet_mode,
                level=p.levels,
            )
            for i, shrink in enumerate(p.shrink_coeffs):
                if shrink:
                    threlshold = calc_universal_threshold(coeffs[-1])
                    coeffs[i] = pywt.threshold(coeffs[i], threlshold, p.threshold_mode)

            self.data_transformed[col] = pywt.waverec(
                coeffs,
                p.wavelet,
                p.wavelet_mode,
            )[: len(self.data.index)]

    def _normalize(self):
        norm_mean = self.data_transformed.mean()
        norm_std = self.data_transformed.std()
        self.target_norm_mean = norm_mean.iloc[self.target_col_idx]
        self.target_norm_std = norm_std.iloc[self.target_col_idx]

        self.data_normalized = (self.data_transformed - norm_mean) / norm_std

    def _sequentialize(self):
        if len(self.data_normalized) <= p.time_step:
            self.X = np.array([self.data_normalized])
            self.y = np.array([])
            return

        X_seq = []
        y_seq = []
        for i in range(len(self.data_normalized) - p.time_step):
            X = self.data_normalized.iloc[i : i + p.time_step, :]
            y = self.data_normalized.iloc[i + p.time_step, self.target_col_idx]
            X_seq.append(X)
            y_seq.append(y)

        self.X = np.array(X_seq)
        self.y = np.array(y_seq)

    def _split(self):
        if not np.isclose(sum(self.sets_sizes.values()), 1.0):
            raise ValueError("sets_sizes doesn't add up to 1")

        train_limit = int(np.round(len(self.X) * self.sets_sizes["train"]))
        valdn_limit = train_limit + int(
            np.round(len(self.X) * self.sets_sizes["valdn"])
        )

        self.X_train = self.X[:train_limit]
        self.y_train = self.y[:train_limit]
        self.X_valdn = self.X[train_limit:valdn_limit]
        self.y_valdn = self.y[train_limit:valdn_limit]
        self.X_test = self.X[valdn_limit:]
        self.y_test = self.y[valdn_limit:]

        self.dates_train = self.dates[:train_limit]
        self.dates_valdn = self.dates[train_limit:valdn_limit]
        self.dates_test = self.dates[valdn_limit:]

        self.label_train = self.data.iloc[:train_limit, self.target_col_idx]
        self.label_valdn = self.data.iloc[train_limit:valdn_limit, self.target_col_idx]
        self.label_test = self.data.iloc[valdn_limit:, self.target_col_idx]

    def run(self):
        self._denoise()
        self._normalize()
        self._sequentialize()
        self._split()
