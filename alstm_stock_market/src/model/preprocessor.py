import numpy as np
import pandas as pd
import pywt

import alstm_stock_market.src.model.params as p


class Preprocessor:
    def __init__(self, data):
        self.data = data.dropna()
        self.dates = self.data.index
        self.target_column_idx = np.where(self.data.columns == p.target)[0][0]

    def _denoise(
        self,
        wavelet,
        wavelet_mode,
        levels,
        shrink_coeffs,
        threshold_mode,
    ):
        def calc_universal_threshold(finnest_coeffs):
            sigma = np.std(finnest_coeffs)
            n = len(finnest_coeffs)
            return sigma * np.sqrt(2 * np.log(n))

        data_transformed = pd.DataFrame(
            index=self.data.index,
            columns=self.data.columns,
        )

        for col in self.data.columns:
            coeffs = pywt.wavedec(self.data[col], wavelet, wavelet_mode, level=levels)

            for i, shrink in enumerate(shrink_coeffs):
                if shrink:
                    threlshold = calc_universal_threshold(coeffs[-1])
                    coeffs[i] = pywt.threshold(coeffs[i], threlshold, threshold_mode)

            data_transformed[col] = pywt.waverec(coeffs, wavelet, wavelet_mode)

        self.data_transformed = data_transformed

    def _normalize(self):
        self.normalization_mean = self.data_transformed.mean()
        self.normalization_std = self.data_transformed.std()

        self.target_normalization_mean = self.normalization_mean[self.target_column_idx]
        self.target_normalization_std = self.normalization_std[self.target_column_idx]

        self.data_normalized = (
            self.data_transformed - self.normalization_mean
        ) / self.normalization_std

    def _split(self, train_size, time_step):
        def create_sequences(data, seq_size):
            x_seq = []
            y_seq = []

            for i in range(len(data) - seq_size):
                x = data.iloc[i : i + seq_size, :]
                y = data.iloc[i + seq_size, self.target_column_idx]
                x_seq.append(x)
                y_seq.append(y)
            return np.array(x_seq), np.array(y_seq)

        train_limit = int(np.round(len(self.data) * train_size))
        remaining_size = len(self.data) - train_limit
        validation_limit = train_limit + int(np.round(remaining_size / 2))

        self.label_train = self.data.iloc[time_step:train_limit, self.target_column_idx]
        self.label_validation = self.data.iloc[
            train_limit + time_step : validation_limit, self.target_column_idx
        ]
        self.label_test = self.data.iloc[
            validation_limit + time_step :, self.target_column_idx
        ]

        self.dates_train = self.dates[time_step:train_limit]
        self.dates_validation = self.dates[train_limit + time_step : validation_limit]
        self.dates_test = self.dates[validation_limit + time_step :]

        self.X_train, self.y_train = create_sequences(
            self.data_normalized[:train_limit],
            time_step,
        )
        self.X_validation, self.y_validation = create_sequences(
            self.data_normalized[train_limit:validation_limit],
            time_step,
        )
        self.X_test, self.y_test = create_sequences(
            self.data_normalized[validation_limit:],
            time_step,
        )

    def reverse_normalize(self, data, column):
        return self.normalization_std[column] * data + self.normalization_mean[column]

    def run(self):
        self._denoise(
            p.wavelet,
            p.wavelet_mode,
            p.levels,
            p.shrink_coeffs,
            p.threshold_mode,
        )
        self._normalize()
        self._split(
            p.train_size,
            p.time_step,
        )
