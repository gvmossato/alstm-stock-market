import numpy as np
import pandas as pd
import pywt
import yfinance as yf


class Preprocessor:
    def get_data(self, ticker, start, end, target_column):
        self.data = yf.download(ticker, start=start, end=end).dropna()
        self.dates = self.data.index
        self.target_column_idx = np.where(self.data.columns == target_column)[0][0]

    def denoise_data(
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

    def normalize_data(self):
        self.normalization_mean = self.data_transformed.mean()
        self.normalization_std = self.data_transformed.std()
        self.data_normalized = (
            self.data_transformed - self.normalization_mean
        ) / self.normalization_std

    def _create_sequences(self, data, seq_size):
        x_seq = []
        y_seq = []

        for i in range(len(data) - seq_size):
            x = data.iloc[i : i + seq_size, :]
            y = data.iloc[i + seq_size, self.target_column_idx]
            x_seq.append(x)
            y_seq.append(y)
        return np.array(x_seq), np.array(y_seq)

    def split_data(self, train_size, time_step):
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

        self.X_train, self.y_train = self._create_sequences(
            self.data_normalized[:train_limit],
            time_step,
        )
        self.X_validation, self.y_validation = self._create_sequences(
            self.data_normalized[train_limit:validation_limit],
            time_step,
        )
        self.X_test, self.y_test = self._create_sequences(
            self.data_normalized[validation_limit:],
            time_step,
        )

    def reverse_normalize(self, data, column):
        return self.normalization_std[column] * data + self.normalization_mean[column]
