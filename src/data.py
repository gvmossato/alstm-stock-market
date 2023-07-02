import yfinance as yf
import pandas as pd
import numpy as np
import pywt


class Preprocessor:
    def get_data(self, ticker, start, end, target_column):
        self.data = yf.download(ticker, start=start, end=end)
        self.dates = self.data.index
        self.target_column_idx = np.where(self.data.columns == target_column)[0][0]

    def denoise_data(self, wavelet, mode, levels, keep_levels):
        data_transformed = pd.DataFrame(index=self.data.index, columns=self.data.columns)

        for col in self.data.columns:
            coeffs = pywt.wavedec(self.data[col], wavelet, mode, level=levels)

            # Zero out details coefficients
            for i in range(keep_levels, levels):
                coeffs[i] = np.zeros(coeffs[i].shape)

            data_transformed[col] = pywt.waverec(coeffs, wavelet, mode)

        self.transformed = data_transformed

    def normalize_data(self):
        self.normalization_mean = self.transformed.mean()
        self.normalization_std = self.transformed.std()
        self.normalized = (self.transformed - self.normalization_mean) / self.normalization_std

    def _create_sequences(self, data, seq_size):
        x_seq = []
        y_seq = []

        for i in range(len(data) - seq_size):
            x = data.iloc[i:i+seq_size, :]
            y = data.iloc[i+seq_size, self.target_column_idx]
            x_seq.append(x)
            y_seq.append(y)
        return np.array(x_seq), np.array(y_seq)

    def train_test_split(self, train_size, time_step):
        limit = int(np.round(len(self.normalized) * train_size))

        self.dates_train = self.dates[:limit]
        self.dates_test = self.dates[limit:]

        self.X_train, self.y_train = self._create_sequences(self.normalized[:limit], time_step)
        self.X_test, self.y_test = self._create_sequences(self.normalized[limit:], time_step)

    def reverse_normalize(self, data, column):
        return self.normalization_std[column] * data + self.normalization_mean[column]
