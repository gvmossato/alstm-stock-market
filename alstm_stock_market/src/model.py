import os

import numpy as np
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import alstm_stock_market.src.params as p
from alstm_stock_market.src.utils import now, save_dict

load_dotenv()


def create_model(learning_rate, dropout_rate, hidden_state_size=20, add_attention=True):
    model = Sequential()
    model.add(Input(shape=(p.time_step, p.num_features)))
    model.add(LSTM(hidden_state_size, return_sequences=True))
    if add_attention:
        model.add(SoftmaxAttention())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="linear"))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=p.loss_function)
    return model


class SoftmaxAttention(Layer):
    def __init__(self, **kwargs):
        super(SoftmaxAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_Q = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_K = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_V = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(SoftmaxAttention, self).build(input_shape)

    def call(self, x):
        Q = K.dot(x, self.W_Q)
        K_mat = K.dot(x, self.W_K)
        V = K.dot(x, self.W_V)

        QK = K.batch_dot(Q, K.permute_dimensions(K_mat, (0, 2, 1)))
        d_k = K.int_shape(Q)[-1]
        scaled_attention_logits = QK / K.sqrt(K.cast(d_k, dtype=K.floatx()))

        attention_weights = K.softmax(scaled_attention_logits, axis=-1)
        weighted_sum = K.batch_dot(attention_weights, V)
        return K.sum(weighted_sum, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Model:
    def __init__(self):
        self.model = create_model(p.learning_rate, p.hidden_state_size)

    def fit(self, X_train, y_train, X_validation, y_validation):
        self.model.fit(
            X_train,
            y_train,
            epochs=p.epochs,
            batch_size=p.batch_size,
            validation_data=(X_validation, y_validation),
            validation_freq=1,
            shuffle=False,
            verbose=1,
        )

    def bayesian_optimization(self, X_train, y_train, param_space):
        estimator = KerasRegressor(
            model=create_model,
            epochs=p.epochs,
            loss=p.loss_function,
            shuffle=False,
            verbose=0,
        )

        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_space,
            n_iter=1,
            scoring=f"neg_{p.loss_function}",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=3),
            refit=True,
            random_state=42,
            verbose=2,
        )
        bayes_search_result = bayes_search.fit(X_train, y_train)

        # Type checking fails to identify dynamic attributes
        print(
            f"Best score ({bayes_search_result.best_score_})",  # type: ignore
            f"obtained with parameters: {bayes_search_result.best_params_}",  # type: ignore
        )
        save_dict(
            os.path.join(os.environ["LOGS"], f"{now()}_BayesSearch_best.txt"),
            bayes_search_result.best_params_,  # type: ignore
        )
        save_dict(
            os.path.join(os.environ["LOGS"], f"{now()}_BayesSearch_results.txt"),
            bayes_search_result.cv_results_,  # type: ignore
        )

    def predict(self, X):
        return self.model.predict(X, batch_size=p.batch_size).flatten()

    def evaluate(self, y, y_pred):
        def rmse(y, yhat):
            return np.sqrt(np.sum((y - yhat) ** 2) / len(y))

        def mae(y, yhat):
            return np.sum(np.abs(y - yhat)) / len(y)

        def r2(y, yhat):
            return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

        def ret(p):
            return np.diff(p) / p[:-1]

        def te(y, yhat):
            y_ret = ret(y)
            yhat_ret = ret(yhat)
            n = len(y_ret)
            return np.sqrt(np.sum((y_ret - yhat_ret) ** 2) / (n - 1))

        metrics = {
            "rmse": rmse(y, y_pred),
            "mae": mae(y, y_pred),
            "r2": r2(y, y_pred),
            "te": te(y, y_pred),
        }

        save_dict(os.path.join(os.environ["LOGS"], f"{now()}_metrics.txt"), metrics)
        return metrics
