import tensorflow.keras.backend as K
from dotenv import load_dotenv
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import alstm_stock_market.src.model.params as p
from alstm_stock_market.src.helpers.utils import (
    get_latest_weights,
    save_txt,
    save_weights,
)

load_dotenv()


def create_model(
    learning_rate=0.00018,
    dropout_rate=0.12241,
    hidden_state_size=20,
    add_attention=True,
):
    model = Sequential()
    model.add(Input(shape=(p.time_step, p.num_features)))
    model.add(LSTM(hidden_state_size, return_sequences=add_attention))
    if add_attention:
        model.add(ClassicAttention())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="linear"))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=p.loss_function)
    return model


class ClassicAttention(Layer):
    def __init__(self, **kwargs):
        super(ClassicAttention, self).__init__(**kwargs)

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
        super(ClassicAttention, self).build(input_shape)

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


class TanhAttention(Layer):
    def __init__(self, **kwargs):
        super(TanhAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_a = self.add_weight(
            name="W_a",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_a = self.add_weight(
            name="b_a",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super(TanhAttention, self).build(input_shape)

    def call(self, x):
        s_t = K.tanh(K.dot(x, self.W_a) + self.b_a)
        attention_weights = K.softmax(s_t, axis=1)
        weighted_input = attention_weights * x
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Model:
    def __init__(self, load_weights=False):
        self.model = create_model(p.learning_rate, p.dropout_rate, p.hidden_state_size)
        self.load_weights = load_weights

        if self.load_weights:
            self.model.load_weights(get_latest_weights())

    def tune(self, method, X_train, y_train, param_grid, param_space):
        methods = {
            "grid": lambda: self._grid_search(param_grid, X_train, y_train),
            "bayes": lambda: self._bayesian_search(param_space, X_train, y_train),
        }

        try:
            return methods[method]()
        except KeyError:
            raise ValueError(
                f"Invalid tuning  method. Valid methods are: {', '.join(methods.keys())}"
            )

    def fit(self, X_train, y_train, X_valdn, y_valdn):
        if self.load_weights:
            return None

        self.fitted = self.model.fit(
            X_train,
            y_train,
            epochs=p.epochs,
            batch_size=p.batch_size,
            validation_data=(X_valdn, y_valdn),
            validation_freq=1,
            shuffle=False,
            verbose=1,
        )
        save_weights(self.model)

    def _bayesian_search(self, X_train, y_train, param_space):
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
            n_iter=100,
            scoring=f"neg_{p.loss_function}",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=3),
            refit=True,
            verbose=2,
        )
        bayes_search_result = bayes_search.fit(X_train, y_train)

        best = {
            "score": bayes_search_result.best_score_,
            "params": bayes_search_result.best_params_,
        }

        save_txt(bayes_search_result, "BayesSearch_results")
        return best

    def predict(self, X, name="y_pred"):
        return self.model.predict(X, batch_size=p.batch_size).flatten()

    def incremental_train(self, X_train, y_train):
        if not self.load_weights:
            print("Model weights were not loaded. Loading latest weights.")
            self.model.load_weights(get_latest_weights())

        self.model.fit(
            X_train,
            y_train,
            epochs=p.incremental_epochs,
            batch_size=p.batch_size,
            shuffle=False,
            verbose=1,
        )

        save_weights(self.model)
