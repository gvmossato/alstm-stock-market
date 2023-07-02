import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


# Custom Attention Layer
class SoftmaxAttention(Layer):
    def __init__(self, **kwargs):
        super(SoftmaxAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = Dense(1, activation='tanh')
        super(SoftmaxAttention, self).build(input_shape)

    def call(self, x):
        logits = self.attention(x)
        x_shape = K.shape(x)
        logits = K.reshape(logits, shape=(x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        attention_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        attention_weights = K.expand_dims(attention_weights)
        weighted_input = x * attention_weights
        result = K.sum(weighted_input, axis=1)
        return result

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[-1]]


class Model:
    def __init__(self, epochs, num_features, time_step, learning_rate, hidden_state_size, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential()
        self.model.add(Input(shape=(time_step, num_features)))
        self.model.add(LSTM(hidden_state_size, return_sequences=True))
        self.model.add(SoftmaxAttention())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='linear'))

        self.optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(
            X_train,
            y_train,

            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
        )

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size).flatten()
