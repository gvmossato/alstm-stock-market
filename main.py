from src.utils import plot_lines
import src.params as p

from src.model import Model
from src.data import Preprocessor


# ==== #
# Data #
# ==== #

# Definition

pre = Preprocessor()
pre.get_data(p.ticker, p.start, p.end, p.target)
pre.denoise_data(p.wavelet, p.mode, p.levels, p.keep_levels)
pre.normalize_data()
pre.train_test_split(p.train_size, p.time_step)

# Results

plot_lines(
    [
        {
            'y': pre.data['Close'],
            'x': pre.dates,
            'label': 'Real',
        },
        {
            'y': pre.transformed['Close'],
            'x': pre.dates,
            'label': 'Reconstrução',
        },
    ],
    "Resultados da Redução de Ruído por Wavelet",
    "Data",
    "Preço"
)

# ===== #
# Model #
# ===== #

# Definition

num_features = pre.X_train.shape[-1]

model = Model(p.epochs, num_features, p.time_step, p.learning_rate, p.hidden_state_size, p.batch_size)
model.fit(pre.X_train, pre.y_train, pre.X_test, pre.y_test)

train_pred = model.predict(pre.X_train)
test_pred = model.predict(pre.X_test)

# Results

plot_lines(
    [
        {
            'y': pre.reverse_normalize(pre.y_train, p.target),
            'x': pre.dates_train,
            'label': 'Real',
        },
        {
            'y': pre.reverse_normalize(train_pred, p.target),
            'x': pre.dates_train,
            'label': 'Predição',
        },
    ],
    "Resultados dataset de treino",
    "Data",
    "Preço"
)

plot_lines(
    [
        {
            'y': pre.reverse_normalize(pre.y_test, p.target),
            'x': pre.dates_test,
            'label': 'Real',
        },
        {
            'y': pre.reverse_normalize(test_pred, p.target),
            'x': pre.dates_test,
            'label': 'Predição',
        },
    ],
    "Resultados dataset de teste",
    "Data",
    "Preço"
)
