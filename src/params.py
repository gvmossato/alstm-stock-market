# Market Data
ticker = "^GSPC"
start = "1983-01-03"
end = "2023-09-02"
target = "Close"
num_features = 6

# Wavelet Transform
wavelet = "coif3"
wavelet_mode = "symmetric"
levels = 3
shrink_coeffs = [False, True, True, True]
threshold_mode = "soft"

# Model Params
train_size = 0.95
epochs = 1000
learning_rate = 0.001
hidden_state_size = 20
batch_size = 256
time_step = 20
loss_function = "mean_squared_error"
