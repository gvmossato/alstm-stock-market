# Market Data
ticker='^GSPC'
start='1983-01-03'
end='2023-05-02'
target='Close'

# Wavelet Transform
wavelet='coif3'
mode='symmetric'
levels=3
keep_levels=1

# Model Params
train_size=0.95
time_step=20
epochs = 1000
learning_rate = 0.001
hidden_state_size = 20
batch_size = 256
time_step = 20
train_size = 0.95
