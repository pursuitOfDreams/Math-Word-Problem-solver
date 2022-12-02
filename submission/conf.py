from torch import device as device1, cuda



TRAIN_DATA_PATH = "./data/old/train/train_all_postfix.pkl"
TEST_DATA_PATH = "./data/old/test/test_all_postfix.pkl"
train_batch_size = 128
test_batch_size = 12
device = device1('cuda' if cuda.is_available() else 'cpu')

lr = 0.001
factor = 0.9
patience = 10
warmup = 100
max_len = 256
d_model= 256
drop_prob = 0.1
beta_1 = 0.95
beta_2 = 0.99
n_layers = 2
inf = float('inf')
clip = 1.0
epochs = 300
n_heads = 8
ffn_hidden = 1024
