PAD = 0
UNK = 1
SOS = 2
EOS = 3

n_vocab = 20000
n_layer = 2
n_hidden = 800
n_embed = 200
temperature = 0.8
max_decoder_len =30

# train_path = "data/train_self_original_no_cands.txt"
# test_path = "data/test_self_original_no_cands.txt"
# valid_path = "data/valid_self_original_no_cands.txt"
train_path = "data/music/ES_3_train.txt"
test_path = "data/music/ES_3_test.txt"
valid_path = "data/music/ES_3_valid.txt"
embedding_path = 'data/music/addcomments_small_embedding.txt'
vocab_path = 'data/music/vocab.json'

model_root = "saved_model"
encoder_restore = "saved_model/pretrain/encoder_epoch_-5_43.513705"
Kencoder_restore = "saved_model/pretrain/Kencoder_epoch_-5_43.513705"
manager_restore = "saved_model/pretrain/manager_epoch_-5_43.513705"
decoder_restore = "saved_model/pretrain/decoder_epoch_-5_43.513705"

encoder_restore_mod = "saved_model/encoder"
Kencoder_restore_mod = "saved_model/Kencoder"
manager_restore_mod = "saved_model/manager"
decoder_restore_mod = "saved_model/decoder"


all_restore=[encoder_restore_mod, Kencoder_restore_mod, manager_restore_mod, decoder_restore_mod]

