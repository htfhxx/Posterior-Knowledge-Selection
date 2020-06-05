import random
import argparse
import json
import os
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import params
from utils import init_model, save_models, \
    build_vocab_music, load_data, get_data_loader, Vocabulary
from model import Encoder, KnowledgeEncoder, Decoder, Manager
import time
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
import warnings
warnings.filterwarnings('ignore')

#os.environ['CUDA_ENABLE_DEVICES'] = '0'

#torch.cuda.set_device(0)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

max_len = params.max_decoder_len

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-n_batch', type=int, default=1,
                   help='number of batches for train')
    return p.parse_args()

def evaluate_loss(model, epoch, test_loader):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.eval(), Kencoder.eval(), manager.eval(), decoder.eval()
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    total_loss = 0
    for step, (src_X, _, src_K, tgt_y) in enumerate(test_loader):
        if step>30:
            break
        if (step + 1) % 10 == 0:
            print(step)
        src_X = src_X.cuda()
        src_K = src_K.cuda()
        tgt_y = tgt_y.cuda()

        encoder_outputs, hidden, x = encoder(src_X)
        encoder_mask = (src_X == 0)[:, :encoder_outputs.size(0)].unsqueeze(1).bool()

        K = Kencoder(src_K)
        k_i = manager(x, None, K)
        n_batch = src_X.size(0)
        max_len = tgt_y.size(1)
        n_vocab = params.n_vocab

        outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
        hidden = hidden[params.n_layer:]
        output = torch.LongTensor([params.SOS] * n_batch).cuda()  # [n_batch]
        for t in range(max_len):
            output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs, encoder_mask)
            outputs[t] = output
            output = output.data.max(1)[1]

        outputs = outputs.transpose(0, 1).contiguous()
        loss = NLLLoss(outputs.view(-1, n_vocab),
                           tgt_y.contiguous().view(-1))
        total_loss += loss.item()
    total_loss /= len(test_loader)
    print("epoch: %d , nll_loss=%.6f" % (epoch, total_loss))
    return total_loss

def get_distinct(seqs):
    #batch_size = len(seqs)
    #print(seqs)
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    dist_1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    dist_2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    return dist_1, dist_2

def evaluate_sample(model, vocab,test_X, test_y, test_K, test_loader):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.eval(), Kencoder.eval(), manager.eval(), decoder.eval()

    reference, candidate, source, keywords= [], [], [], []


    for step, (src_X, _, src_K, tgt_y) in enumerate(test_loader):
        # if step>100:
        #     break
        if (step + 1) % 10 == 0:
            print(step)
        src_X = src_X.cuda()
        src_K = src_K.cuda()
        tgt_y = tgt_y.cuda()

        encoder_outputs, hidden, x = encoder(src_X)
        K = Kencoder(src_K)
        k_i = manager(x, None, K)
        #n_batch = src_X.size(0)
        max_len = tgt_y.size(1)
        n_vocab = params.n_vocab

        #outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
        outputs = torch.zeros(max_len, 1, n_vocab).cuda()
        hidden = hidden[params.n_layer:]
        #output = torch.LongTensor([params.SOS] * n_batch).cuda()  # [n_batch]
        output = torch.LongTensor([params.SOS] ).cuda()  # [n_batch]

        for t in range(max_len):
            output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
            outputs[t] = output
            output = output.data.max(1)[1]

        answer = ""
        #outputs = outputs.transpose(0, 1).contiguous()
        outputs = outputs.max(2)[1]


        # print(test_X[step])
        # print(test_y[step])
        # print(test_K[step])

        x_string = ""
        for idx in test_X[step]:
            x_string += vocab.itos[idx]
        y_string = ""
        for idx in test_y[step]:
            y_string += vocab.itos[idx]
        k_string = ""
        for keyword in test_K[step]:
            for idx in keyword:
                k_string += vocab.itos[idx]
            k_string += " "

        answer = ""
        for idx in outputs:
            if idx == params.EOS:
                break
            answer += vocab.itos[idx]
        #print(src_X)

        # print("x_string:", x_string) #, "\n")
        # print("k_string:", k_string) #, "\n")
        # print("y_string:", y_string) #, "\n")
        # print("answer:", answer[:-1]) #, "\n")
        # print('\n')

        reference.append([y_string])
        candidate.append(answer[:-1])
        source.append(x_string)
        keywords.append(k_string)

    print(source)
    print(keywords)
    print(reference)
    print(candidate)

    total_bleu1 = corpus_bleu(reference, candidate, weights=(1, 0, 0, 0))
    total_bleu2 = corpus_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    distinct1, distinct2 = get_distinct(candidate)
    print(
        "Testing Result       Bleu1 sore: %.6f,  Bleu2 sore: %.6f, distinct1: %.6f,  distinct2: %.6f \n"
        % ( total_bleu1, total_bleu2,distinct1, distinct2))

        #break



def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature

    test_path = params.test_path
    vocab_path =  params.vocab_path
    assert torch.cuda.is_available()


    print("loading the vocab...")
    vocab = Vocabulary()
    with open(vocab_path, 'r',encoding='utf-8') as fp:
        vocab.stoi = json.load(fp)
    for key, value in vocab.stoi.items():
        vocab.itos.append(key)

    # load data and change to id
    print("loading_data...")
    test_X, test_y, test_K = load_data(test_path, vocab)

    test_loader = get_data_loader(test_X, test_y, test_K, n_batch,False)
    print("successfully loaded test data")

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).cuda()


    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)
    print("models successfully loaded!\n")

    model = [encoder, Kencoder, manager, decoder]

    #evaluate_loss(model, 0, test_loader)
    evaluate_sample(model, vocab,test_X, test_y, test_K, test_loader)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

