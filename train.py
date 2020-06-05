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
from evaluation import *
import time

import warnings
warnings.filterwarnings('ignore')

#os.environ['CUDA_ENABLE_DEVICES'] = '0'

#torch.cuda.set_device(0)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-pre_epoch', type=int, default=5,
                   help='number of epochs for pre_train')
    p.add_argument('-n_epoch', type=int, default=30,
                   help='number of epochs for train')
    p.add_argument('-n_batch', type=int, default=8,
                   help='number of batches for train')
    p.add_argument('-lr', type=float, default=5e-4,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    p.add_argument('-tfr', type=float, default=0.5,
                   help='teacher forcing ratio')
    p.add_argument('-restore', default=False, action='store_true',
                   help='whether restore trained model')
    return p.parse_args()


def pre_train(model, optimizer, train_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.train(), Kencoder.train(), manager.train(), decoder.train()
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)

    for epoch in range(args.pre_epoch):
        epoch_start_time =time.time()
        b_loss = 0
        print('Start preTraining epoch %d ...'%(epoch))
        for step, (src_X, src_y, src_K, _) in enumerate(train_loader):
            # if step>300:
            #     break
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()

            optimizer.zero_grad()
            _, _, x = encoder(src_X)
            y = Kencoder(src_y)
            K = Kencoder(src_K)

            n_vocab = params.n_vocab
            seq_len = src_y.size(1) - 1
            _, _, _, k_logits = manager(x, y, K) # k_logits: [n_batch, n_vocab]
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab)
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))
            bow_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            b_loss += bow_loss.item()
            if (step + 1) % 10 == 0:
                b_loss /= 10
                print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f, epoch_time:%.2f s" % (epoch + 1, args.pre_epoch,
                                                                             step + 1, len(train_loader),
                                                                             b_loss,time.time()-epoch_start_time))
                b_loss = 0
        # save models
        save_models(model, params.all_restore,0-epoch-1,b_loss)


def train(model, optimizer, train_loader,valid_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(args.n_epoch):
        epoch_start_time = time.time()
        encoder.train(), Kencoder.train(), manager.train(), decoder.train()
        b_loss = 0
        k_loss = 0
        n_loss = 0
        t_loss = 0
        print('Start training epoch %d ...'%(epoch))
        for step, (src_X, src_y, src_K, tgt_y) in enumerate(train_loader):
            # if step>100:
            #     break
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()
            tgt_y = tgt_y.cuda()

            optimizer.zero_grad()
            encoder_outputs, hidden, x = encoder(src_X)
            #Warning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead. (masked_fill__cuda at ..\aten\src\ATen\native\cuda\LegacyDefinitions.cpp:19)
            #encoder_mask = (src_X == 0)[:, :encoder_outputs.size(0)].unsqueeze(1).byte()
            encoder_mask = (src_X == 0)[:, :encoder_outputs.size(0)].unsqueeze(1).bool()

            y = Kencoder(src_y)
            K = Kencoder(src_K)
            prior, posterior, k_i, k_logits = manager(x, y, K)
            kldiv_loss = KLDLoss(prior, posterior.detach())

            n_vocab = params.n_vocab
            seq_len = src_y.size(1) - 1
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab)
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))

            n_batch = src_X.size(0)
            max_len = tgt_y.size(1)

            outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
            hidden = hidden[params.n_layer:]
            output = torch.LongTensor([params.SOS] * n_batch).cuda()  # [n_batch]
            for t in range(max_len):
                output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs, encoder_mask)
                outputs[t] = output
                is_teacher = random.random() < args.tfr  # teacher forcing ratio
                top1 = output.data.max(1)[1]
                output = tgt_y[:, t] if is_teacher else top1

            outputs = outputs.transpose(0, 1).contiguous()
            nll_loss = NLLLoss(outputs.view(-1, n_vocab),
                               tgt_y.contiguous().view(-1))

            loss = kldiv_loss + nll_loss + bow_loss
            loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            b_loss += bow_loss.item()
            k_loss += kldiv_loss.item()
            n_loss += nll_loss.item()
            t_loss += loss.item()

            if (step + 1) % 10 == 0:
                k_loss /= 10
                n_loss /= 10
                b_loss /= 10
                t_loss /= 10
                print("Epoch [%.2d/%.2d],  Step [%.4d/%.4d], epoch_time:%.2f s: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
                      % (epoch + 1, args.n_epoch,
                         step + 1, len(train_loader), time.time()-epoch_start_time,
                         t_loss, k_loss, b_loss, n_loss))
                k_loss = 0
                n_loss = 0
                b_loss = 0
                t_loss = 0

        print('Start evaluateing epoch %d ...' % (epoch))
        valid_loss =  evaluate_loss(model, epoch, valid_loader)
        # save models
        save_models(model, params.all_restore,epoch, valid_loss)


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    train_path = params.train_path
    #test_path = params.test_path
    valid_path = params.valid_path
    vocab_path =  params.vocab_path
    assert torch.cuda.is_available()

    # print("building the vocab...")
    # vocab = build_vocab_music(train_path, n_vocab)

    # # save vocab
    # print("saving the vocab...")
    # with open(vocab_path, 'w',encoding='utf-8') as fp:
    #     json.dump(vocab.stoi, fp, ensure_ascii=False)
    # load vocab
    print("loading the vocab...")
    vocab = Vocabulary()
    with open(vocab_path, 'r',encoding='utf-8') as fp:
        vocab.stoi = json.load(fp)

    # load data and change to id
    print("loading_data...")
    train_X, train_y, train_K = load_data(train_path, vocab)

    train_loader = get_data_loader(train_X, train_y, train_K, n_batch,True)
    print("successfully loaded train data")
    valid_X, valid_y, valid_K = load_data(valid_path, vocab)
    valid_loader = get_data_loader(valid_X, valid_y, valid_K, n_batch,False)
    print("successfully loaded  valid data")


    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer, vocab).cuda()

    if args.restore:
        encoder = init_model(encoder, restore=params.encoder_restore)
        Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
        manager = init_model(manager, restore=params.manager_restore)
        decoder = init_model(decoder, restore=params.decoder_restore)


    model = [encoder, Kencoder, manager, decoder]
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    # pre_train knowledge manager
    print("start pre-training")
    pre_train(model, optimizer, train_loader,args)
    print("start training")
    train(model, optimizer, train_loader,valid_loader , args)

    # save final model
    save_models(model, params.all_restore)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

