import json
import torch
import torch.nn as nn
import params
import argparse
from utils import init_model, Vocabulary, load_data, get_data_loader
from model import Encoder, KnowledgeEncoder, Decoder, Manager


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-n_batch', type=int, default=128,
                   help='number of epochs for train')
    return p.parse_args()


def evaluate(model, test_loader):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.eval(), Kencoder.eval(), manager.eval(), decoder.eval()
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    total_loss = 0

    for step, (src_X, _, src_K, tgt_y) in enumerate(test_loader):
        src_X = src_X.cuda()
        src_K = src_K.cuda()
        tgt_y = tgt_y.cuda()

        encoder_outputs, hidden, x = encoder(src_X)
        K = Kencoder(src_K)
        k_i = manager(x, None, K)
        n_batch = src_X.size(0)
        max_len = tgt_y.size(1)
        n_vocab = decoder.n_vocab

        outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
        hidden = hidden[:decoder.n_layer]
        output = torch.LongTensor([params.SOS] * n_batch).cuda()  # [n_batch]
        for t in range(max_len):
            output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
            outputs[t] = output
            output = output.data.max(1)[1]

        loss = NLLLoss(outputs.view(-1, n_vocab),
                           tgt_y.contiguous().view(-1))
        total_loss += loss.item()
    total_loss /= len(test_loader)
    print("total_loss=%.4f" % (total_loss))


def main():
    args = parse_arguments()
    n_vocab = 20000
    n_layer = 2
    n_hidden = 100
    n_embed = 100
    n_batch = args.n_batch
    temperature = 0.8
    test_path = params.test_path
    assert torch.cuda.is_available()

    print("loading_data...")
    vocab = Vocabulary()

    with open('vocab.json', 'r') as fp:
        vocab.stoi = json.load(fp)

    for key, value in vocab.stoi.items():
        vocab.itos.append(key)

    test_X, test_y, test_K = load_data(test_path, vocab)
    test_loader = get_data_loader(test_X, test_y, test_K, n_batch)
    print("successfully loaded")

    encoder = Encoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    Kencoder = KnowledgeEncoder(n_vocab, n_embed, n_hidden, n_layer).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden, n_layer).cuda()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        Kencoder = torch.nn.DataParallel(Kencoder)
        manager = torch.nn.DataParallel(manager)
        decoder = torch.nn.DataParallel(decoder)

    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)

    model = [encoder, Kencoder, manager, decoder]
    print("evaluate...")
    evaluate(model, test_loader)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:

        print("[STOP]", e)