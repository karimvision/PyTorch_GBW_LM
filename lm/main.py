import argparse
import time
import math
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua

#from stream_gbw import Vocabulary, StreamGBWDataset
from gbw import GBWDataset
from fast_gbw import FastGBWDataset
import model
import util

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/gbw',
                    help='location of the data corpus')
parser.add_argument('--train_file', type=str, default='../data/gbw',
                    help='location of the training file')
parser.add_argument('--validation_file', type=str, default='../data/gbw',
                    help='location of the training file')
parser.add_argument('--freq_file', type=str, default='../data/gbw',
                    help='location of the word_freq file')
parser.add_argument('--sid_file', type=str, default='../data/gbw',
                    help='location of the sid file')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--proj', type=bool, default=True,
                    help='use linear projection layer to map LSTM to word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.01,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# Torch
word_freq = np.load(os.path.join(args.data,args.freq_file))
mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long()
print("load word frequency mapping - complete")

ntokens = len(word_freq)
nsampled = 8192

train_corpus = FastGBWDataset(args.data, args.train_file, args.sid_file, mapto)
print("load train data - complete")

test_corpus = GBWDataset(args.data, args.validation_file, mapto)
print("load test data - complete")

# Streaming
'''
vocabulary = Vocabulary.from_file(os.path.join(args.data, "1b_word_vocab.txt"))

ntokens = len(vocabulary)
nsampled = 8192

train_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "training-monolingual.tokenized.shuffled/*"))
test_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "heldout-monolingual.tokenized.shuffled/*"), deterministic=True)
print("load dataset - complete")
'''

###############################################################################
# Build the model
###############################################################################
eval_batch_size = 1
net = model.RNNModel(ntokens, args.emsize, args.nhid, args.emsize, args.nlayers, args.proj, args.dropout)

encoder = nn.Embedding(ntokens, args.emsize)
util.initialize(encoder.weight)

twht = None
if args.tied:
    if args.nhid != args.emsize and not args.proj:
        raise ValueError('When using the tied flag, hidden must be equal to embedding size')
    twht = encoder.weight

D = args.emsize if args.proj else args.nhid
ss = model.SampledSoftmax(ntokens, nsampled, D, tied_weight=twht)

net.add_module("encoder", encoder)
net.add_module("decoder", ss)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(), args.lr, weight_decay=1e-6)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h, device_id=0):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data).cuda(device_id)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(item, device_id=0):
    data, target, wrd_cnt, batch_num = item
    return Variable(data.cuda(device_id)), Variable(target.view(-1).cuda(device_id)), wrd_cnt, batch_num

def evaluate(data_source, data_gen):
    # Turn on evaluation mode which disables dropout.
    net.eval()

    total_loss = 0
    total_word_count = 0

    hidden = net.init_hidden(eval_batch_size)
    for item in data_gen:
        data, targets, word_cnt, batch_num = get_batch(item)
        hidden = repackage_hidden(hidden)

        emb = encoder(data)
        output, hidden = net(emb, hidden)
        logits, new_targets = ss(output, targets)

        logits_flat = logits.view(-1, ntokens)
        total_loss += word_cnt * criterion(logits_flat, targets).data
        total_word_count += word_cnt
    return total_loss[0] / total_word_count

def train():
    train_loader = train_corpus.batch_generator(seq_length=args.bptt, batch_size=args.batch_size)
    total_loss = 0
    total_word_count = 0

    start_time = time.time()
    hidden = net.init_hidden(args.batch_size)
    for batch, item in enumerate(train_loader):
        net.train()
        data, targets, word_cnt, batch_len = get_batch(item)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # Network
        # RNN Hidden => GPU 0
        # embedding, softmax => GPU 1
        emb = encoder(data)
        output, hidden = net(emb, hidden)
        logits, new_targets = ss(output, targets)

        loss = criterion(logits.view(-1, nsampled+1), new_targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(net.rnn.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm(ss.parameters(), args.clip)
        if args.proj:
            torch.nn.utils.clip_grad_norm(net.proj.parameters(), args.clip)

        optimizer.step()

        total_loss += word_cnt * loss.data
        total_word_count += word_cnt

        interval = max(10, 1000)
        if (batch % interval) == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                  .format(epoch, batch, batch_len, args.lr, elapsed * 1000 / interval, loss.data[0], math.exp(loss.data[0])))
            start_time = time.time()
            sys.stdout.flush()  

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        
        test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=eval_batch_size, shuffle=False)
        val_loss = evaluate(test_corpus, test_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
               .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        print("saving model ..")
        # save the model
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(state,args.save)
        print("done saving")
        sys.stdout.flush()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.stdout.flush()

# Run on test data.
test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=eval_batch_size, shuffle=False)
test_loss = evaluate(test_corpus, test_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
