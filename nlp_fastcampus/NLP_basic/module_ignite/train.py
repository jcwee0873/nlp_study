import argparse
from ast import arg
from pkgutil import get_loader

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier, ConvolutionalClassifier
from trainer import Trainer_ignite

from utils import load_mnist, get_loaders
from utils import split_data
from utils import get_hidden_sizes

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    p.add_argument('--model', type=str, default='fc')

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == 'fc':
        model = ImageClassifier(28**2, 10)
    elif config.model =='cnn':
        model = ConvolutionalClassifier(10)
    else:
        raise NotImplementedError('You need to specify model name.')
    
    return model

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)
    # x, y = load_mnist(is_train=True, flatten=True)
    # x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    # print("Train:", x[0].shape, y[0].shape)
    # print("Valid:", x[1].shape, y[1].shape)

    # input_size = int(x[0].shape[-1])
    # output_size = int(max(y[0])) + 1

    model = get_model(config).to(device)
    # model = ImageClassifier(
    #     input_size=28**2,
    #     output_size=10,
    #     # hidden_sizes=get_hidden_sizes(input_size,
    #     #                               output_size,
    #     #                               config.n_layers),
    #     use_batch_norm=not config.use_dropout,
    #     dropout_p=config.dropout_p,
    # ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()
    # crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    # trainer = Trainer(model, optimizer, crit)

    # trainer.train(
    #     train_data=(x[0], y[0]), 
    #     valid_data=(x[1], y[1]), 
    #     config=config
    # )
    trainer = Trainer_ignite(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)





