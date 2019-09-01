import argparse
import datetime
import os
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import config
import ds
import logger
import model
import trainer

parser = argparse.ArgumentParser(description='Train IMDB Model')
parser.add_argument('--model', default=config.MovieNet, choices=config.models, help='the model to train')
parser.add_argument('--fc_hidden_dim', default=64, type=int, help='fully connected hidden layer dimension')
parser.add_argument('--gcn_hidden_dim', default=64, type=int, help='gcnhidden layer dimension')
parser.add_argument('--gcn_out_dim', default=20, type=int, help='gcn output dimension')

parser.add_argument('--ignore_cuda', action='store_true', help='should ignore cuda on device')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--max_epochs', default=10, type=int, help='the maximum number of epochs')

parser.add_argument('--runname', default='train', help='the exp name')
parser.add_argument('--save_dir', default='./saved_runs/', help='the path to the root run dir')
parser.add_argument('--save_model', default='model.model', help='model file name')
parser.add_argument('--save_stats', default='stats.pkl', help='statistics file name')
parser.add_argument('--ignore_timestamp', action='store_true', help='dont add datetime stamp to run dir')

parser.add_argument('--verbose', action='store_true', help='print info to screen')

config.args = parser.parse_args()

config.args.torch_device = 'cuda' if torch.cuda.is_available() and not config.args.ignore_cuda else 'cpu'
config.args.use_cuda = config.args.torch_device == 'cuda'

# create run dir and setup
if not config.args.ignore_timestamp:
    config.args.runname += datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')

run_dir = os.path.join(config.args.save_dir, config.args.runname)

os.makedirs(run_dir, exist_ok=True)
logger.set_logger(os.path.join(run_dir, 'log_' + str(config.args.runname) + '.log'))
configfile = os.path.join(run_dir, 'conf_' + str(config.args.runname) + '.config')

config.log.info(f'==> Created subdir for run at: {run_dir}')

# save configuration parameters
with open(configfile, 'w') as f:
    for arg in vars(config.args):
        f.write('{}: {}\n'.format(arg, getattr(config.args, arg)))

config.log.info('==> Loading dataset...')

config.cast_graph = ds.get_cast_graph()
train_loader, eval_loader, test_loader = ds.get_loaders()

dataset: ds.IMDBDataset = train_loader.dataset

config.log.info('==> Building model...')

if config.args.model == config.MovieNet:
    net = model.MovieNet(config.args.gcn_hidden_dim, config.args.gcn_out_dim, 2, config.args.fc_hidden_dim)
elif config.args.model == config.SimpleDNN:
    net = model.SimpleNN(config.args.fc_hidden_dim)
start_epoch = 1
net = net.to(config.args.torch_device)
# support cuda
if config.args.use_cuda:
    config.log.info('Using CUDA')
    config.log.info('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config.args.lr)

# start training
stats = []
train_avg_loss = train_acc = None
test_avg_loss = test_acc = None
for epoch in range(start_epoch, start_epoch + config.args.max_epochs):
    train_avg_loss, train_acc = trainer.train_network(epoch, net, criterion, optimizer, train_loader)

    config.log.info("Eval acc:")
    test_avg_loss, test_acc, = trainer.evaluate_network(net, criterion, eval_loader)
    stats.append([test_avg_loss, test_acc])

    config.log.info('Saving network...')
    state = {'net': net.module.state_dict() if config.args.use_cuda else net.state_dict(),
             'epoch': epoch}
    torch.save(state, os.path.join(run_dir, config.args.save_model))

    config.log.info('Saving eval statistics...')
    with open(os.path.join(run_dir, config.args.save_stats), "wb") as f:
        pickle.dump(stats, f)
