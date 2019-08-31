import torch

import config


def train_network(epoch, model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    correct = 0
    current_samples = 0
    train_len = len(loader.sampler)
    graph_features = torch.tensor(config.cast_graph.graph_features).float()
    if config.args.use_cuda:
        graph_features = graph_features.cuda()
    for batch_idx, (features, label, cast_indices) in enumerate(loader):
        disp_batch = batch_idx + 1
        if config.args.use_cuda:
            features, label, cast_indices = features.cuda(), label.cuda(), cast_indices.cuda()
        optimizer.zero_grad()
        output = model(graph_features, features, cast_indices)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        loss = criterion(output, label)  # calculate loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
        current_samples += len(features)
        # log batch
        if disp_batch % config.train_log_interval == 0:
            percent_done = 100. * disp_batch / len(loader)
            config.log.info(f'Train Epoch: {epoch} [{current_samples}/{train_len} ({percent_done:.0f}%)]'
                            f'\tLoss: {loss.item():.6f}')
    avg_loss = epoch_loss / train_len
    acc = 100. * correct / train_len
    config.log.info(f'Train Epoch: {epoch} Average loss: {avg_loss:.4f}\tAccuracy: {correct}/{train_len} ({acc:.0f}%)')
    return avg_loss, acc


def evaluate_network(model, criterion, loader):
    model.eval()
    test_loss = 0
    correct = 0
    graph_features = torch.tensor(config.cast_graph.graph_features).float()
    for features, label, cast_indices in loader:
        if config.args.use_cuda:
            features, label = features.cuda(), label.cuda()

        output = model(graph_features, features, cast_indices)
        test_loss += criterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    n = len(loader.dataset)
    test_loss /= n
    acc = 100. * correct / n
    config.log.info('Test set: Average loss: {:.4f}\tAccuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, n, acc))
    return test_loss, acc
