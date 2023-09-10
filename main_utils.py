import argparse
import networkx as nx
import random
import torch
import torch.nn as nn

def arg_parser():
    parser = argparse.ArgumentParser()

    # training params
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--patience', dest='patience', type=int, default=100)
    parser.add_argument('--tol', dest='tol', type=float, default=2e-4)
    parser.add_argument('--loss', dest='loss', default=None)
    parser.add_argument('--max_sample_size', dest='max_sample_size', type=int, default=None)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=1)
    parser.add_argument('--init_std', dest='init_std', type=float, default=1e-2)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=-2,
                        help='Number of hidden channels, -1:same as input channels, -2:1/2 of input channels')

    # dataset params
    parser.add_argument('--dataset_name', dest='dataset_name', type=str)  # to add additional experiments
    parser.add_argument('--num_nodes', dest='num_nodes', type=int)
    parser.add_argument('--num_features', dest='num_features', type=int, default=128)
    parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8)

    # graph args
    parser.add_argument('--graph_dist', dest='graph_dist', type=str, default=None, help='Available distributions: "ba", "gnp", "regular"')
    parser.add_argument('--graph_dist_params', dest='dist_param', type=float, default=None)  # e.g p=0.5 for gnp
    parser.add_argument('--num_graphs_per_sample', dest='num_graphs_per_sample', type=int, default=1)  # multiple copies of same set with different graphs

    # misc
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--num_iter', dest='num_iter', type=int, default=1)  # number of times to run the experiment
    parser.add_argument('--debug', dest='debug', action='store_true', help='run on small subset for debugging')
    parser.add_argument('--test_run', dest='test_run', action='store_true', help='run on small subset for testing (larger than debug)')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='')  # prefix for experiment names in wandb
    parser.add_argument('--graphless_baseline', dest='graphless_baseline', action='store_true')  # run graphless baseline
    parser.add_argument('--eval_margin', dest='eval_margin', action='store_true')  # eval the effective margin in data and log

    return parser


def get_graph_constructor(graph_dist, dist_param):
    """Returns a graph constructor function based on the graph distribution and its parameters"""
    if graph_dist == 'gnp':
        graph_dist_name = 'gnp'
        dist_param = DEFAULT_DIST_VALUES['gnp'] if dist_param is None else dist_param
        graph_dist = lambda num_nodes: nx.gnp_random_graph(n=num_nodes, p=dist_param)
    elif graph_dist == 'ba':
        graph_dist_name = 'ba'
        dist_param = DEFAULT_DIST_VALUES['ba'] if dist_param is None else dist_param
        dist_param = int(dist_param)
        graph_dist = lambda num_nodes: nx.barabasi_albert_graph(n=num_nodes, m=dist_param)
    elif graph_dist in ['regular', 'reg']:
        graph_dist_name = 'regular'
        dist_param = DEFAULT_DIST_VALUES['regular'] if dist_param is None else dist_param
        dist_param = int(dist_param)
        graph_dist = lambda num_nodes: nx.random_regular_graph(n=num_nodes, d=dist_param)
    elif graph_dist in ['star', 'star_graph']:
        # this param is ignored for star graph
        dist_param = DEFAULT_DIST_VALUES['star'] if dist_param is None else dist_param
        def star_graph_constructor(num_nodes):
            """Create star graph with star node in random position"""
            nodes = [i for i in range(num_nodes)]
            random.shuffle(nodes)
            G = nx.star_graph(n=nodes)
            return G
        graph_dist = star_graph_constructor
    elif graph_dist in ['empty', 'empty_graph']:
        graph_dist_name = 'empty'
        graph_dist = lambda num_nodes: nx.empty_graph(n=num_nodes)
    else:
        raise ValueError('Unknown graph distribution - available distributions: "ba", "gnp", "regular"')

    return graph_dist


def ExpLoss(preds, labels):
    # cast 0-1 labels to -1, 1
    labels[labels==0] = -1

    labels = labels.view(preds.shape)
    loss = ( - labels * preds).exp()

    # loss = loss.sum()
    loss = loss.mean()
    return loss


def additional_metrics(model, prefix=''):
    """hook for extracting additional model metrics.
    Return a nested dictionary of metrics"""

    metrics = {}

    # norm of weights (without bias)
    metrics['weight_norms'] = model_weight_stats(model, prefix=prefix)

    # additional metrics

    return metrics


def model_weight_stats(model, prefix=''):
    """extracting model weight norm"""
    with torch.no_grad():
        weight_stats_dict = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_stats_dict[prefix+name] = torch.norm(param).item()

        for i, layer in enumerate(model.graph_convs):
            root = None
            top = None
            for name, param in layer.named_parameters():
                if 'lin_r' in name and 'weight' in name:
                    root = param.reshape(-1)
                elif 'lin_l' in name and 'weight' in name:
                    top = param.reshape(-1)

            # compute correlation between root and top
            corr = torch.dot(root, top) / (torch.norm(root) * torch.norm(top))

            weight_stats_dict[prefix+f'layer{i}_corr'] = corr.item()

            # norm ratio
            weight_stats_dict[prefix+f'layer{i}_norm_ratio'] = torch.norm(top) / torch.norm(root)

    return weight_stats_dict


def train_epoch(model, dloader, loss_fn, optimizer, device, classify=False, logger=None):
    running_loss = 0.0
    if classify:
        running_acc = 0.0
    for i, data in enumerate(dloader):
        labels = data.y
        inputs = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.forward(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if classify:
            running_acc += get_accuracy(outputs, labels)

    if classify:
        return running_loss / len(dloader), running_acc / len(dloader)
    else:
        return running_loss / len(dloader)


def test_epoch(model, dloader, loss_fn, device, classify=False):
    with torch.no_grad():
        running_loss = 0.0

        if classify:
            running_acc = 0.0

        for i, data in enumerate(dloader):
            # get the inputs; data is a list of [inputs, labels]
            labels = data.y
            inputs = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model.forward(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)

        if classify:
            return running_loss / len(dloader), running_acc / len(dloader)
        else:
            return running_loss / len(dloader)


def get_accuracy(outputs, labels):

    if outputs.dim() == 2 and outputs.shape[-1] > 1:
        return get_multiclass_accuracy(outputs, labels)
    else:
        preds = torch.sign(outputs).view(-1)
        correct = (preds == labels).sum()
        acc = correct / len(preds)
    return acc.item()


def get_multiclass_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == labels).sum()
    acc = correct / len(preds)
    return acc.item()


DEFAULT_DIST_VALUES = {'gnp': 0.5, 'ba': 3, 'regular': 4, 'star': None}
DEFAULT_MAX_ELEMENT_LABELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LOSS_FUNCTIONS = {'ce': nn.CrossEntropyLoss(),
                  'cross_entropy': nn.CrossEntropyLoss(),
                  'exp': ExpLoss,
                  'mse': nn.MSELoss(),}