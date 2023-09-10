from copy import deepcopy
import random
from models import GraphModel
import datasets as datasets
import main_utils as utils
import wandb
import torch_geometric as pyg
import torch
import numpy as np



DEFAULT_DIST_PARAMS = {
    'gnp': [0.3, 0.6, 0.9],
    'ba': [1, 3, 5],
    'regular': [5, 10, 15],
    'star': [None]
}

SAMPLE_SIZES = {
    'sum_binary_cls': [20, 40, 60, 100, 200, 300,
                       400, 500, 1000, 2000, 4000],
    'test_run': [20, 40, 60, 100, 200, 400, 1000, 2000, 4000]
}



def graphless_task_main(args, run=None, run_name='', dataset=None):
    # retrieve dataset
    if dataset is None:
        dataset = datasets.get_dataset(args.dataset_name,
                                       num_samples=args.num_samples,
                                       num_nodes=args.num_nodes,
                                       train_ratio=args.train_ratio,
                                       num_features=args.num_features)
        dataset._shuffle()
        dataset._train_test_split(train_ratio=args.train_ratio)

    # create a graph dataset - optional multiple graphs per sample
    if not args.graphless:
        if args.verbose: print('Augmenting dataset with graphs')
        graph_dist = utils.get_graph_constructor(graph_dist=args.graph_dist, dist_param=args.dist_param)
        dataset.augment_dataset_with_graphs(args.num_graphs_per_sample, graph_dist)
        if args.verbose: print('Done augmenting')

    trainset = dataset.trainset
    testset = dataset.testset

    # update configs with dataset info
    args.num_train_samples = len(trainset)
    args.num_val_samples = len(testset)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    if args.hidden_channels == -1:
        args.hidden_channels = args.num_features
    elif args.hidden_channels == -2:
        args.hidden_channels = args.num_features // 2

    # extract loss fn
    criterion = torch.nn.CrossEntropyLoss()
    classify = True
    if args.loss is not None:
        criterion = utils.LOSS_FUNCTIONS[args.loss]

        if args.loss == 'mse':
            classify = False
        if args.loss in ['exp', 'mse']:
            args.num_classes = 1
        else:
            args.num_classes = dataset.num_classes

    # check for debug mode
    if args.debug:
        trainset = trainset[:max([args.num_train_samples, 100])]
        testset = testset[:max([args.num_val_samples, 100])]

    # set dataloaders
    train_loader = pyg.loader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(testset, batch_size=32, shuffle=False)

    # define model, optimizer
    model = GraphModel(in_channels=args.num_features, out_channels=args.num_classes,
                       **args.__dict__)
    args.num_trainable_params = np.array([p.numel() for p in model.parameters() if p.requires_grad]).sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set-up logger
    if args.wandb:
        if run is None:
            if args.exp_name == '':
                args.exp_name = None
            run = wandb.init(project="empirical_evaluations",
                             config=args.__dict__,
                             reinit=True,
                             entity='gnnsimplbias',
                             settings=wandb.Settings(start_method='fork'),
                             name=args.exp_name)
        else:  # if logger is used - update config
            run.config.update(args.__dict__, allow_val_change=True)

    # print the configs
    if args.verbose:
        for key, item in sorted(args.__dict__.items()):
            print(f'{key}: {item}')

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    patience = args.patience
    tol = args.tol
    counter = 0

    # train model
    for epoch in range(args.num_epochs):

        train_out = utils.train_epoch(model=model,
                                      optimizer=optimizer,
                                      loss_fn=criterion,
                                      dloader=train_loader,
                                      device=device,
                                      classify=classify)

        test_out = utils.test_epoch(model=model,
                                    loss_fn=criterion,
                                    dloader=val_loader,
                                    device=device,
                                    classify=classify)

        if classify:
            train_loss, train_acc = train_out
            val_loss, val_acc = test_out
        else:
            train_loss = train_out
            train_acc = 0
            val_loss = test_out
            val_acc = 0

        # store metrics
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # extract model metrics - metrics is a nested dict
        metrics = utils.additional_metrics(model=model, prefix=run_name)

        if args.wandb:
            run.log({run_name + 'train_loss': train_loss,
                     run_name + 'train_acc': train_acc,
                     run_name + 'val_loss': val_loss,
                     run_name + 'val_acc': val_acc,
                     'epoch': epoch})

            for key, metric_dict in metrics.items():
                metric_dict.update({'epoch': epoch})
                run.log(metric_dict)

        if args.verbose:
            print(f'Epoch: {epoch}/{args.num_epochs}, Train loss: {train_loss:.5f}, Train acc: {train_acc:.3f}, '
                  f'Val loss: {val_loss:.5f}, Val acc: {val_acc:.3f}')
            for item in metrics.values():
                for key, val in item.items():
                    print(f'{key}: {val:.3f}')

        if train_loss < tol:
            counter += 1
            if counter >= patience:
                break

        # terminate on NaN
        if np.isnan(train_loss):
            break

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist


def sample_complexity_main(args, sample_sizes, run=None, run_name='', seed=0):
    """run a sample complexity experiment based on 'main'"""
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    print('Generating dataset')
    num_samples = datasets.CREATE_NUM_SAMPLES
    # small sample size for debugging
    if args.debug:
        num_samples = 2 * max(sample_sizes) + 1000
        args.verbose = True
    elif args.test_run:
        num_samples = 2 * max(sample_sizes) + 1000

    # generate the dataset
    dataset = datasets.get_dataset(args.dataset_name, root=None,
                                   num_samples=num_samples,
                                   num_nodes=args.num_nodes,
                                   train_ratio=args.train_ratio,
                                   num_features=args.num_features)
    if args.verbose: print('Done generating data')

    # record losses for each sample size - not equal length
    train_acc_hist = []  # (sample_sizes, num_epochs)
    train_loss_hist = []  # (sample_sizes, num_epochs)
    val_acc_hist = []  # (sample_sizes, num_epochs)
    val_loss_hist = []  # (sample_sizes, num_epochs)

    for sample_size in sample_sizes:
        args.num_samples = sample_size

        # prefix name for wandb
        prefix = run_name + 'N=' + str(sample_size) + '_'

        dataset._shuffle()
        dataset._train_test_split(num_train_samples=args.num_samples)

        # basic train-test pipeline - return
        train_loss, train_acc_, val_loss, val_acc = graphless_task_main(args, run, prefix, dataset)

        # store the history
        train_acc_hist.append(train_acc_)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist


def run_experimnet(macro_args, sample_sizes):
    """this method performs multiple sample complexity experiments and processes results"""

    assert macro_args.num_iter < 100, "number of iterations is limited to 100"

    # generate fixed random seeds for each iteration
    np.random.seed(0)
    seeds = np.random.randint(low=0, high=10000, size=100)

    # setup wandb logger - same over all experiment
    exp_name = macro_args.exp_name
    run = None
    if macro_args.wandb:

        if (macro_args.graph_dist is not None and macro_args.graph_dist != '') and not macro_args.graphless:
            exp_name += f'graph_dist={macro_args.graph_dist}_{macro_args.dist_param}_'

        run = wandb.init(project="your_project",  # TODO: change to your project name
                         config=macro_args.__dict__,
                         reinit=True,
                         entity='your_entity',  # TODO: change to your username
                         settings=wandb.Settings(start_method='fork'),
                         name=exp_name)

    # record the sample complexity
    sample_complexity_acc = [0 for i in range(len(sample_sizes))]
    sample_complexity_loss = [0 for i in range(len(sample_sizes))]

    # record the history of the metrics - will be a list of lists
    train_acc_history = []  # (num_iter, sample_sizes, epochs)
    train_loss_history = []  # (num_iter, sample_sizes, epochs)
    val_acc_history = []  # (num_iter, sample_sizes, epochs)
    val_loss_history = []  # (num_iter, sample_sizes, epochs)

    num_iter = macro_args.num_iter

    for i in range(num_iter):
        run_name = 'iter' + str(i) + '_'
        args = deepcopy(macro_args)
        args.seed = seeds[i]

        # run the experiment
        train_loss, train_acc, val_loss, val_acc = sample_complexity_main(args,
                                                                          sample_sizes=sample_sizes,
                                                                          run=run,
                                                                          run_name=run_name,
                                                                          seed=args.seed)

        # add metrics to histories
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        # record the best values for sample complexities
        for j in range(len(sample_sizes)):
            sample_complexity_acc[j] += max(val_acc[j]) / num_iter
            sample_complexity_loss[j] += min(val_loss[j]) / num_iter

    # log the sample complexities
    if macro_args.wandb:
        data = [[x, y] for (x, y) in zip(sample_sizes, sample_complexity_acc)]
        table = wandb.Table(data=data, columns=["sample_size", "accuracy"])
        wandb.log(
            {"sample_complexity_acc": wandb.plot.line(table, "sample_size", "accuracy",
                                                      title="sample_complexity_acc")})
        data = [[x, y] for (x, y) in zip(sample_sizes, sample_complexity_loss)]
        table = wandb.Table(data=data, columns=["sample_size", "loss"])
        wandb.log(
            {"sample_complexity_loss": wandb.plot.line(table, "sample_size", "loss",
                                                       title="sample_complexity_loss")})


    # get the mean values for each metric
    ## trim length to be uniform across iterations
    for i in range(len(sample_sizes)):
        min_len = np.inf
        for j in range(num_iter):
            min_len = min(min_len, len(val_acc_history[j][i]))

        # get the mean value for each metric
        train_loss_mean = np.stack([val_acc_history[j][i][:min_len] for j in range(num_iter)], axis=0).mean(axis=0)
        train_acc_mean = np.stack([train_acc_history[j][i][:min_len] for j in range(num_iter)], axis=0).mean(axis=0)
        val_loss_mean = np.stack([val_loss_history[j][i][:min_len] for j in range(num_iter)], axis=0).mean(axis=0)
        val_acc_mean = np.stack([val_acc_history[j][i][:min_len] for j in range(num_iter)], axis=0).mean(axis=0)

        # log to wandb
        if macro_args.wandb:
            ## logging mean values
            values = [val_acc_mean, val_loss_mean, train_acc_mean, train_loss_mean]
            names = ['val_mean_acc', 'val_mean_loss', 'train_mean_acc', 'train_mean_loss']

            for val, name in zip(values, names):
                epochs = [i for i in range(min_len)]
                plot_name =name + '_N=' + str(sample_sizes[i])

                data = [[x, y] for (x, y) in zip(epochs, val)]
                table = wandb.Table(data=data, columns=["epochs", name])
                wandb.log(
                    {plot_name : wandb.plot.line(table, "epochs", name,
                                                              title=plot_name)})




if __name__ == "__main__":
    # This script parses the arguments and runs the experiment

    macro_args = utils.arg_parser().parse_args()

    # if needs to use a specific name
    if len(macro_args.exp_name) > 0 and not macro_args.exp_name.endswith('_'):
        macro_args.exp_name += '_'

    # the sample sizes are hard coded in the beginning of the script
    sample_sizes = SAMPLE_SIZES[macro_args.dataset_name]
    if macro_args.max_sample_size is not None:
        sample_sizes = [size for size in sample_sizes if size <= macro_args.max_sample_size]

    if macro_args.dist_param is None:
        graph_dist_params = DEFAULT_DIST_PARAMS[macro_args.graph_dist]
    else:
        graph_dist_params = [macro_args.dist_param]

    # small sample sizes for debugging
    if macro_args.debug:
        sample_sizes = [50, 100]
    elif macro_args.test_run:
        sample_sizes = SAMPLE_SIZES['test_run']

    # if flag exists - run a sample complexity experiment with a graphless model
    if macro_args.graphless_baseline:
        graphless_args = deepcopy(macro_args)

        graphless_args.graphless = True
        graphless_args.graph_trans = None
        graphless_args.graph_dist = None
        graphless_args.exp_name += 'graphless_'

        run_experimnet(graphless_args, sample_sizes=sample_sizes)

    else:
        for p in graph_dist_params:
            graph_args = deepcopy(macro_args)

            graph_args.graph_trans = 'random_graph'
            # graph_args.graph_dist = 'gnp'
            graph_args.dist_param = p
            graph_args.graphless = False
            graph_args.exp_name += 'graph_'

            run_experimnet(graph_args, sample_sizes=sample_sizes)

    print('Done!')
