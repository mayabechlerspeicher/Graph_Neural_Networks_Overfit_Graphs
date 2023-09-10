import torch
import torch_geometric as pyg
from models import RCOVmodel
from networkx.generators.random_graphs import erdos_renyi_graph
import wandb
import argparse
import utils
import numpy as np
import train_utils
from utils import EarlyStopping
import random

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', dest='num_nodes', type=int, default=20)
    parser.add_argument('--num_samples', dest='num_samples', type=int, default=1000)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--d', dest='d', type=int, default=16)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=1)
    parser.add_argument('--num_test_samples', dest='num_test_samples', type=int, default=100)
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--num_train_samples', dest='num_train_samples', type=int, default=1000)
    parser.add_argument('--wandb', dest='wandb', type=int, default=1)
    parser.add_argument('--patience', dest='patience', type=int, default=100)
    parser.add_argument('--task_type', dest='task_type', type=str, default='Node Sum')
    parser.add_argument('--RCOV_threshold', dest='RCOV_threshold', type=float, default=0.15)
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_samples = args.num_samples
    num_test_samples = args.num_test_samples
    eps = args.eps
    train_p = args.train_p
    output_dim = 2
    d = args.d
    n_layers = args.n_layers
    train_batch_size = args.train_batch_size
    hidden_channels = args.hidden_channels
    model_type = RCOVmodel
    wd = 0.0001
    RCOV_threshold = args.RCOV_threshold
    n_conv_layers = args.n_layers
    loss = torch.nn.CrossEntropyLoss
    num_epochs = args.num_epochs
    lr = args.lr
    optimizer_type = torch.optim.Adam
    loss_thresh = 0.0001
    n_conv_layers = args.n_layers
    num_epochs = args.num_epochs
    num_classes = 2
    margin = 0.01

    teacher = utils.sample_teacher(d, self=True, top=False)
    train_dist_type = erdos_renyi_graph
    train_dist = lambda num_nodes: train_dist_type(n=num_nodes, p=train_p)

    test_acc_original = []
    test_acc_RCOV = []
    test_acc_empty = []

    np.random.seed(0)
    seeds = np.random.randint(low=0, high=10000, size=10)

    for i in range(10):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_original_pyg_graphs, train_original_nx_graphs = utils.get_dif_graphs_dataset(num_samples=num_samples,
                                                                               num_nodes=num_nodes, num_features=d,
                                                                               teacher=teacher, graph_dist=train_dist,
                                                                               margin=margin,
                                                                               self_teacher=True,
                                                                               with_edge_weight=True)
        utils.add_constant_feature_1(train_original_pyg_graphs)
        train_original_mean, _, _, _ = utils.get_moments(train_original_nx_graphs)
        if args.task_type== 'Edges':
            utils.set_label_to_num_of_edges(train_original_pyg_graphs, threshold=train_original_mean * num_nodes)

        val_original_pyg_graphs, _ = utils.get_dif_graphs_dataset(num_samples=num_samples,
                                                                            num_nodes=num_nodes, num_features=d,
                                                                            teacher=teacher, graph_dist=train_dist,
                                                                            margin=margin,
                                                                            self_teacher=True,
                                                                            with_edge_weight=True)
        utils.add_constant_feature_1(val_original_pyg_graphs)
        if args.task_type== 'Edges':
            utils.set_label_to_num_of_edges(val_original_pyg_graphs, threshold=train_original_mean * num_nodes)


        test_original_pyg_graphs, _ = utils.get_dif_graphs_dataset(num_samples=num_samples,
                                                                             num_nodes=num_nodes, num_features=d,
                                                                             teacher=teacher, graph_dist=train_dist,
                                                                             margin=margin,
                                                                             self_teacher=True,
                                                                             with_edge_weight=True)
        utils.add_constant_feature_1(test_original_pyg_graphs)
        if args.task_type== 'Edges':
            utils.set_label_to_num_of_edges(test_original_pyg_graphs, threshold=train_original_mean * num_nodes)


        train_RCOV_pyg_graphs, train_RCOV_nx_graphs = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=train_original_pyg_graphs, reduce_goal=RCOV_threshold)
        train_RCOV_mean, train_RCOV_std, train_RCOV_skew, train_RCOV_kurt = utils.get_moments(train_RCOV_nx_graphs)
        train_RCOV_std_mean_ratio = train_RCOV_std / train_RCOV_mean
        print('train_RCOV_std_mean_ratio: ', train_RCOV_std_mean_ratio)

        val_RCOV_pyg_graphs, val_RCOV_nx_graphs = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=val_original_pyg_graphs, reduce_goal=RCOV_threshold)
        val_RCOV_mean, val_RCOV_std, val_RCOV_skew, val_RCOV_kurt = utils.get_moments(val_RCOV_nx_graphs)
        val_RCOV_std_mean_ratio = val_RCOV_std / val_RCOV_mean
        print('val_RCOV_std_mean_ratio: ', val_RCOV_std_mean_ratio)

        test_RCOV_pyg_graphs, test_RCOV_nx_graphs = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=test_original_pyg_graphs, reduce_goal=RCOV_threshold)
        test_RCOV_mean, test_RCOV_std, test_RCOV_skew, test_RCOV_kurt = utils.get_moments(test_RCOV_nx_graphs)
        test_RCOV_std_mean_ratio = test_RCOV_std / test_RCOV_mean
        print('test_RCOV_std_mean_ratio: ', test_RCOV_std_mean_ratio)


        n_features = train_original_pyg_graphs[0].x.shape[1]

        train_original_loader = pyg.loader.DataLoader(train_original_pyg_graphs, batch_size=train_batch_size)
        train_RCOV_loader = pyg.loader.DataLoader(train_original_pyg_graphs, batch_size=train_batch_size)

        val_original_loader = pyg.loader.DataLoader(val_original_pyg_graphs, batch_size=num_test_samples)
        val_RCOV_loader = pyg.loader.DataLoader(val_RCOV_pyg_graphs, batch_size=num_test_samples)


        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        config = {
            'data_dir_number': args.data_dir_number,
            'd': d,
            'num_nodes': num_nodes,
            'eps': eps,
            'lr': lr,
            'loss': loss.__name__,
            'train_batch_size': train_batch_size,
            'hidden_channels': hidden_channels,
            'n_conv_layers': n_conv_layers,
            'num_epochs': num_epochs,
            'optimizer': optimizer_type.__name__,
            'train_dist': train_dist_type.__name__,
            'model': model_type.__name__,
            'file': 'adding_edges_to_real_data.py',
            'num_final_feauturs': n_features,
            'device': device.type,
            'loss_thresh': loss_thresh,
            'wd': wd,
            'iter': i,
            'task_type' : args.task_type,
        }

        if args.wandb:
            for name, val in config.items():
                print(f'{name}: {val}')
            run = wandb.init(project='your_project', reinit=True, entity='you_entity',
                             settings=wandb.Settings(start_method='fork'),
                             config=config)

        # train on empty graphs
        model_empty = model_type(in_channels=n_features, out_channels=num_classes, num_layers=n_layers,
                                    hidden_channels=hidden_channels)

        optimizer_empty = optimizer_type(params=model_empty.parameters(), lr=lr, weight_decay=wd)
        loss_fn_empty = loss()
        model_empty.to(device)
        early_stop = EarlyStopping(patience=args.patience)
        empty_graphs_train = utils.get_empty_graphs_from_datasets(train_original_pyg_graphs)
        empty_graphs_val = utils.get_empty_graphs_from_datasets(val_original_pyg_graphs)
        empty_train_loader = pyg.loader.DataLoader(empty_graphs_train, batch_size=train_batch_size, shuffle=True)
        empty_val_loader = pyg.loader.DataLoader(empty_graphs_val, batch_size=train_batch_size, shuffle=True)
        for epoch in range(num_epochs):
            train_empty_loss, train_empty_acc = train_utils.train_epoch(model_empty, dloader=empty_train_loader,
                                                            loss_fn=loss_fn_empty,
                                                            optimizer=optimizer_empty,
                                                            classify=True, device=device)
            val_empty_loss, val_empty_acc = train_utils.test_epoch(model_empty, dloader=empty_val_loader, loss_fn=loss_fn_empty,
                                                       classify=True,
                                                       device=device)

            early_stop(val_empty_loss)
            if early_stop.early_stop:
                print(f'early stop at epoch: {epoch}')
                break

            if args.wandb:
                wandb.log({'train_empty_loss': train_empty_loss,
                           'train_empty_acc': train_empty_acc,
                           'val_empty_loss': val_empty_loss,
                           'val_empty_acc': val_empty_acc})

        model_original = model_type(in_channels=n_features, out_channels=num_classes, num_layers=n_layers,
                            hidden_channels=hidden_channels)
        optimizer_original = optimizer_type(params=model_original.parameters(), lr=lr, weight_decay=wd)
        loss_fn_original = loss()
        model_original.to(device)
        early_stop_original = EarlyStopping(patience=args.patience)
        for epoch in range(num_epochs):
            train_original_loss, train_original_acc = train_utils.train_epoch(model_original, dloader=train_original_loader, loss_fn=loss_fn_original,
                                                        optimizer=optimizer_original,
                                                        classify=True, device=device)
            val_original_loss, val_original_acc = train_utils.test_epoch(model_original, dloader=val_original_loader,
                                                                loss_fn=loss_fn_original, classify=True, device=device)
            early_stop_original(val_original_loss)
            if early_stop_original.early_stop:
                print(f'early stop at epoch: {epoch}')
                break

            if args.wandb:
                wandb.log({'train_original_loss': train_original_loss,
                           'train_original_acc': train_original_acc,
                           'val_original_loss': val_original_loss,
                           'val_original_acc': val_original_acc})


        model_RCOV = model_type(in_channels=n_features, out_channels=num_classes, num_layers=n_layers,
                            hidden_channels=hidden_channels)

        optimizer_RCOV = optimizer_type(params=model_RCOV.parameters(), lr=lr, weight_decay=wd)
        loss_fn_RCOV = loss()
        model_RCOV.to(device)
        early_stop = EarlyStopping(patience=args.patience)

        for epoch in range(num_epochs):
            train_RCOV_loss, train_RCOV_acc = train_utils.train_epoch(model_RCOV, dloader=train_RCOV_loader, loss_fn=loss_fn_RCOV,
                                                        optimizer=optimizer_RCOV,
                                                        classify=True, device=device)
            val_RCOV_loss, val_RCOV_acc = train_utils.test_epoch(model_RCOV, dloader=val_RCOV_loader, loss_fn=loss_fn_RCOV,
                                                     classify=True,
                                                     device=device)
            early_stop(val_RCOV_loss)
            if early_stop_original.early_stop:
                print(f'early stop at epoch: {epoch}')
                break

            if args.wandb:
                wandb.log({'train_RCOV_loss': train_RCOV_loss,
                           'train_RCOV_acc': train_RCOV_acc,
                           'val_RCOV_loss': val_RCOV_loss,
                           'val_RCOV_acc': val_RCOV_acc})


        #test

        test_original_loader = pyg.loader.DataLoader(test_original_pyg_graphs, batch_size=num_test_samples, shuffle=True)
        test_RCOV_loader = pyg.loader.DataLoader(test_RCOV_pyg_graphs, batch_size=num_test_samples, shuffle=True)
        empty_graphs_test = utils.get_empty_graphs_from_datasets(test_original_pyg_graphs)
        empty_test_loader = pyg.loader.DataLoader(empty_graphs_test, batch_size=num_test_samples, shuffle=True)

        test_original_loss, test_original_acc = train_utils.test_epoch(model_original, dloader=test_original_loader
                                                                           , loss_fn=loss_fn_original, classify=True, device=device)
        test_RCOV_loss, test_RCOV_acc = train_utils.test_epoch(model_RCOV, dloader=test_RCOV_loader
                                                                            , loss_fn=loss_fn_RCOV, classify=True, device=device)
        test_empty_loss, test_empty_acc = train_utils.test_epoch(model_empty, dloader=empty_test_loader
                                                                            , loss_fn=loss_fn_empty, classify=True, device=device)

        test_acc_original.append(test_original_acc)
        test_acc_RCOV.append(test_RCOV_acc)
        test_acc_empty.append(test_empty_acc)

        if args.wandb:
            wandb.log({'test_original_loss': test_original_loss,
                       'test_original_acc': test_original_acc,
                       'test_RCOV_loss': test_RCOV_loss,
                       'test_RCOV_acc': test_RCOV_acc,
                       'test_empty_loss': test_empty_loss,
                       'test_empty_acc': test_empty_acc})



    if args.wandb:
        wandb.log({'test_acc_original': np.mean(test_acc_original),
                       'test_acc_RCOV': np.mean(test_acc_RCOV),
                       'test_acc_empty': np.mean(test_acc_empty),
                       'test_acc_original_std': np.std(test_acc_original),
                       'test_acc_RCOV_std': np.std(test_acc_RCOV),
                       'test_acc_empty_std': np.std(test_acc_empty)})






