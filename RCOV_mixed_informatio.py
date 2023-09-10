import torch
import torch_geometric as pyg
from models import RCOVmodel
from networkx.generators.random_graphs import erdos_renyi_graph
import wandb
import argparse
import utils
import numpy as np
import random
import train_utils

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', dest='num_nodes', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--d', dest='d', type=int, default=16)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=1)
    parser.add_argument('--num_test_samples', dest='num_test_samples', type=int, default=100)
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--lin_trans_for_features_dim', dest='lin_trans_for_features_dim', type=int, default=128)
    parser.add_argument('--RCOV_threshold', dest='RCOV_threshold', type=float, default=0.15)
    parser.add_argument('--patience', dest='patience', type=int, default=100)
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_samples = args.num_samples
    num_test_samples = args.num_test_samples
    margin = 0.01
    RCOV_threshold = args.RCOV_threshold
    output_dim = 2
    d = args.d
    n_layers = args.n_layers
    train_batch_size = args.train_batch_size
    hidden_channels = args.hidden_channels
    model_type = RCOVmodel
    wd = 0.0001
    lin_trans_for_features_dim = args.lin_trans_for_features_dim
    n_conv_layers = args.n_layers
    loss = torch.nn.CrossEntropyLoss
    num_epochs = args.num_epochs
    lr = args.lr
    train_p = 0.5
    optimizer_type = torch.optim.Adam
    num_training_samples_for_learning_curve = [10, 20, 50, 80, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    max_num_of_sampels = num_training_samples_for_learning_curve[-1]
    np.random.seed(0)

    teacher = utils.sample_teacher(d, self=True, top=False)  # should have unit norm
    train_dist_type = erdos_renyi_graph
    train_dist = lambda num_nodes: train_dist_type(n=num_nodes, p=train_p)
    fixed_lin_transoform = np.random.random(size=(num_nodes, lin_trans_for_features_dim))
    seeds = np.random.randint(low=0, high=10000, size=10)

    test_informative_accs = []
    test_non_informative_accs = []
    test_non_informative_rcov_accs = []
    test_empty_graphs_accs = []

    for i in range(10):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_informative_pyg, train_informative_nx = utils.get_dif_graphs_dataset(num_samples=max_num_of_sampels,
                                                                                   num_nodes=num_nodes, num_features=d,
                                                                                   teacher=teacher,
                                                                                   graph_dist=train_dist,
                                                                                   margin=margin,
                                                                                   self_teacher=True,
                                                                                   with_edge_weight=True)

        utils.encode_undirected_graph_in_features_then_linear_transform(train_informative_pyg, fixed_lin_transoform)
        utils.add_random_uniform_feature(train_informative_pyg)
        random_uniform_feature_index = train_informative_pyg[0].x.shape[1] - 1
        utils.add_constant_feature_1(train_informative_pyg)
        train_informative_mean, train_informative_std, train_informative_skew, train_informative_kurt = utils.get_moments(
            train_informative_nx)
        utils.set_label_to_num_of_edges(train_informative_pyg, threshold=train_informative_mean * num_nodes)

        val_informative_pyg, _ = utils.get_dif_graphs_dataset(num_samples=num_test_samples,
                                                              num_nodes=num_nodes, num_features=d,
                                                              teacher=teacher, graph_dist=train_dist,
                                                              margin=margin,
                                                              self_teacher=True,
                                                              with_edge_weight=True)

        utils.encode_undirected_graph_in_features_then_linear_transform(val_informative_pyg, fixed_lin_transoform)
        utils.add_random_uniform_feature(val_informative_pyg)
        utils.add_constant_feature_1(val_informative_pyg)

        utils.set_label_to_num_of_edges(val_informative_pyg, threshold=train_informative_mean * num_nodes)

        test_informative_pyg, _ = utils.get_dif_graphs_dataset(num_samples=max_num_of_sampels,
                                                               num_nodes=num_nodes, num_features=d,
                                                               teacher=teacher, graph_dist=train_dist,
                                                               margin=margin,
                                                               self_teacher=True,
                                                               with_edge_weight=True)

        utils.encode_undirected_graph_in_features_then_linear_transform(test_informative_pyg, fixed_lin_transoform)
        utils.add_random_uniform_feature(test_informative_pyg)
        utils.add_constant_feature_1(test_informative_pyg)
        utils.set_label_to_num_of_edges(test_informative_pyg, threshold=train_informative_mean * num_nodes)

        train_non_informative_pyg = utils.build_new_graphs_from_feature(train_informative_pyg,
                                                                        feature_index=random_uniform_feature_index,
                                                                        with_edge_weight=True)

        val_non_informative_pyg = utils.build_new_graphs_from_feature(val_informative_pyg,
                                                                      feature_index=random_uniform_feature_index,
                                                                      with_edge_weight=True)

        test_non_informative_pyg = utils.build_new_graphs_from_feature(test_informative_pyg,
                                                                       feature_index=random_uniform_feature_index,
                                                                       with_edge_weight=True)

        train_non_informative_rcov_pyg, _ = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=train_non_informative_pyg, reduce_goal=RCOV_threshold)

        val_non_informative_rcov_pyg, _ = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=val_non_informative_pyg, reduce_goal=RCOV_threshold)

        test_non_informative_rcov_pyg, _ = utils.batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(
            pyg_graphs=test_non_informative_pyg, reduce_goal=RCOV_threshold)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        num_of_features = test_informative_pyg[0].x.shape[1]

        config = {
            'num_test_samples': num_test_samples,
            'd': d,
            'num_nodes': num_nodes,
            'lr': lr,
            'loss': loss.__name__,
            'train_batch_size': train_batch_size,
            'hidden_channels': hidden_channels,
            'n_conv_layers': n_conv_layers,
            'output_dim': output_dim,
            'num_epochs': num_epochs,
            'optimizer': optimizer_type.__name__,
            'num_final_feauturs': num_of_features,
            'device': device.type,
            'num_training_samples_for_learning_curve': num_training_samples_for_learning_curve,
            'max_num_of_sampels': max_num_of_sampels,
            'wd': wd,
            'num_data_samples': len(train_informative_pyg),
            'data_dir_number': args.data_dir_number,
            'RCOV_threshold': RCOV_threshold,
            'iter': i,
        }

        if args.wandb:
            for name, val in config.items():
                print(f'{name}: {val}')
            run = wandb.init(project='your project', reinit=True, entity='your entity', config=config)

        val_informative_loader = pyg.loader.DataLoader(val_informative_pyg[:num_test_samples],
                                                       batch_size=num_test_samples)
        val_non_informative_loader = pyg.loader.DataLoader(val_non_informative_pyg[:num_test_samples],
                                                           batch_size=num_test_samples)
        val_non_informative_rcov_loader = pyg.loader.DataLoader(val_non_informative_rcov_pyg[:num_test_samples],
                                                                batch_size=num_test_samples)
        val_empty_graphs = utils.get_empty_graphs_from_datasets(val_informative_pyg[:num_test_samples])
        val_empty_graphs_loader = pyg.loader.DataLoader(val_empty_graphs, batch_size=num_test_samples)

        test_informative_loader = pyg.loader.DataLoader(test_informative_pyg[:num_test_samples],
                                                        batch_size=num_test_samples)
        test_non_informative_loader = pyg.loader.DataLoader(test_non_informative_pyg[:num_test_samples],
                                                            batch_size=num_test_samples)
        test_non_informative_rcov_loader = pyg.loader.DataLoader(test_non_informative_rcov_pyg[:num_test_samples],
                                                                 batch_size=num_test_samples)
        test_empty_graphs = utils.get_empty_graphs_from_datasets(test_informative_loader)
        test_empty_graphs_loader = pyg.loader.DataLoader(test_empty_graphs[:num_test_samples],
                                                         batch_size=num_test_samples)

        train_empty_graphs = utils.get_empty_graphs_from_datasets(train_informative_pyg)
        for num_training_samples in num_training_samples_for_learning_curve:
            if args.wandb:
                print(f'Training with {num_training_samples} samples')
                wandb.log({'num_training_samples': num_training_samples})

            train_informative_loader = pyg.loader.DataLoader(train_informative_pyg[:num_training_samples],
                                                             batch_size=train_batch_size)
            model_informative = model_type(in_channels=num_of_features, out_channels=output_dim, num_layers=n_layers,
                                           hidden_channels=hidden_channels)
            optimizer = optimizer_type(params=model_informative.parameters(), lr=lr, weight_decay=wd)
            model_informative.to(device)
            early_stop = utils.EarlyStopping(patience=args.patience)
            loss_fn = loss()

            for epoch in range(num_epochs):
                print(f'epoch: {epoch}, num_training_samples: {num_training_samples}, iter: {i}, informative graphs')
                train_informative_loss, train_informative_acc = train_utils.train_epoch(model_informative,
                                                                                        dloader=train_informative_loader,
                                                                                        loss_fn=loss_fn,
                                                                                        optimizer=optimizer,
                                                                                        classify=True,
                                                                                        device=device)
                val_informative_loss, val_informative_acc = train_utils.test_epoch(model_informative,
                                                                                   dloader=val_informative_loader,
                                                                                   loss_fn=loss_fn,
                                                                                   classify=True,
                                                                                   device=device)
                early_stop(val_informative_loss)
                if early_stop.early_stop:
                    print(f'early stop at epoch: {epoch}')
                    break

            if args.wandb:
                wandb.log({'val_informative_loss': val_informative_loss, 'val_informative_acc': val_informative_acc,
                           'train_informative_loss': train_informative_loss,
                           'train_informative_acc': train_informative_acc,
                           'num_training_samples': num_training_samples})

            train_non_informative_loader = pyg.loader.DataLoader(train_non_informative_pyg[:num_training_samples],
                                                                 batch_size=train_batch_size)
            model_non_informative = model_type(in_channels=num_of_features, out_channels=output_dim,
                                               num_layers=n_layers,
                                               hidden_channels=hidden_channels)
            optimizer = optimizer_type(params=model_non_informative.parameters(), lr=lr, weight_decay=wd)
            model_non_informative.to(device)
            early_stop = utils.EarlyStopping(patience=args.patience)
            loss_fn = loss()

            for epoch in range(num_epochs):
                print(f'epoch: {epoch}, num_training_samples: {num_training_samples}, iter: {i}, non-informative graphs')
                train_non_informative_loss, train_non_informative_acc = train_utils.train_epoch(model_non_informative,
                                                                                        dloader=train_non_informative_loader,
                                                                                        loss_fn=loss_fn,
                                                                                        optimizer=optimizer,
                                                                                        classify=True,
                                                                                        device=device)
                val_non_informative_loss, val_non_informative_acc = train_utils.test_epoch(model_non_informative,
                                                                                   dloader=val_non_informative_rcov_loader,
                                                                                   loss_fn=loss_fn,
                                                                                   classify=True,
                                                                                   device=device)
                early_stop(val_non_informative_loss)
                if early_stop.early_stop:
                    print(f'early stop at epoch: {epoch}')
                    break

            if args.wandb:
                wandb.log({'val_non_informative_loss': val_non_informative_loss,
                           'val_non_informative_acc': val_non_informative_acc,
                           'train_non_informative_loss': train_non_informative_loss,
                           'train_non_informative_acc': train_non_informative_acc})

        train_non_informative_rcov_loader = pyg.loader.DataLoader(train_non_informative_rcov_pyg[:num_training_samples], batch_size=train_batch_size)
        model_non_informative_rcov = model_type(in_channels=num_of_features, out_channels=output_dim,
                                                num_layers=n_layers,
                                                hidden_channels=hidden_channels)
        optimizer = optimizer_type(params=model_non_informative_rcov.parameters(), lr=lr, weight_decay=wd)
        model_non_informative_rcov.to(device)
        early_stop = utils.EarlyStopping(patience=args.patience)
        loss_fn = loss()

        for epoch in range(num_epochs):
            train_non_informative_rcov_loss, train_non_informative_rcov_acc = train_utils.train_epoch(model_non_informative_rcov,
                                                                                        dloader=train_non_informative_rcov_loader,
                                                                                        loss_fn=loss_fn,
                                                                                        optimizer=optimizer,
                                                                                        classify=True,
                                                                                        device=device)
            val_non_informative_rcov_loss, val_non_informative_rcov_acc = train_utils.test_epoch(model_non_informative_rcov,
                                                                                   dloader=val_non_informative_rcov_loader,
                                                                                   loss_fn=loss_fn,
                                                                                   classify=True,
                                                                                   device=device)
            early_stop(val_non_informative_rcov_loss)
            if early_stop.early_stop:
                print(f'early stop at epoch: {epoch}')
                break

        if args.wandb:
            wandb.log({'val_non_informative_rcov_loss': val_non_informative_rcov_loss,
                       'val_non_informative_rcov_acc': val_non_informative_rcov_acc,
                       'train_non_informative_rcov_loss': train_non_informative_rcov_loss,
                       'train_non_informative_rcov_acc': train_non_informative_rcov_acc})

        train_empty_graphs_loader = pyg.loader.DataLoader(train_empty_graphs[:num_training_samples],
                                                          batch_size=train_batch_size)
        model_empty_graphs = model_type(in_channels=num_of_features, out_channels=output_dim, num_layers=n_layers,
                                        hidden_channels=hidden_channels)
        optimizer = optimizer_type(params=model_empty_graphs.parameters(), lr=lr, weight_decay=wd)
        model_empty_graphs.to(device)
        early_stop = utils.EarlyStopping(patience=args.patience)
        loss_fn = loss()

        for epoch in range(num_epochs):
            train_empty_graphs_loss, train_empty_graphs_acc = train_utils.train_epoch(model_empty_graphs,
                                                                                        dloader=train_empty_graphs_loader,
                                                                                        loss_fn=loss_fn,
                                                                                        optimizer=optimizer,
                                                                                        classify=True,
                                                                                        device=device)
            val_empty_graphs_loss, val_empty_graphs_acc = train_utils.test_epoch(model_empty_graphs,
                                                                                   dloader=val_empty_graphs_loader,
                                                                                   loss_fn=loss_fn,
                                                                                   classify=True,
                                                                                   device=device)
            early_stop(val_empty_graphs_loss)
            if early_stop.early_stop:
                print(f'early stop at epoch: {epoch}')
                break

        if args.wandb:
            wandb.log({'val_empty_graphs_loss': val_empty_graphs_loss,
                       'val_empty_graphs_acc': val_empty_graphs_acc,
                       'train_empty_graphs_loss': train_empty_graphs_loss,
                       'train_empty_graphs_acc': train_empty_graphs_acc})

        # test
        test_informative_loss, test_informative_acc = train_utils.test_epoch(model_informative,
                                                                             dloader=test_informative_loader,
                                                                             loss_fn=loss_fn,
                                                                             classify=True,
                                                                             device=device)
        test_non_informative_loss, test_non_informative_acc = train_utils.test_epoch(model_non_informative,
                                                                                     dloader=test_non_informative_loader,
                                                                                     loss_fn=loss_fn,
                                                                                     classify=True,
                                                                                     device=device)
        test_non_informative_rcov_loss, test_non_informative_rcov_acc = train_utils.test_epoch(model_non_informative_rcov,
                                                                                               dloader=test_non_informative_rcov_loader,
                                                                                               loss_fn=loss_fn,
                                                                                               classify=True,
                                                                                               device=device)
        test_empty_graphs_loss, test_empty_graphs_acc = train_utils.test_epoch(model_empty_graphs,
                                                                               dloader=test_empty_graphs_loader,
                                                                               loss_fn=loss_fn,
                                                                               classify=True,
                                                                               device=device)
        test_informative_accs.append(test_informative_acc)
        test_non_informative_accs.append(test_non_informative_acc)
        test_non_informative_rcov_accs.append(test_non_informative_rcov_acc)
        test_empty_graphs_accs.append(test_empty_graphs_acc)

        if args.wandb:
            wandb.log({'test_informative_loss': test_informative_loss,
                       'test_informative_acc': test_informative_acc,
                       'test_non_informative_loss': test_non_informative_loss,
                       'test_non_informative_acc': test_non_informative_acc,
                       'test_non_informative_rcov_loss': test_non_informative_rcov_loss,
                       'test_non_informative_rcov_acc': test_non_informative_rcov_acc,
                       'test_empty_graphs_loss': test_empty_graphs_loss,
                       'test_empty_graphs_acc': test_empty_graphs_acc})

if args.wandb:
    wandb.log({'test_informative_acc': np.mean(test_informative_accs),
               'test_non_informative_acc': np.mean(test_non_informative_accs),
               'test_non_informative_rcov_acc': np.mean(test_non_informative_rcov_accs),
               'test_empty_graphs_acc': np.mean(test_empty_graphs_accs),
               'test_informative_std': np.std(test_informative_accs),
               'test_non_informative_std': np.std(test_non_informative_accs),
               'test_non_informative_rcov_std': np.std(test_non_informative_rcov_accs),
               'test_empty_graphs_std': np.std(test_empty_graphs_accs)})
