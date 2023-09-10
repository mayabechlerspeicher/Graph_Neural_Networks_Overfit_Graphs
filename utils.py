import torch
import torch_geometric as pyg
import networkx as nx
import numpy as np
import random
import models
from numpy.linalg import norm as np_norm
import torch_geometric.utils.convert as convert
from scipy.stats import skew, kurtosis
from itertools import compress
from numpy import dot
from numpy.linalg import norm

wandb_flag = True


def pyg_edge_index_to_numpy_adj(edge_index, num_nodes=None):
    """
    pyg_adj: (num_nodes, num_nodes)
    """
    adj = convert.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).todense()
    return adj


def sample_teacher(d, self: bool = True, top: bool = False):
    self_teacher = np.zeros(shape=(1, d)).flatten()
    top_teacher = np.zeros_like(self_teacher)
    if self:
        self_teacher = sample_self_teacher(d).flatten()
    if top:
        top_teacher = sample_top_teacher(d).flatten()
    teacher = np.concatenate([self_teacher, top_teacher])
    teacher = teacher / np_norm(teacher)
    return teacher


def sample_self_teacher(d):
    teacher = np.random.randn(d, 1).reshape(-1, 1).astype(np.float32)
    return teacher


def sample_top_teacher(d):
    teacher = np.random.randn(d, 1).reshape(-1, 1).astype(np.float32)
    return teacher


def sample_features(num_samples, num_nodes, d, std=1.0):
    features = std * np.random.randn(num_samples, num_nodes, d).astype(np.float32)
    return features


def get_moments(Gs):
    std_mean = 0
    total_mean = 0
    skew_mean = 0
    kurt_mean = 0
    for g in Gs:
        degrees = np.array([d for n, d in g.degree()])
        std_mean += degrees.std()
        total_mean += degrees.mean()
        skew_mean += skew(degrees)
        kurt_mean += kurtosis(degrees)
    return (total_mean / len(Gs)), (std_mean / len(Gs)), (skew_mean / len(Gs)), (kurt_mean / len(Gs))


def get_dif_graphs_dataset(num_samples, num_nodes, num_features, teacher, graph_dist, margin,
                           self_teacher: bool, normalize_features: bool = False, return_margin=False,
                           with_edge_weight=False):
    """
    returns a dataset matching the parameters. graphdist should be an nx distribution (maybe through lambda expression)
    that recieves the number of nodes in the graph and no other params.
    """
    max_num_samples = num_samples
    num_samples = 2 * num_samples  # gen more samples since will throw away some due to margin
    features_std = 1. / num_nodes if normalize_features else 1.0

    generation_batch_size = min([500, num_samples])
    num_batches = num_samples // generation_batch_size + 1
    dataset = []
    graph_stack = []

    for i in range(num_batches):
        if len(dataset) > max_num_samples:
            break
        features = sample_features(generation_batch_size, num_nodes, num_features, std=features_std)
        graphs = []
        edge_indexes = []
        labels = []
        adj_matrices = []
        edge_weights = []
        for i in range(generation_batch_size):
            G = graph_dist(num_nodes=num_nodes)
            graphs.append(G)
            adj_matrices.append(nx.adjacency_matrix(G).toarray())

            edge_index = convert.from_networkx(G).edge_index
            edge_indexes.append(edge_index)
            if with_edge_weight:
                edge_weights.append(torch.ones(size=(edge_index.shape[1],)))

        adj_matrices = np.stack(adj_matrices)
        label = get_labels(features, teacher, adj_matrices)
        labels.append(label)

        labels = np.concatenate(labels)
        labels = torch.from_numpy(labels)

        x = remove_under_margin(features=features, labels=labels, graphs=graphs,
                                edge_indexes=edge_indexes, teacher=teacher,
                                margin=margin, self_teacher=self_teacher,
                                adj_matrices=adj_matrices,
                                return_margin=return_margin, edge_weights=edge_weights)
        if return_margin:
            features, labels, graphs, edge_indexes, edge_weights, margin = x
        else:
            features, labels, graphs, edge_indexes, edge_weights = x

        features = torch.from_numpy(features)
        added_graphs = [pyg.data.Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes) for x, edge_index, y,
                        in
                        zip(features, edge_indexes, labels)]
        if with_edge_weight:
            for i in range(len(added_graphs)):
                added_graphs[i].edge_weight = edge_weights[i]
                if added_graphs[i].edge_weight.shape[0] != added_graphs[i].edge_index.shape[1]:
                    print("edge weight shape: ", added_graphs[i].edge_weight.shape)
                    print("edge index shape: ", added_graphs[i].edge_index.shape)
                    raise Exception("edge weight shape does not match edge index shape")

        dataset = dataset + added_graphs
        graph_stack = graph_stack + graphs

    if len(dataset) > max_num_samples:
        dataset = dataset[:max_num_samples]
        graph_stack = graph_stack[:max_num_samples]
    if len(dataset) < max_num_samples:
        print("dataset size is %d smaller than num_samples" % len(dataset))
        raise Exception("dataset size is smaller than num_samples")

    if return_margin:
        return dataset, graph_stack, margin
    else:
        return dataset, graph_stack


def get_labels(features, teacher, A):
    """
    features: (B, num_nodes, d) - a single graph feature matrix.
    A: (B, num_nodes, num_nodes) - adjacency matrix of the graph.
    """
    # message passing step - node features weighted by degree
    degrees = A.sum(axis=1)

    # pool
    sum_features = features.sum(-2)
    weighted_sum_features = np.expand_dims(degrees, -1) * features
    weighted_sum_features = weighted_sum_features.sum(-2)

    # stack features for SVM-Space
    svm_features = np.concatenate([sum_features, weighted_sum_features], axis=-1)
    labels = np.sign(svm_features @ teacher).flatten()
    return labels


def remove_under_margin(features: np.ndarray, labels, teacher, margin, self_teacher: bool,
                        adj_matrices=None, edge_indexes=None, graphs: list = None, return_margin=False,
                        edge_weights=None):
    """
    Remove samples from "features", "labels", "graphs" whose signed margin w.r.t to "teacher" is < "margin"
    len(features) = num_samples. If 1 graph is used for the whole dataset pass as [graph]
    """
    if not isinstance(graphs, list):
        raise Exception("margin function expecting list of graphs ,if using single graph pass as [graph]")

    teacher = teacher.flatten()
    sum_features = features.sum(axis=-2)
    degrees = adj_matrices.sum(axis=1)
    weighted_sum_features = np.expand_dims(degrees, -1) * features
    weighted_sum_features = weighted_sum_features.sum(axis=-2)
    svm_features = np.concatenate([sum_features, weighted_sum_features], axis=-1)

    gamma = labels * (svm_features @ teacher)
    good_indices = gamma >= margin

    if return_margin:
        eff_margin = gamma[good_indices].min()

    features = features[good_indices]
    labels = labels[good_indices]
    if edge_indexes is not None:
        graphs = list(compress(graphs, good_indices))
        edge_indexes = list(compress(edge_indexes, good_indices))
    if edge_weights is not None and len(edge_weights) > 0:
        edge_weights = list(compress(edge_weights, good_indices))

    if return_margin:
        return features, labels, graphs, edge_indexes, edge_weights, eff_margin
    else:
        return features, labels, graphs, edge_indexes, edge_weights


def regular_graph_decomp(graph, degree_sequence, self_weights, top_weights, r, model=None, readout='mean'):
    """
    Represent the loss of the model over batch represented as a regular graph with degree r.
    Args:
        batch:
        self_weights: model self weights
        top_weights: model topo weights
        num_nodes: number of nodes in the graph
        r: degree of the regular graph
        model: (optional) for validation - measures the discrapency between the model and the decomposition
    Returns:
        self: self component of the decomposition
        top_regular: regular component of the decomposition
        top_delta: delta component of the decomposition
        top_res_angles: angles between top weights and delta features
        norm_delta_node_sum: norm of the delta sum features
    """
    # convert to numpy
    self_weights = self_weights.numpy()
    top_weights = top_weights.numpy()
    x = graph.x.numpy()  # graph features (num_nodes, d)
    num_nodes = x.shape[0]

    delta_degree_sequence = degree_sequence - r

    # compute sum features, delta sum features
    node_sum = x.sum(axis=0)
    if len(delta_degree_sequence) != num_nodes:
        print('bad bad not good')

    delta_node_sum = delta_degree_sequence @ x
    degree_node_sum = degree_sequence @ x
    norm_delta_node_sum = norm(delta_node_sum)
    self = self_weights @ node_sum
    top_degrees = top_weights @ degree_node_sum
    top_regular = r * top_weights @ node_sum
    top_delta = top_weights @ delta_node_sum
    top_res_angle = dot(top_weights, delta_node_sum) / (norm(top_weights) * norm(delta_node_sum))
    if np.isnan(top_res_angle):
        top_res_angle = 0.0

    # if model normalizes the output apply the same normalization to the decomposition
    if readout == 'mean':
        self = self / num_nodes
        top_degrees = top_degrees / num_nodes
        top_regular = top_regular / num_nodes
        top_delta = top_delta / num_nodes

    # validate the decomposition - should be same as model output
    if model is not None:
        with torch.no_grad():
            gt_value = model.forward(graph)
        value = self + top_regular + top_delta
        np.allclose(gt_value, value)
        # torch.allclose(gt_value, value)

    return self, top_degrees, top_regular, top_delta, top_res_angle, norm_delta_node_sum


def eval_regular_approx(model, dataset):
    """
    Evaluate the decomposition of the model outputs over the dataset.
    Args:
        model:
        dataset:
        num_nodes:
        readout: 'mean' or 'sum'
    Returns:
        top_deg_angle_components: angles between top weights and weighted sum features by degree
        norm_degree_sum_components: norm of the weighted sum features by degree
    """
    assert isinstance(model,
                      models.LinearReadoutGraphSage), "model must be LinearReadoutGraphSage to eval regular approximation"

    # convert model to cpu and eval mode
    model.cpu()
    model.eval()
    if hasattr(model, 'readout_method'):
        readout = model.readout_method

    # get model collapsed weights
    with torch.no_grad():
        self_weights = model.conv.lin_r.weight
        top_weights = model.conv.lin_l.weight
        w3 = model.lin.weight

        self_weights = w3 @ self_weights
        top_weights = w3 @ top_weights

    # init variables
    dloader = pyg.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    self_components = np.zeros(shape=(1, len(dataset))).flatten()
    top_degree_components = np.zeros(shape=(1, len(dataset))).flatten()
    top_regular_components = np.zeros_like(self_components) + 100
    top_delta_components = np.zeros_like(self_components) + 100
    decomp_r = np.zeros_like(self_components)
    top_res_angle_components = np.zeros_like(self_components)
    top_deg_angle_components = np.zeros_like(self_components)
    norm_degree_sum_components = np.zeros_like(self_components)
    norm_residual_sum_components = np.zeros_like(self_components)
    norm_regular_sum_components = np.zeros_like(self_components)
    # find best decomposition for every graph over all r's
    for i, graph in enumerate(dloader):

        x = graph.x.numpy()  # graph features (num_nodes, d)
        y = graph.y.numpy()
        num_nodes = x.shape[0]

        # compute the delta degree sequence from graph adjacency matrix & r
        edge_index = graph.edge_index
        G = nx.from_numpy_matrix(pyg_edge_index_to_numpy_adj(edge_index, num_nodes))
        degree_sequence = np.array([d for n, d in G.degree()])
        degree_node_sum = degree_sequence @ x
        norm_degree_node_sum = norm(degree_node_sum)
        nodes_sum = x.sum(axis=0)
        norm_degree_sum_components[i] = norm_degree_node_sum
        if norm(degree_node_sum) == 0:
            top_deg_angle_components[i] = 0
        else:
            top_deg_angle_components[i] = dot(top_weights, degree_node_sum) / (
                    norm(top_weights) * norm(degree_node_sum))

        for r in range(0, num_nodes):
            self, top_degrees, top_regular, top_delta, top_res_angle, norm_delta_sum = regular_graph_decomp(graph,
                                                                                                            degree_sequence,
                                                                                                            self_weights,
                                                                                                            top_weights,
                                                                                                            r,
                                                                                                            model)

            if np.abs(top_delta) < np.abs(top_delta_components[i]):
                top_delta_components[i] = top_delta
                self_components[i] = self
                top_regular_components[i] = top_regular
                decomp_r[i] = r
                top_res_angle_components[i] = top_res_angle
                top_degree_components[i] = top_degrees

                norm_delta_sum_components[i] = norm_delta_sum
                norm_regular_sum_components[i] = norm(r * nodes_sum)

    return self_components, top_degree_components, top_regular_components, top_delta_components, decomp_r, top_res_angle_components, norm_delta_sum_components, top_deg_angle_components, norm_degree_sum_components, norm_regular_sum_components


def get_model_preds(model, dataset):
    model.eval()
    dloader = pyg.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dloader:
            pred = torch.sign(model.forward(batch))
            preds.append(pred.item())
            labels.append(batch.y.item())

    preds = np.array(preds)
    labels = np.array(labels)
    return preds, labels


def get_empty_graphs_from_datasets(data_list):
    """clone the datasets and remove all edges from the graphs"""
    empty_graph_data_list = []
    for graph in data_list:
        graph_copy = graph.clone()
        graph_copy.edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_copy.edge_weight = torch.empty((0, 1), dtype=torch.float)
        empty_graph_data_list.append(graph_copy)
    return empty_graph_data_list


def batch_adges_reduce_ratio_add_random_edges_to_pyg_graphs(pyg_graphs, reduce_goal, batch_size=3):
    nx_graphs = [pyg.utils.to_networkx(graph, to_undirected=True) for graph in pyg_graphs]
    new_pyg_graphs = []
    for i, pyg_graph in enumerate(pyg_graphs):
        graph = pyg_graph.clone()
        nx_graph = nx_graphs[i]
        new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
        curr_ratio = new_std / new_mean
        goal_ratio = reduce_goal
        while curr_ratio > goal_ratio:
            degrees_sorted_tuples = sorted(list(nx_graph.degree), key=lambda x: x[1])
            low_deg_nodes = [x[0] for x in degrees_sorted_tuples[:10]]
            possible_edges = list(nx.non_edges(nx_graph))
            possible_edges_low_deg_nodes = []
            for edge in possible_edges:
                if edge[0] in low_deg_nodes or edge[1] in low_deg_nodes:
                    possible_edges_low_deg_nodes.append(edge)
            if len(possible_edges_low_deg_nodes) == 0:
                new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
                print('num of nodes in graph:' + str(len(nx_graph.nodes)))
                print('new mean:' + str(new_mean))
                print('new std:' + str(new_std))
                print('new skew:' + str(new_skew))
                print('new kurt:' + str(new_kurt))
                print('no possible edges')
                print('num of edges:' + str(len(nx_graph.edges)))
                print(nx_graph.edges)

            new_edges = random.choice(possible_edges_low_deg_nodes, size=batch_size)
            for new_edge in new_edges:
                graph.edge_index = torch.cat(
                    [graph.edge_index, torch.tensor([[new_edge[0], new_edge[1]], [new_edge[1], new_edge[0]]]).T], dim=1)
                graph.edge_weight = torch.cat([graph.edge_weight, torch.tensor([0.5, 0.5])])
                nx_graph.add_edge(*new_edge)
            new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
            curr_ratio = new_std / new_mean
        new_pyg_graphs.append(graph)
    return new_pyg_graphs, nx_graphs


def add_constant_feature_1(pyg_graphs):
    for graph in pyg_graphs:
        if graph.x == None:
            graph.x = torch.ones(graph.num_nodes, 1)
        else:
            graph.x = torch.cat((graph.x, torch.ones(graph.num_nodes, 1)), dim=1)


def set_label_to_num_of_edges(pyg_graphs, threshold=None):
    for graph in pyg_graphs:
        if threshold is None:
            graph.y = torch.tensor(graph.num_edges)
        else:
            graph.y = torch.tensor(1) if graph.num_edges > threshold else torch.tensor(-1)
        graph.y = graph.y.type(torch.LongTensor)


def encode_undirected_graph_in_features_then_linear_transform(pyg_graphs, lin_transform):
    for graph in pyg_graphs:
        num_nodes = graph.num_nodes
        graph_in_features = torch.zeros(num_nodes, num_nodes)
        for edge in graph.edge_index.T:
            graph_in_features[edge[0]][edge[1]] = 1
            graph_in_features[edge[1]][edge[0]] = 1

        final_features = (graph_in_features @ lin_transform).float()
        # append to feature matrix
        graph.x = torch.cat((graph.x, final_features), dim=1)


def add_random_uniform_feature(pyg_graphs):
    for graph in pyg_graphs:
        num_nodes = graph.num_nodes
        random_feautre = torch.rand(num_nodes, 1)
        graph.x = torch.cat((graph.x, random_feautre), dim=1)


def build_new_graphs_from_feature(pyg_graphs, feature_index=-1, diff_threshold=0.3, with_edge_weight=False):
    new_graphs = []
    for graph in pyg_graphs:
        new_graph = graph.clone()
        num_nodes = new_graph.num_nodes
        node_selected_feature = new_graph.x[:, feature_index]
        edge_index = torch.empty((2, 0), dtype=torch.long)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.abs(node_selected_feature[i] - node_selected_feature[j]) < diff_threshold:
                    edge_index = torch.cat((edge_index, torch.tensor([[i], [j]])), dim=1)
        new_graph.edge_index = edge_index
        if with_edge_weight:
            new_graph.edge_weight = torch.ones(new_graph.edge_index.shape[1], )
        new_graphs.append(new_graph)
    return new_graphs


class EarlyStopping:
    def __init__(self, metric_name='loss', patience=100, min_is_better=True):
        self.metric_name = metric_name
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def reset(self):
        self.counter = 0

    def __call__(self, score):
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
