import torch
import torch_geometric as pyg
import networkx as nx
import utils as exp_utils

CREATE_NUM_SAMPLES = 20000


class Dataset():
    """Dummy class for a dataset to hold some fields with info"""
    def __init__(self, dataset_name, dataset, num_features, num_classes, num_nodes, num_samples,
                 max_el=None, one_hot=False):

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.max_el = max_el
        self.one_hot = one_hot

        self.trainset = None
        self.testset = None

    def config_dict(self):
        return {'dataset_name': self.dataset_name,
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_nodes': self.num_nodes,
                'num_samples': self.num_samples}


    def _train_test_split(self, train_ratio=0.8, num_train_samples=None):
        """split the dataset into train and test - split is linear, assigns first elements
        to train and the rest to test"""

        train_size = int(train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size

        if num_train_samples is not None:
            train_size = num_train_samples
            test_size = max([int(num_train_samples * (1 - train_ratio)), 1000])

        self.trainset = self.dataset[:train_size]
        self.testset = self.dataset[train_size:train_size + test_size]

    def augment_dataset_with_graphs(self, num_graphs_per_sample, graph_dist):
        """augment a pyg dataset with graphs from a given distribution
        dataset: pyg dataset
        num_graphs_per_sample: number of graphs to augment per sample
        graph_dist: graph distribution to sample from - should except an argument of num_nodes
        """

        augmented_dataset = []
        for k in range(num_graphs_per_sample):
            augmented_sample = exp_utils.get_graph_dataset_from_given_features_and_labels_optimized(graph_dist,
                                                                                                    self.dataset)
            augmented_dataset += augmented_sample

        self.dataset = augmented_dataset


def create_pyg_data_obj(x, edge_index, label):
    return pyg.data.Data(x=x, edge_index=edge_index, y=label)


def cast_labels2binary(sample):
    if sample.y == 1:
        sample.y = torch.tensor([1], dtype=torch.long)
    elif sample.y == -1:
        sample.y = torch.tensor([0], dtype=torch.long)
    return sample


def get_dataset(dataset_name, num_samples, num_nodes, train_ratio=0.8, **kwargs):
    """
    Get the desired dataset
    Args:
        dataset_name:
        root:
        num_samples:
        pre_transforms:
        test:
    Returns: trainset, testset
    """
    num_samples_to_generate = num_samples

    if dataset_name == 'sum_binary_cls':
        assert 'num_features' in kwargs.keys(), 'num_features must be specified for sum_binary_cls'
        dataset = sum_binary_cls(num_samples=num_samples_to_generate,
                                 num_features=kwargs['num_features'],
                                 num_nodes=num_nodes)

    dataset._train_test_split(train_ratio=train_ratio, num_train_samples=num_samples)
    return dataset


def sum_binary_cls(num_samples, num_features, num_nodes, teacher=None, margin=0.1):
    """
    binary classification over sum of elements in set (graph_
    Args:
        num_samples: number of samples
        num_features: number of features per node
        num_nodes: number of nodes
        teacher: optional, teacher acting as linear classifier randomly drawn otherwise
        margin: optional, desired margin between classes, defauly 0.1

    Returns: pyg dataset
    """

    # set proxy variables for dataset generator
    graph_dist = lambda num_nodes: nx.empty_graph(num_nodes)
    if teacher is None:
        teacher = exp_utils.sample_teacher(d=num_features, self=True, top=False)

    dataset, _ = exp_utils.get_dif_graphs_dataset(num_samples=num_samples,
                                                  num_features=num_features,
                                                  num_nodes=num_nodes,
                                                  graph_dist=graph_dist,
                                                  teacher=teacher,
                                                  margin=margin,
                                                  normalize_features=False,
                                                  self_teacher=True)
    dataset = list(map(cast_labels2binary, dataset))

    dataset = Dataset(dataset=dataset,
                      dataset_name='sum_binary_cls',
                      num_features=num_features,
                      num_classes=2,
                      num_nodes=num_nodes,
                      num_samples=num_samples)

    return dataset