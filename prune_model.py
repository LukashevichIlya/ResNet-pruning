import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from pruning_utils import make_idx_dict, get_layer_from_idx, set_layer_to_idx


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class NaiveClusterModel:
    def __init__(self, model, args):
        self.model = model
        self.n_clusters_per_layer = args['n_clusters_per_layer']

        self.compressed_model = None
        self.idx_dict = {}

    def cluster_model(self):
        # create compressed model and dictionary for the layers of the model
        self.compressed_model = copy.deepcopy(self.model)
        _, self.idx_dict = make_idx_dict(self.compressed_model, -1, [], {})

        # perform clustering and pruning
        for layer_idx in sorted(self.n_clusters_per_layer.keys()):
            self.cluster_layer(layer_idx, self.n_clusters_per_layer[layer_idx])

        return self.compressed_model

    def cluster_layer(self, layer_idx, n_clusters):

        # get current layer
        layer = get_layer_from_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx)

        # get features for the layer
        if isinstance(layer, nn.Conv2d):
            features = self.get_features(layer)

        # get indices of features, which are assigned to one cluster, and cluster centers
        cluster_idxs, cluster_centers = self.get_clustered_idx(features, n_clusters)

        # update layer
        updated_layer = self.update_layer(layer, cluster_idxs, cluster_centers)

        # update compressed model
        set_layer_to_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx, updated_layer)

    def get_features(self, layer):
        features = []
        weights = layer.weight.data

        if isinstance(layer, nn.Conv2d):
            features = self.get_channel_features(weights)

        return features

    def get_channel_features(self, weights):
        # get shapes of convolutional layer
        out_channels, in_channels, kernel_height, kernel_width = weights.shape

        return weights.reshape(out_channels, in_channels * kernel_height * kernel_width)

    def get_clustered_idx(self, features, n_clusters):

        # apply KMeans clustering, get labels of clusters and cluster centers
        kmeans = KMeans(n_clusters=n_clusters).fit(features.cpu())
        clusters = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # create dictionary where the key is a cluster number
        # and a value is a list with indices of features from this cluster
        merged_idx_dict = {}
        for idx, cluster in enumerate(clusters):
            try:
                merged_idx_dict[cluster].append(idx)
            except:
                merged_idx_dict[cluster] = [idx]

        # create a list of lists with indices of features in one cluster
        order = np.arange(n_clusters)
        merged_idxs = [merged_idx_dict[x] for x in order]
        return merged_idxs, cluster_centers

    def update_layer(self, layer, cluster_idxs, cluster_centers):
        weights = layer.weight.data
        out_channels, in_channels, kernel_height, kernel_width = weights.shape

        for idx, cluster_idx in enumerate(cluster_idxs):
            weights[cluster_idx, :, :, :] = torch.tensor(cluster_centers[idx], dtype=torch.float32).reshape(in_channels,
                                                                                                            kernel_height,
                                                                                                            kernel_width).cuda()
        layer.weight.data = weights
        return layer


class ArchModClusterModel:
    def __init__(self, model, args):
        self.model = model
        self.linkage_method = args['linkage_method']
        self.distance_metric = args['distance_metric']
        self.n_clusters_per_layer = args['n_clusters_per_layer']
        self.reshape_exists = args['reshape_exists']
        self.conv_feature_size = args['conv_feature_size']

        self.compressed_model = None
        self.idx_dict = {}

    def cluster_model(self):
        # create compressed model and dictionary for the layers of the model
        self.compressed_model = copy.deepcopy(self.model)
        _, self.idx_dict = make_idx_dict(self.compressed_model, -1, [], {})

        # perform clustering and pruning
        for layer_idx in sorted(self.n_clusters_per_layer.keys()):
            self.cluster_layer(layer_idx, self.n_clusters_per_layer[layer_idx])

        return self.compressed_model

    def cluster_layer(self, layer_idx, n_clusters):

        # get the current layer, the next non-batchnorm layer,
        # batchnorm layer if it exists
        layer = get_layer_from_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx)
        next_layer = None
        next_layer_idx = layer_idx
        batchnorm_idx, batchnorm_layer = None, None
        while not (isinstance(next_layer, nn.Linear) or isinstance(next_layer, nn.Conv2d)):
            next_layer_idx += 1
            next_layer = get_layer_from_idx(self.compressed_model, copy.deepcopy(self.idx_dict), next_layer_idx)

            if isinstance(next_layer, nn.BatchNorm2d):
                batchnorm_idx = next_layer_idx
                batchnorm_layer = next_layer

        # get features for the layer
        if isinstance(layer, nn.Conv2d):
            features = self.get_features(layer)

        # get indices of features, which are assigned to one cluster
        merged_idx = self.get_clustered_idx(features, n_clusters)

        # merge clusters
        merged_layer, merged_next_layer, pruned_batchnorm_layer, reshape_info = self.merge_clusters(features,
                                                                                                    merged_idx, layer,
                                                                                                    layer_idx,
                                                                                                    next_layer,
                                                                                                    batchnorm_layer)

        # update compressed model
        set_layer_to_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx, merged_layer)

        if self.reshape_exists and reshape_info is not None:
            set_layer_to_idx(self.compressed_model, copy.deepcopy(self.idx_dict), reshape_info[0], reshape_info[1])
        if pruned_batchnorm_layer is not None:
            set_layer_to_idx(self.compressed_model, copy.deepcopy(self.idx_dict), batchnorm_idx, pruned_batchnorm_layer)

        set_layer_to_idx(self.compressed_model, copy.deepcopy(self.idx_dict), next_layer_idx, merged_next_layer)

    def get_features(self, layer):
        features = []
        weights = layer.weight.data

        if isinstance(layer, nn.Conv2d):
            features = self.get_channel_features(weights)

        return features

    def get_channel_features(self, weights):
        # get shapes of convolutional layer
        out_channels, in_channels, kernel_height, kernel_width = weights.shape

        # Compute channel-wise Frobenius norm of a 3D tensor (as in original paper)
        features = torch.norm(weights.reshape(out_channels, in_channels, kernel_height * kernel_width), dim=2)
        return features

    def get_clustered_idx(self, features, n_clusters):

        # apply  agglomerative hierarchical clustering with maximum number of clusters equal to n_clusters
        Z = linkage(y=features.cpu(), method=self.linkage_method, metric=self.distance_metric)
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        c, coph_dists = cophenet(Z, pdist(features.cpu()))

        # create dictionary where the key is a cluster number
        # and a value is a list with indices of features from this cluster
        merged_idx_dict = {}
        for idx, cluster in enumerate(clusters):
            try:
                merged_idx_dict[cluster].append(idx)
            except:
                merged_idx_dict[cluster] = [idx]

        # create a list of lists with indices of features in one cluster
        merged_idx = [value for key, value in merged_idx_dict.items()]
        return merged_idx

    def merge_clusters(self, features, merged_idx, layer, layer_idx, next_layer, batchnorm_layer):

        # get initial weights of a layer
        W1 = layer.weight.data
        W2, B2 = next_layer.weight.data, next_layer.bias.data if next_layer.bias is not None else None
        reshape_info = None

        # perform norm based subset selection (cluster representative contains a single filter)
        idx_for_cluster = [idx[torch.argmax(torch.norm(features[idx, :], dim=1))] for idx in merged_idx]
        W1_merged = W1[idx_for_cluster, :, :, :]

        # create a new merged layer
        merged_layer = nn.Conv2d(in_channels=W1_merged.shape[1], out_channels=W1_merged.shape[0],
                                 kernel_size=W1_merged.shape[2], stride=layer.stride,
                                 padding=layer.padding, bias=False)

        # process the next layer
        if isinstance(next_layer, nn.Conv2d):
            W2_merged = W2[:, idx_for_cluster, :, :]
            merged_next_layer = nn.Conv2d(in_channels=W2_merged.shape[1], out_channels=W2_merged.shape[0],
                                          kernel_size=W2_merged.shape[2], stride=next_layer.stride,
                                          padding=next_layer.padding, bias=False)

        elif isinstance(next_layer, nn.Linear):
            fm_window = self.conv_feature_size * self.conv_feature_size
            W2_merged = torch.cat(
                [torch.stack([W2[:, j] for j in range(f * fm_window, (f + 1) * fm_window)]) for f in idx_for_cluster])
            W2_merged = torch.t(W2_merged)
            merged_next_layer = nn.Linear(in_features=W2_merged.shape[1], out_features=W2_merged.shape[0])

            if self.reshape_exists:
                reshape_layer = get_layer_from_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx)
                # while not(isinstance(reshape_layer, Reshape)):
                #     layer_idx += 1
                #     reshape_layer = get_layer_from_idx(self.compressed_model, copy.deepcopy(self.idx_dict), layer_idx)
                reshape_layer = Reshape(-1, fm_window * len(idx_for_cluster))
                reshape_info = (layer_idx + 1, reshape_layer)

        # put values into new layers
        print(W1_merged.shape, W2_merged.shape)
        merged_layer.weight.data = W1_merged
        merged_next_layer.weight.data = W2_merged

        if B2 is not None:
            merged_next_layer.bias.data = B2

        # edit batchnorm layer, if it is next to the current layer
        pruned_batchnorm_layer = None
        if batchnorm_layer:
            pruned_batchnorm_layer = nn.BatchNorm2d(num_features=len(idx_for_cluster))
            pruned_batchnorm_layer.weight.data = batchnorm_layer.weight.data[idx_for_cluster]
            pruned_batchnorm_layer.bias.data = batchnorm_layer.bias.data[idx_for_cluster]
            pruned_batchnorm_layer.running_mean.data = batchnorm_layer.running_mean.data[idx_for_cluster]
            pruned_batchnorm_layer.running_var.data = batchnorm_layer.running_var.data[idx_for_cluster]

        return merged_layer, merged_next_layer, pruned_batchnorm_layer, reshape_info
