import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sift', choices=['sift', 'mnist', 'femnist', 'gist'], help="dataset name")
    parser.add_argument('--num_clusters', type=list, default=[3, 4], help="number of clusters, k in our paper")
    parser.add_argument('--num_subsample', type=int, default=1, help="subsample number of the full dataset")
    parser.add_argument('--sample_size', type=int, default=1000, help="subsample size of the dataset")
    parser.add_argument('--noise_frac', type=float, default=[0], help='μ in our paper')
    parser.add_argument('--mu', type=float, default=0.2, help='μ in our paper')
    parser.add_argument('--init', type=str, default='random', choices=['kmeans++', 'random'], help="initialization method")
    parser.add_argument('--solver', type=str, default='fast_rkm', choices=['rkm', 'fast_rkm', 'balanced_rkm'], help="solver name")
    parser.add_argument('--t', type=list, default=[-0.05], choices=[-0.01, -0.02, -0.05, -0.1, -0.2, -0.5],help="t in our paper, when t=0, tkm generalize to kmeans via sgd")
    parser.add_argument('--maxIter', type=int, default=100, help="maximum iterations")
    parser.add_argument('--epoch_list', type=list, default=[1], choices=[1,3,5], help="number of local epochs: E")
    parser.add_argument('--num_batch', type=int, default=500, help="local batch size: B")
    parser.add_argument('--lr_list', type=float, default=[0.1], choices=[0.01, 0.05, 0.1, 0.2, 0.5, 1],help='learning rate')
    parser.add_argument('--seeds', type=list, default=[0, 1, 2, 3], help="random seed")
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    args = parser.parse_args()
    return args