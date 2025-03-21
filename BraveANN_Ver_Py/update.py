from utils import compute_tilted_sse, compute_tilted_sse_InEachCluster
import numpy as np
from collections import Counter


def balanced_kmeans(X, labels, centroids, max_iter=300):
    n_clusters = centroids.shape[0]
    for _ in range(max_iter):
        distances_squared = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
        cluster_sizes = np.array([
            np.sum(labels == i) for i in range(n_clusters)
        ])
        lambda_param = calculate_lambda(labels, distances_squared)
        balanced_distances = distances_squared + lambda_param * cluster_sizes
        new_labels = np.argmin(balanced_distances, axis=1)
        labels = new_labels
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(n_clusters)
        ])
        centroids = new_centroids

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    print(f"Balanced KMeans - Cluster sizes: {cluster_sizes}")
    return centroids, labels

def lloyd_heuristic(data, centroids, labels, max_iter):
    n_clusters = centroids.shape[0]

    for _ in range(max_iter):
        # Step 1: Assign each point to the nearest centroid
        distances = np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Step 2: Recalculate centroids
        new_centroids = np.array([data[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centroids[k]
                                  for k in range(n_clusters)])

        labels = new_labels
        centroids = new_centroids

    return centroids, labels


def kmeans_cluster_size_variance(labels):
    """
    Calculate \sum_{j=1}^k (cluster_size - n/k)^2 in K-means clustering

    Parameters:
    labels : list or numpy array, the cluster label of each data point

    Returns:
    float, the calculated variance term
    """
    # Total number of data points
    n = len(labels)
    # Number of clusters
    unique_clusters = set(labels)
    k = len(unique_clusters)

    # Calculate the size of each cluster
    cluster_counts = Counter(labels)

    # Calculate the theoretical mean n/k
    expected_size = n / k

    # Calculate the variance term
    variance = sum((cluster_counts[j] - expected_size) ** 2 for j in unique_clusters)

    return variance


def calculate_lambda(labels, distances):
    # Count the number of samples in each cluster
    label_counts = Counter(labels)

    # Find the cluster with the most points
    most_common_label = label_counts.most_common(1)[0][0]

    # Find all points belonging to the most populated cluster and their distance to the cluster center
    most_common_cluster_distances = []
    for i in range(len(labels)):
        if labels[i] == most_common_label:
            # For each point, choose its minimum distance to all centroids
            most_common_cluster_distances.append(min(distances[i]))

    # Maximum distance d_max
    d_max = max(most_common_cluster_distances)

    # Mean distance d_mean
    d_mean = np.mean(most_common_cluster_distances)

    # Total number of samples n
    n = len(labels)

    # Calculate lambda
    lambda_value = (d_max - d_mean) / n
    lambda0 = 0.01
    lambda_value = min(lambda_value, lambda0)
    return lambda_value


def kmeans_plusplus_init(X, k):
    centers = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centers]) for x in X])
        prob = distances / distances.sum()
        cumulative_prob = prob.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if r < p:
                centers.append(X[j])
                break
    return np.array(centers)

def rkm(X, args, t, k, num_epoch, lr, centroids, labels):
    batch_size = args.num_batch
    max_iters = args.maxIter
    n_samples, n_features = X.shape
    SSE_all = []
    tilted_SSE_all = []
    for _ in range(max_iters):
        for _ in range(num_epoch):
            if t == 0:
                distances_to_centroids = np.exp(t * np.linalg.norm(X - centroids[labels], axis=1)**2)
                cluster_distances_sum = np.zeros((k,))
            else:
                distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1)**2

            phi = np.zeros((k,))
            for j in range(k):
                if t == 0:
                    cluster_distances_sum[j] = np.sum(distances_to_centroids[labels == j])
                else:
                    phi[j] = (np.logaddexp.reduce(t * distances_to_centroids*((labels == j).astype(int))) + np.log(1/n_samples))/t
            if t == 0:
                weights = np.ones([n_samples,1])/n_samples
            else:
                weights = (np.exp(t * (distances_to_centroids - phi[labels]))/n_samples).reshape(n_samples,1)

            batch_indices = np.random.choice(n_samples, batch_size, replace=True)
            batch = X[batch_indices]
            weights_batch = weights[batch_indices]
            distances_batch = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
            labels_batch = np.argmin(distances_batch, axis=1)
            gradients = 2 * (batch - centroids[labels_batch])
            weighted_gradients = np.multiply(np.repeat(weights_batch, gradients.shape[1], axis=1), gradients)
            new_centroids = np.array([weighted_gradients[labels_batch == j].sum(axis=0) for j in range(k)])
            learning_rate = lr
            centroids = centroids + learning_rate * new_centroids

        # compute loss
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        SSE = np.sum((X - centroids[labels]) ** 2)

        if t == 0:
            tilted_SSE = 0
        else:
            tilted_SSE = compute_tilted_sse(X, centroids, labels, k, t, n_samples)

        SSE_all.append(SSE)
        tilted_SSE_all.append(tilted_SSE)

    return centroids, labels, SSE_all, tilted_SSE_all


def Fastrkm(X, args, t, k, num_epoch, lr, centroids, labels, phi):
    batch_size = args.num_batch
    max_iters = args.maxIter
    mu = args.mu
    n_samples, n_features = X.shape
    SSE_all = []
    tilted_SSE_all = []
    for _ in range(max_iters):
        for _ in range(num_epoch):
            batch_indices = np.random.choice(n_samples, batch_size, replace=True)
            batch = X[batch_indices]
            distances_batch = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
            distances_batch_min = np.min(distances_batch, axis=1)
            labels_batch = np.argmin(distances_batch, axis=1)
            gradients = 2 * (batch - centroids[labels_batch])
            phi_batch = compute_tilted_sse_InEachCluster(batch, centroids, labels_batch, k, t)
            for j in range(k):
                phi[j] = 1/t * (np.log((1-mu) * np.exp(t * phi[j]) + mu * np.exp(t * phi_batch[j])))
            weights_batch = (np.exp(t * (distances_batch_min - phi[labels_batch]))/batch_size).reshape(batch_size, 1)
            weighted_gradients = np.multiply(np.repeat(weights_batch, gradients.shape[1], axis=1), gradients)
            new_centroids = np.array([weighted_gradients[labels_batch == j].sum(axis=0) for j in range(k)])
            # update centroids
            learning_rate = lr
            centroids = centroids + learning_rate * new_centroids

        # compute loss
        distances_squared = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)

        cluster_sizes = np.array(
            [np.sum(distances_squared[:, i] == np.min(distances_squared, axis=1)) for i in range(k)])
        labels = np.argmin(distances_squared, axis=1)

        SSE = np.sum((X - centroids[labels]) ** 2)

        if t == 0:
            tilted_SSE = 0
        else:
            tilted_SSE = compute_tilted_sse(X, centroids, labels, k, t, n_samples)
        SSE_all.append(SSE)
        tilted_SSE_all.append(tilted_SSE)
    print(f"Cluster sizes: {cluster_sizes}")

    return centroids, labels, SSE_all, tilted_SSE_all


def Balanced_rkm(X, args, t, k, num_epoch, lr, centroids, labels, phi):
    batch_size = args.num_batch
    max_iters = args.maxIter
    mu = args.mu
    n_samples, n_features = X.shape
    SSE_all = []
    for _ in range(max_iters):
        for _ in range(num_epoch):
            batch_indices = np.random.choice(n_samples, batch_size, replace=True)
            batch = X[batch_indices]
            distances_batch = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
            distances_batch_min = np.min(distances_batch, axis=1)
            labels_batch = labels[batch_indices]
            gradients = 2 * (batch - centroids[labels_batch])
            phi_batch = compute_tilted_sse_InEachCluster(batch, centroids, labels_batch, k, t)
            for j in range(k):
                phi[j] = 1/t * (np.log((1-mu) * np.exp(t * phi[j]) + mu * np.exp(t * phi_batch[j])))
            weights_batch = (np.exp(t * (distances_batch_min - phi[labels_batch]))/batch_size).reshape(batch_size, 1)
            weighted_gradients = np.multiply(np.repeat(weights_batch, gradients.shape[1], axis=1), gradients)
            new_centroids = np.array([weighted_gradients[labels_batch == j].sum(axis=0) for j in range(k)])
            learning_rate = lr
            centroids = centroids + learning_rate * new_centroids

        distances_squared = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
        cluster_sizes = np.array(
            [np.sum(distances_squared[:, i] == np.min(distances_squared, axis=1)) for i in range(k)])
        rho = calculate_lambda(labels, distances_squared)
        balanced_distances = distances_squared + rho * cluster_sizes
        labels = np.argmin(balanced_distances, axis=1)

    print(f"Balanced KMeans - Cluster sizes: {cluster_sizes}")

    return centroids, labels, SSE_all
