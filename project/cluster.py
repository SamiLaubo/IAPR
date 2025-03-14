# Input: list of feature vectors for each puzzle piece, list of the images in the same order
# Output: list of list where the inner list is the images in the same puzzles, list of outliers
import numpy as np

def _init_clusters(data,K):
    """
    Initialize centers randomly.

    Parameters:
    data: (NxD) array where N amount of data and D amount of features
    K: (int) Amount of clusters

    Output:
    cluster_centers: (KxD) position of the random clusters
    """
    # Create K random indexes in the range of the data
    N = data.shape[0]
    random_idxs = np.random.permutation(range(N))[:K]

    # Pick K random samples as initial centers
    centers = data[random_idxs]

    return centers

def _std_of_clusters(data, cluster_assignments,cluster_centers):
    """
    Calculate the standard deviation of each cluster.

    Parameters:
    data: (NxD) array where N amount of data and D amount of features
    cluster_assignments: (Nx1) array of which cluster the data is assigned
    cluster_centers: (KxD) position of the centers

    Output:
    std_clusters: (Kx1) standard deviation of the clusters
    """
    std_clusters = np.zeros((cluster_centers.shape[0],))

    for center_id, center in enumerate(cluster_centers):

        # Get all datapoints belonging to the cluster
        mask = cluster_assignments == center_id
        points = data[mask]
        N = points.shape[0]

        # Calculate the standard deviation
        dist = abs(points - center)**2
        std = np.sqrt(dist.sum()/N)
        std_clusters[center_id] = std
    
    return std_clusters


def _split_cluster(data, cluster_assignments,cluster_centers, std_dev, std_thres, max_splits):
    """
    Split centers with high intra class standard deviation.
    
    Parameters:
    data: (NxD) array where N amount of data and D amount of features
    cluster_assignments: (Nx1) array of which cluster the data is assigned
    cluster_centers: (KxD) position of the centers
    std_dev: (Kx1) standard deviation for the clusters
    std_thres: (float) threshold for splitting a cluster

    Output:
    new_cluster_centers: (K2xD) new position of the centers
    """
    new_cluster_centers = cluster_centers.copy()
    splits = 0

    for center_id, center in enumerate(cluster_centers):
        if splits >= max_splits:
            break
        # Check if the standard deviation of this cluster is large
        if std_dev[center_id] > std_thres:
            splits +=1
            # Split the cluster
            # Find datapoint with highest distance from the center
            mask = cluster_assignments == center_id
            points = data[mask]
            dists = _compute_distance(points,center.reshape(1,-1))
            most_dst_point = points[np.argmax(dists)].reshape(1,-1)

            # Add new centerpoint as the most distant point in the cluster
            new_cluster_centers = np.append(new_cluster_centers,most_dst_point, axis=0)

    return new_cluster_centers

def _merge_clusters(cluster_centers, dist_thres, max_merged, min_clusters):
    """
    Merge clusters with short inter class distance.
    
    Parameters:
    cluster_centers: (KxD) position of the centers
    dist_thres: (float) minimum distance between centers
    max_merged: (int) max amount of cluster center merges per iteration
    min_clusters: (int) minimum amount of clusters alowed by the algorithm

    Output:
    new_cluster_centers: (K2xD) new position of the centers
    """
    new_cluster_centers = cluster_centers.copy()
    merged_centers = []

    for id, center in enumerate(cluster_centers):
        # Check if max amount of merges per iteration is reached
        if len(merged_centers)/2 >= max_merged:
            break
        # Check if the minimal amount of centers are reached
        if new_cluster_centers.shape[0] <= min_clusters:
            break

        # Check if center already merged
        if id in merged_centers:
            continue

        # Calculate distance to all other centers
        center_dists = _compute_distance(cluster_centers,center.reshape(1,-1))
        # 'Remove' the distance to itself
        center_dists[center_dists == 0] = 1000.0

        # Check if the distance between this center and the closest other center is small
        if (np.min(center_dists) > dist_thres):
            # Distance suffiecently large
            continue

        center_close_id = np.argmin(center_dists)
        # Check if this center is already merged
        if center_close_id in merged_centers:
            continue

        # Add both centers to the merged list
        merged_centers.append(id)
        merged_centers.append(center_close_id)

        # Replace the two close centers with a new center as the mean of the two
        center_close = cluster_centers[center_close_id]
        center_mean = (center_close + center)/2
        new_cluster_centers = np.append(new_cluster_centers, center_mean.reshape(1,-1), axis=0)
        # Get these centers id in the new center list

    # Remove all merged centers
    keep_inicies = [i for i in range(new_cluster_centers.shape[0]) if i not in merged_centers]
    new_cluster_centers = new_cluster_centers[keep_inicies]

    return new_cluster_centers


def _compute_distance(data,cluster_centers):
    """
    Compute the euclidean distance between each datapoint and each center.
    
    Arguments:

        data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
        centers: array of shape (K, D), centers of the K clusters.
    Returns:
        distances: array of shape (N, K) with the distances between the N points and the K clusters.
    """

    N = data.shape[0]
    K = cluster_centers.shape[0]
        
    distances = np.zeros((N, K))
    
    for k in range(K):
        # Compute the euclidean distance for each data point to each center
        center = cluster_centers[k]
        distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))

    return distances


def _find_closest_cluster(distances):
    """
    Assign datapoints to the closest clusters.
    
    Arguments:
        distances: array of shape (N, K), the distance of each data point to each cluster center.
    Returns:
        cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
    """

    cluster_assignments = np.argmin(distances, axis=1)

    return cluster_assignments

def _compute_centers(K,data,cluster_assignments):
    """
    Compute the center of each cluster based on the assigned points.

    Arguments:
        K: (int) number of clusters
        data: data array of shape (N,D), where N is the number of samples, D is number of features
        cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
    Returns:
        centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
    """

    D = data.shape[1]

    centers = np.zeros((K,D))
    
    # Loop through the centers and calculate the mean for the cluster
    for cluster_center in range(K):

        # Get number of points assigned to this cluster
        Nk = np.sum(cluster_assignments == cluster_center)
        
        # Get the indicies of the points assigned to the cluster
        binary_indicis = cluster_assignments == cluster_center
        
        # Get the points assigned to the cluster
        points_assigned = data[binary_indicis,:]
        
        # Calculate the new center based on the assigned points
        centers[cluster_center] = 1/Nk * np.sum(points_assigned, axis = 0)
    
    return centers


def Kmeans_ISODATA(data, K_init, std_thres, dist_thres, max_iter = 100, max_merged = 2, min_clusters = 3, max_splits = 2, verbose= False):
    """
    K-means ISODATA algorithm for unsupervised clustering.

    Parameters:
    data: (NxD) array where N amount of data and D amount of features
    K_init: (int) number of cluster centers to assign the data from beginning
    std_thres: (float) threshold for splitting a cluster based on the standard deviation within it
    dist_thres: (float) threshold for merging two clusters based on the distance between the centers
    max_iter: (int) max amount of iterations of the algorithm
    max_merged: (int) max amount of cluster center merges per iteration
    min_clusters: (int) minimum amount of clusters alowed by the algorithm

    Output:
    cluster assignments: (Nx1) array of which center the sample is assigned
    cluster_centers: (KxD) array of the position of the centers
    """

    K = K_init

    # Initialize clusters randomly
    cluster_centers = _init_clusters(data,K_init)

    # Iterative loop of ISODATA K-means
    i = 1
    while (i < max_iter):

        i += 1
        # Copy centers for comparison
        old_centers = cluster_centers.copy()

        # Calculate the distance from each datapoint to all cluster centers
        distances = _compute_distance(data, cluster_centers)

        # Calculate which cluster center the datapoints belong to
        cluster_assignments = _find_closest_cluster(distances)

        # Compute the new centers
        cluster_centers = _compute_centers(K, data,cluster_assignments)

        # Merge centers with short distance
        merge_cluster_centers = _merge_clusters(cluster_centers, dist_thres, max_merged, min_clusters)

        # If some centers where merged, update the assignments and clusters once
        # This is done because we want new cluster assignments before splitting
        if merge_cluster_centers.shape[0] != cluster_centers.shape[0]:
            # Calculate the distance from each datapoint to all cluster centers
            distances = _compute_distance(data, merge_cluster_centers)
            #  Update K
            K = merge_cluster_centers.shape[0]
            # Calculate which cluster center the datapoints belong to
            cluster_assignments = _find_closest_cluster(distances)

            # Compute the new centers
            cluster_centers = _compute_centers(K, data,cluster_assignments)

        # Compute standard deviation whitin each cluster
        std_dev = _std_of_clusters(data,cluster_assignments,cluster_centers)

        # Split the clusters that have high standard deviation
        cluster_centers = _split_cluster(data, cluster_assignments, cluster_centers, std_dev, std_thres, max_splits)

        #  Update K
        K = cluster_centers.shape[0]

        # End of the algorithm if the centers have not moved
        if np.all(cluster_centers == old_centers):
            print(f"K-Means has converged after {i} iterations!")
            break

    # Compute the final cluster assignments
        
    # Distances from data to the centers
    distances = _compute_distance(data,cluster_centers)
        
    cluster_assignments = _find_closest_cluster(distances)

    # Compute the new centers
    cluster_centers = _compute_centers(K, data,cluster_assignments)

    # For diagnosing which thresholds to use as hyperparameters of the algorithm
    if verbose:
        for id, center in enumerate(cluster_centers):
            print(f'Final center: {id}')
            # Calculate distance to all other centers
            center_dists = _compute_distance(cluster_centers,center.reshape(1,-1))
            # 'Remove' the distance to itself
            center_dists[center_dists == 0] = 1000.0

            print(f'Final centers closest distance to another center: {np.min(center_dists)}')

            # Get all datapoints belonging to the cluster
            mask = cluster_assignments == id
            points = data[mask]
            N = points.shape[0]

            # Calculate the standard deviation
            dist = abs(points - center)**2
            std = np.sqrt(dist.sum()/N)

            print(f'Final cluster standard deviation: {std}')

    return cluster_assignments, cluster_centers

def arrange_pieces(puzzle_pieces, pieces_assignments):
    """
    Arrange the puzzle images according to their cluster assignment.

    Parameters:
    puzzle_pieces: (list) of piece images
    pieces_assignments: (N) array of the image assignments

    Output:
    arranged_puzzles: list of list of arranged puzzles images
    outliers: list of outlier images
    """
    arranged_puzzles = []
    outliers = []

    for cluster in range(np.max(pieces_assignments) + 1):
        ids = pieces_assignments == cluster
        pieces = np.array(puzzle_pieces)[ids]
        pieces_list = []
        [pieces_list.append(image) for image in pieces]
        if len(pieces_list) >= 9:
            arranged_puzzles.append(pieces_list)
        else:
            outliers.extend(pieces_list)

    return arranged_puzzles, outliers


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    # Create test dataset
    X, _ = make_blobs(
    n_samples=150, n_features=2,
    centers=5, cluster_std=0.5,
    shuffle=True
    )

    # plot of data
    plt.scatter(
    X[:, 0], X[:, 1],
    c='white', marker='o',
    edgecolor='black', s=50,
    label='Data'
    )
    plt.show()

    # Hyperparameters for the K-means ISODATA algorithm
    # Initial number of clusters
    K_init = 10
    # Threshold for standard deviation, if a clusters standard deviation is higher than this value, we split the cluster
    std_thres = 1.2

    # Threshold for distance between cluster centers, if the distance is lower than this value, we merge these centers
    dist_thres = 1.5

    # Max iteration of K-means to converge
    max_iter = 200
    # Maximum amount of merges per iteration, useful for bad initialization when the model does not converge
    max_merged = 2
    # Minimum amount of cluster centers of the algorithm
    min_clusters = 3
    # maximum amount of cluster splits per iteration
    max_splits = 2

    cluster_assignments, centers = Kmeans_ISODATA(X ,K_init, std_thres, dist_thres, max_iter, max_merged, min_clusters, max_splits)

    print(f'Shape of final centers and assignments: {centers.shape}, {cluster_assignments.shape}')

    # Plot the centers with their attributed data
    for id, _ in enumerate(centers):
        data = X[cluster_assignments == id]
        plt.scatter(
        data[:, 0], data[:, 1],
        marker='o',
        edgecolor='black', s=50,
        label='Center: ' + str(id)
    )
        
    plt.scatter(
    centers[:, 0], centers[:, 1],
    c='blue', marker='*',
    edgecolor='black', s=50,
    label='Centers'
    )
    plt.legend()
    plt.show()


