import numpy as np
import matplotlib.pyplot as plt


def plot_cdf(data):
    """
    Display CDF plot
    :param data:
    :return: None, displays CDF
    """
    sorted_data = np.sort(data)[::-1]
    data_cumsum = np.cumsum(sorted_data)
    data_normalized = data_cumsum / data_cumsum[-1]
    # Plot the CDF of eigenvalues
    plt.plot(np.arange(1, len(sorted_data) + 1), data_normalized)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion of Variance')
    plt.title('Cumulative Distribution Function of Eigenvalues')
    plt.show()


def load_data(file_path):
    """
    Loads the data set
    :param file_path: csv file path
    :return: np arrayas of pixels (x) and labels (y)
    """
    # Load data from CSV file
    data_path = open(file_path, 'r')
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    y = data[:, 0].astype(int)  # labels
    x = data[:, 1:].astype(int)  # pixels
    return x, y


def display_samples(data, amount_of_samples, rows, columns, shape):
    """
    Display plot with sample from the data
    :param data:  numpy array with pixels value
    :param amount_of_samples: number of samples to show
    :param rows: number of rows of samples in the plot
    :param columns: number of columns of samples in the plot
    :param shape: size of one of the axis of the plot
    :return: None, display plot of samples
    """
    fig = plt.figure(figsize=(columns * 2, rows * 2))
    for i in range(amount_of_samples):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(data[i].reshape(shape, shape))
    plt.show()


def fit_pca(train_data, new_dimension):
    """
    Fit pca
    :param train_data: numpy array of the data
    :param new_dimension: number of the new demotion to reduce to
    :return: Eigenvectors matrix and eigenvalues matrix
    """
    mu = train_data.mean(axis=0)  # Find the samples mean
    Z = train_data - mu  # Subtract  samples mean from the data
    S = np.matmul(Z.transpose(), Z)  # Compute the scatter matrix
    eigen_v, eigenvectors = np.linalg.eig(S)  # compute eigenvectors and eigenvalues of S
    # Sort the eigenvectors
    sorted_indices = np.argsort(eigen_v)[::-1][:new_dimension]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # build matrix E from the top-new_dimension eigenvectors
    e_matrix = sorted_eigenvectors[:, :new_dimension].transpose().real
    return e_matrix, eigen_v[sorted_indices]


def inverse_pca_transform(X_reduced, eigenvectors, mean):
    """
    Apply inverse transformation on reduced data by PCA
    :param X_reduced: np array of the changed data
    :param eigenvectors:np array of eigenvectors (from PCA)
    :param mean: the mean value of the original data
    :return: recovered data
    """
    # Inverse pca transformation
    x_inverse = np.matmul(X_reduced, eigenvectors)
    # Add samples mean to the data
    x_inverse += mean
    return x_inverse


def apply_pca(data, eigenvectors):
    """
    Apply PCA to reduce the dimension of the data
    :param data: np array of the data
    :param eigenvectors: eigenvectors from PCA
    :return: np array after dimension reduction
    """
    mu = data.mean(axis=0)
    Z = data - mu
    reduced_data = np.matmul(eigenvectors, Z.transpose())
    return reduced_data


def build_distances_matrix(train_data, test_data, batch_size=2000):
    """
    Builds distance matrices for KNN
    :param train_data: np array of trained data
    :param test_data: np array of test data
    :param batch_size: the size of a batch to perform the distance calculation
    :return: distances matrix between train and test data
    """
    # Calculate the distance
    distances_matrix = []
    for i in range(0, len(test_data), batch_size):  # Loop for each batch
        batch = test_data[i:i + batch_size]
        # Calculate distances for the batch with euclidean distance
        batch_distances = np.sqrt(np.sum((batch[:, np.newaxis] - train_data) ** 2, axis=2))
        distances_matrix.append(batch_distances)
    distances_matrix = np.concatenate(distances_matrix)
    return distances_matrix


def knn_classifier(distances_matrix, k, Y_train):
    """
    KNN classifier
    :param distances_matrix: distances matrix between train and test data
    :param k: number of nearest neighbors
    :param Y_train: labels of train data
    :return: array with predictions
    """
    # Indices of knn for distance_matrix (all test samples)
    nearest_neighbors_indices = np.argsort(distances_matrix, axis=1)[:, :k]
    # Get labels of the k nearest neighbors
    nearest_y = Y_train[nearest_neighbors_indices]
    # Get the label of the majority from k nearest neighbors
    predictions_array = np.array([np.bincount(labels).argmax() for labels in nearest_y])
    return predictions_array


# load data
x_train, y_train = load_data("fashion-mnist_train.csv")
x_test, y_test = load_data("fashion-mnist_test.csv")

# display_samples(data=x_train, amount_of_samples=1, rows=1, columns=1, shape=28)
# E, eigenvalues = fit_pca(x_train, 81)
# reduced_x_train = apply_pca(x_train, E)
# plot_cdf(eigenvalues)
# reduced_x_train = reduced_x_train.transpose()
# display_samples(data=reduced_x_train, amount_of_samples=1, rows=1, columns=1, shape=9)
# inverse_images = inverse_pca_transform(reduced_x_train, E, np.mean(x_train, axis=0))
# display_samples(data=inverse_images, amount_of_samples=1, rows=1, columns=1, shape=28)
# display_samples(data=x_train, amount_of_samples=1, rows=1, columns=1, shape=28)

# fit pca and apply on x_train
E, eigenvalues = fit_pca(x_train, 36)
reduced_x_train = apply_pca(x_train, E)
reduced_x_train = reduced_x_train.transpose()

# plot_cdf(eigenvalues)
# display_samples(data=reduced_x_train, amount_of_samples=1, rows=1, columns=1, shape=6)
# inverse_images = inverse_pca_transform(reduced_x_train, E, np.mean(x_train, axis=0))
# display_samples(data=inverse_images, amount_of_samples=1, rows=1, columns=1, shape=28)

# Apply PCA on test data
reduced_x_test = apply_pca(x_test, E)
reduced_x_test = reduced_x_test.transpose()

# Classify with KNN
distance_matrix = build_distances_matrix(train_data=reduced_x_train, test_data=reduced_x_test)
predictions = knn_classifier(distances_matrix=distance_matrix, k=9, Y_train=y_train)

# Calculate model accuracy
accuracy = np.sum(predictions == y_test)
accuracy /= len(predictions)

print(f"Test accuracy is: {accuracy * 100}%")
