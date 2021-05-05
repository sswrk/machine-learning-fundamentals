import pandas as pd
import random
import sklearn
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path)
    data.pop('Item')
    data.pop('Serving Size')
    data.pop('Total Fat (% Daily Value)')
    data.pop('Saturated Fat (% Daily Value)')
    data.pop('Cholesterol (% Daily Value)')
    data.pop('Sodium (% Daily Value)')
    data.pop('Carbohydrates (% Daily Value)')
    data.pop('Dietary Fiber (% Daily Value)')

    category_number = {}
    categories = []
    for i, category in enumerate(set(data['Category'])):
        category_number[category] = i
        categories.append(category)

    numbers = []
    for category in data['Category']:
        numbers.append(category_number[category])
    data.pop('Category')

    scaled = preprocessing.MinMaxScaler((0, 1)).fit_transform(data.values)

    return pd.DataFrame(scaled, columns=data.columns), np.array(numbers), categories


def run_k_means(data, init, k):
    k_means = KMeans(n_clusters=k, n_init=1, max_iter=1, init=init)

    previous_iteration_labels = None
    scores = []
    while True:
        k_means.fit(data)

        current_labels = k_means.labels_

        if np.array_equal(previous_iteration_labels, current_labels):
            break

        centroids = k_means.cluster_centers_
        k_means.init = centroids
        previous_iteration_labels = current_labels

        inertia = k_means.inertia_
        score = sklearn.metrics.davies_bouldin_score(data, current_labels)

        scores.append(score)
        print(inertia, score)

    print('\n')

    return scores


def draw_k_means_different_inits(data, k, iterations):
    random.seed(12562)
    init_names = ['k-means++', 'random', 'random_values']
    init_scores = [[] for i in range(len(init_names))]

    max_iterations = 0
    for i, init in enumerate(init_names):
        for j in range(iterations):
            if init == 'random_values':
                scores = run_k_means(data, np.random.uniform(0.0, 1.0, (k, data.shape[1])), k)
            else:
                scores = run_k_means(data, init, k)
            init_scores[i].append(scores)
            max_iterations = max(len(scores), max_iterations)
    for i in range(len(init_names)):
        for j in range(iterations):
            tail_score = init_scores[i][j][-1]
            while len(init_scores[i][j]) < max_iterations:
                init_scores[i][j].append(tail_score)

    mean_scores = []
    standard_errors = []
    for i in range(len(init_names)):
        mean_scores.append(np.mean(np.array(init_scores[i]), axis=0))
        standard_errors.append(np.std(np.array(init_scores[i]), axis=0))
    for init_name, mean, std in zip(init_names, mean_scores, standard_errors):
        plt.errorbar(list(range(1, max_iterations + 1)), mean, std, label=init_name, capsize=4, marker='o')
    plt.xlabel('Iteracje')
    plt.ylabel('Indeks Davida-Bouldina')
    plt.legend()
    plt.show()


def draw_k_means_different_k(data, iterations):
    min_k = 3
    max_k = 20
    mean_scores = []
    standard_errors = []
    for k in list(range(min_k, max_k + 1)):
        scores = []
        for i in range(iterations):
            k_means = KMeans(n_clusters=k, init='k-means++')
            k_means.fit(data)
            score = sklearn.metrics.davies_bouldin_score(data, k_means.labels_)
            scores.append(score)
        mean_scores.append(np.mean(scores))
        standard_errors.append(np.std(scores))
    plt.errorbar(list(range(min_k, max_k + 1)), mean_scores, standard_errors)
    plt.xlabel('k')
    plt.ylabel('Indeks Davida-Bouldina')
    plt.show()


def project_pca(data, labels, k, centers):
    pca = sklearn.decomposition.PCA(2).fit_transform(data)
    c = []
    for i in range(data.shape[0]):
        c.append(list(range(k))[labels[i]])
    plt.scatter(pca[:, 0], pca[:, 1], label=labels, c=c, s=80) if centers \
        else plt.scatter(pca[:, 0], pca[:, 1], label=labels, c=c)


def draw_clusters_pca(values, k, init):
    k_means = KMeans(n_clusters=k, init=init)
    k_means.fit(values)

    project_pca(values, k_means.labels_, k, False)
    project_pca(k_means.cluster_centers_, np.array(range(k)), k, True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def draw_categories_pca(values, k, categories):
    project_pca(values, categories, k, False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    data_frame, category_labels, category_names = load_data('resources/menu.csv')

    draw_k_means_different_inits(data_frame.values, 5, 10)
    draw_k_means_different_k(data_frame.values, 50)

    draw_clusters_pca(data_frame.values, 9, 'k-means++')
    draw_categories_pca(data_frame.values, len(category_names), category_labels)
