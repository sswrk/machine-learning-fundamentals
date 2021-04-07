import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

colors = ['red', 'blue', 'green']


def generate_data():
    samples = 1000
    dimensions = 2

    red = make_blobs(centers=[[.3, 1.5], [.8, 1.0]], cluster_std=[.2, .2], n_samples=samples, n_features=dimensions,
                     random_state=0)[0]
    blue = make_blobs(centers=[[1.2, .7], [1.6, .4]], cluster_std=[.2, .15], n_samples=samples, n_features=dimensions,
                      random_state=0)[0]
    green = make_blobs(centers=[[.5, .7], [1.0, .5]], cluster_std=[.2, .2], n_samples=samples, n_features=dimensions,
                       random_state=0)[0]

    return [red, blue, green]


def draw_data(points):
    for point, color in zip(points, colors):
        plt.scatter(point[:, 0], point[:, 1], c=color, marker='.')
    plt.show()


def draw_partitions(X, y):
    colors_partitions = ['coral', 'lightblue', 'lightgreen']

    classifiers = [
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=13),
        KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params={'V': np.cov(X.T)}),
        KNeighborsClassifier(n_neighbors=9, weights='distance')]

    classifiers_names = [
        'k=1, głosowanie większościowe, metryka Euklidesa',
        'k=13, głosowanie większościowe, metryką Euklidesa',
        'k=1, głosowanie większościowe, metryka Mahalanobisa',
        'k=9, głosowanie ważone odległością, metryka Euklidesa',
    ]

    plt.figure(figsize=(27, 9))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .005),
                         np.arange(y_min, y_max, .005))

    i = 1
    for name, classifier in zip(classifiers_names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, i)
        classifier.fit(X, y)
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.pcolormesh(xx, yy, Z, cmap=ListedColormap(colors_partitions), shading='auto')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), edgecolors='k', alpha=.05)

        ax.set_title(name)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

    plt.tight_layout()
    plt.show()


names = [
    'Gł. większościowe m. Euklidesa',
    'Gł. większościowe m. Mahalanobisa'
]


def test_classifier(X, y, classifier_name):
    repeats = 20

    def classify(X, y, classifier):
        classify_repeats = 20
        classification_results = []
        for i in range(classify_repeats):
            X_train, X_validate, y_train, y_validate= train_test_split(X, y, test_size=.2)
            classifier.fit(X_train, y_train)
            classification_results.append(classifier.score(X_validate, y_validate))

        return np.mean(classification_results), np.std(classification_results)

    def get_classifier():
        if classifier_name == names[0]:
            return lambda k: KNeighborsClassifier(n_neighbors=k)
        else:
            return lambda k: KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'V': np.cov(X.T)})

    results = []
    errors = []
    for i in range(repeats):
        current_classifier = get_classifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

        k_results = []
        k_error = []
        for k in range(1, 21):
            k_result = classify(X_train, y_train, current_classifier(k))
            k_results.append(k_result)
            k_error.append(((1 - k_result[0]) * 100))
        best_k = np.argmax(list(map(lambda x: x[0], k_results))) + 1

        errors.append(k_error)
        best_classifier = current_classifier(best_k)
        best_classifier.fit(X_train, y_train)
        results.append(best_classifier.score(X_test, y_test))

    errors = np.vstack([e for e in errors])

    return np.mean(results), 100 - np.mean(errors, axis=0), np.std(results)


def draw_comparison(X, y):
    mean_results = []
    std_list = []
    for i, classifier_name in enumerate(names):
        mean_result, mean_accuracy, std = test_classifier(X, y, classifier_name)
        mean_results.append(mean_result * 100)
        std_list.append(std * 100)
        plt.plot(np.arange(1, 21), mean_accuracy, marker='.', label=classifier_name)

    plt.xlabel('k')
    plt.ylabel('Średnia poprawność (%)')
    plt.legend(loc="lower right")
    plt.show()

    _, ax = plt.subplots()
    ax.bar(np.arange(len(names)), mean_results, yerr=std_list)
    ax.set_ylabel('Średnia poprawność (%)')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)

    plt.show()


if __name__ == '__main__':
    data = generate_data()
    draw_data(data)
    X = np.concatenate(data, axis=0)
    y = np.concatenate([np.full(data_class.shape[0], i) for i, data_class in enumerate(data)], axis=0)
    draw_partitions(X, y)
    draw_comparison(X, y)
