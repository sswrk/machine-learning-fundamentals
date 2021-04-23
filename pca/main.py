import glob
import os
import string

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_images(dataset_path):
    results = []
    classes = []
    for path in glob.glob(f'{dataset_path}/*.png'):
        results.append(Image.open(path))
        classes.append(os.path.basename(path).translate(str.maketrans('', '', string.digits)).rsplit(".", 1)[0])
    return results, classes


def convert_images_grayscale_size(dataset):
    return [image.convert('L').resize((100, 100)) for image in dataset]


def convert_images_vector(dataset):
    vectors = [np.asarray(image).reshape((-1, 1))/255.0 for image in dataset]
    return np.hstack(vectors)


def vector_to_image(img_vector):
    return Image.fromarray(img_vector.reshape((100, 100))*255.0)


def draw_dataset(dataset):
    fig, axes = plt.subplots(5, 6, subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(vector_to_image(dataset[:, i]))
    plt.show()


def mean_element(dataset):
    return np.mean(dataset, axis=1)


def draw_mean_element(dataset):
    mean_image = vector_to_image(mean_element(dataset))
    plt.imshow(mean_image.resize((300, 300)))
    plt.axis('off')
    plt.show()


def draw_principal_components(dataset):
    pca = PCA().fit(dataset.T)
    fig, axes = plt.subplots(5, 6, subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.35, wspace=0.2))
    for i, ax in enumerate(axes.flat):
        components = pca.components_[i]
        ax.imshow(vector_to_image(components/np.max(components)))
        variance = pca.explained_variance_ratio_[i] * 100
        ax.set_xlabel(f'{variance:.2f}%')

    plt.show()


def reduce_dimensions(dataset, dimensions):
    pca = PCA(n_components=dimensions).fit(dataset.T)
    projected = pca.inverse_transform(pca.transform(dataset.T)).T
    return projected


def draw_2d_dataset(classes, dataset):
    dataset = PCA(2).fit_transform(dataset.T)

    for current_class in set(classes):
        xs, ys = [], []
        for i in range(len(classes)):
            if classes[i] == current_class:
                xs.append(dataset[i, 0])
                ys.append(dataset[i, 1])

        plt.scatter(xs, ys, label=current_class)

    plt.xlabel('Wymiar 1')
    plt.ylabel('Wymiar 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    images, classes = load_images('resources2')
    converted_images = convert_images_grayscale_size(images)
    image_vectors = convert_images_vector(converted_images)
    draw_dataset(image_vectors)

    draw_mean_element(image_vectors)
    draw_principal_components(image_vectors)

    dataset_4d = reduce_dimensions(image_vectors, 4)
    draw_dataset(dataset_4d)
    dataset_16d = reduce_dimensions(image_vectors, 16)
    draw_dataset(dataset_16d)
    dataset_2d = reduce_dimensions(image_vectors, 2)
    draw_2d_dataset(classes, dataset_2d)
