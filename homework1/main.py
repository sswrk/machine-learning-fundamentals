import numpy as np
import random
import matplotlib.pyplot as plt


def random_points(quantity, dimensions):
    return np.random.uniform(-1, 1, size=(quantity, dimensions))


def is_in_sphere(point):
    return np.sqrt(sum(np.square(point))) <= 1


def get_angle(point1, point2):
    return np.arccos(np.clip(np.dot(point1 / np.linalg.norm(point1), point2 / np.linalg.norm(point2)), -1.0, 1.0))


def task1_results():
    results = {}
    for dimension in range(2, 11):
        dimension_results = []
        for try_number in range(0, 10):
            points_to_generate = 10000
            points = random_points(points_to_generate, dimension)
            points_in_sphere = 0
            for point in points:
                if is_in_sphere(point):
                    points_in_sphere += 1
            dimension_results.append(points_in_sphere)
        results[dimension] = dimension_results
    return results


def task1_plot(results):
    x = list(range(2, 11))
    y = []
    y_error = []
    for element in x:
        y.append(np.average(results[element]))
        y_error.append((max(results[element]) - np.average(results[element]),
                        np.average(results[element]) - min(results[element])))

    plt.errorbar(x, y, yerr=np.array(y_error).T)
    plt.show()


def task2_results():
    results = {}
    for dimension in range(2, 11):
        dimension_results = []
        for try_number in range(0, 10):
            points_to_generate = 100
            points = random_points(points_to_generate, dimension)
            distances = []
            for point in points:
                point_distances = []
                for other_point in point:
                    distance = np.sqrt(sum(np.square(np.subtract(point, other_point))))
                    if distance != 0:
                        point_distances.append(distance)
                distances.append(np.average(point_distances))
            dimension_results.append(np.average(distances))
        results[dimension] = dimension_results
    return results


def task2_plot(results):
    x = list(range(2, 11))
    y = []
    y_error = []
    for element in x:
        y.append(np.average(results[element]))
        y_error.append((max(results[element]) - np.average(results[element]),
                        np.average(results[element]) - min(results[element])))

    plt.errorbar(x, y, yerr=np.array(y_error).T, fmt='o')
    plt.show()


def task3_results():
    results = {}
    for dimension in range(2, 11):
        dimension_results = []
        points_to_generate = 1000
        points = random_points(points_to_generate, dimension)
        for try_number in range(0, 10000):
            p1, p2 = random.sample(list(points), 2)
            dimension_results.append(np.degrees(get_angle(p1, p2)))
        results[dimension] = dimension_results
    return results


def task3_plot(results):
    for i in range(2, 11):
        plt.plot(results[i], '.')
        plt.gca().set_ylim([0, 360])
        plt.show()


if __name__ == '__main__':
    task1_plot(task1_results())
    task2_plot(task2_results())
    task3_plot(task3_results())
