import cv2
import math
import numpy as np
import statistics
from scipy.stats.mstats import skew
import matplotlib.pyplot as plt


# a method that partitions a input image into 100 * 100 windows
def get_100_by_100_windows(input_image):
    vertical_partitions = input_image.shape[1] / 100
    horizontal_partitions = input_image.shape[0] / 100
    windows_set = []
    windows_set_1 = np.vsplit(input_image, horizontal_partitions)
    for np_array in windows_set_1:
        windows_set_2 = np.hsplit(np_array, vertical_partitions)
        for i in windows_set_2:
            windows_set.append(i)
    return windows_set


# computing euclidean distance
def euclidean_distance(v1, v2):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(v1, v2)]))


# Computes color moments(mean, std, skewness) for the given input image
def color_moments(input_image, filename):
    # converting the input image to yuv before computing image color moments
    yuv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    windows_set = get_100_by_100_windows(yuv_image)

    y_channel_descriptor = []
    u_channel_descriptor = []
    v_channel_descriptor = []
    for i in windows_set:
        y_channel = i[:, :, 0]
        u_channel = i[:, :, 1]
        v_channel = i[:, :, 2]
        # computing the mean(first moment) for each channel
        first_moment_y = np.mean(y_channel)
        first_moment_u = np.mean(u_channel)
        first_moment_v = np.mean(v_channel)
        # computing the standard deviation(second moment) for each channel
        second_moment_y = np.std(y_channel)
        second_moment_u = np.std(u_channel)
        second_moment_v = np.std(v_channel)
        # computing the skewness(third moment) for each channel
        third_moment_y = skew(y_channel, axis=None)
        third_moment_u = skew(u_channel, axis=None)
        third_moment_v = skew(v_channel, axis=None)
        # each of the moment value is rounded to three decimals. Easy to read
        y_channel_descriptor.extend([round(first_moment_y, 3), round(second_moment_y, 3), round(third_moment_y, 3)])
        u_channel_descriptor.extend([round(first_moment_u, 3), round(second_moment_u, 3), round(third_moment_u, 3)])
        v_channel_descriptor.extend([round(first_moment_v, 3), round(second_moment_v, 3), round(third_moment_v, 3)])
    return [filename] + y_channel_descriptor + u_channel_descriptor + v_channel_descriptor


# Compute similarity score for color moments
def compute_similarity_scores_color_moments(descriptor_map_input_image, descriptor_map_all_test_images):
    distances_list = []
    # each value is of the form <filename, d1 ... >
    # Only the descriptor <d1...> is added to the map
    descriptor_list_input_image = descriptor_map_input_image[descriptor_map_input_image.keys()[0]][1:]
    for test_image in descriptor_map_all_test_images:
        descriptor_test_image = descriptor_map_all_test_images[test_image][0]
        distance = euclidean_distance(descriptor_list_input_image, descriptor_test_image)
        distances_list.append((test_image, distance))
    return sort_list_of_tuples(distances_list)


# Compute sift descriptor
def sift(input_image, image_id):
    sift_create = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift_create.detectAndCompute(input_image, None)
    keypoints_vector = []
    for key_point, descriptor in zip(key_points, descriptors):
        row = []
        # each row is of the form < filename, keypoint_xcoord, keypoint_ycoord, scale, orientation, d1...d128 >
        row.extend([image_id, key_point.pt[0], key_point.pt[1], key_point.size, key_point.angle])
        row.extend(descriptor.tolist())
        keypoints_vector.append(row)
    for i in keypoints_vector:
        for index in range(1, len(i)):
            element = i[index]
            i[index] = round(element, 3)
    return keypoints_vector


# computes similarity score for sift
def compute_similarity_scores_sift(keypoints_map_input_image, keypoints_map_all_test_images):
    keypoints_list_input_image = keypoints_map_input_image[keypoints_map_input_image.keys()[0]]
    distances_map = {}
    # compute the first closest key point
    for test_image in keypoints_map_all_test_images:
        keypoints_list_test_image = keypoints_map_all_test_images[test_image]
        for keypoint1 in keypoints_list_input_image:
            min_distance_from_keypoint1 = float('inf')
            for keypoint2 in keypoints_list_test_image:
                distance = euclidean_distance(keypoint1, keypoint2)
                if distance < min_distance_from_keypoint1:
                    min_distance_from_keypoint1 = distance
            distances_map.setdefault(test_image, []).append(min_distance_from_keypoint1)
    mean_distances_list = []
    # the mean of all the distances is considered the score for an image
    for key in distances_map:
        value = distances_map[key]
        distance_mean = statistics.mean(value)
        mean_distances_list.append((key, distance_mean))
    sort_list_of_tuples(mean_distances_list)
    return mean_distances_list


# plots multiple images in a single window
def display_image_grid(image_count, top_k_images, base_path):
    display_grid = plt.figure(figsize=(20, 20))
    column_count = 6
    row_count = (image_count / 6) + 1
    for itr in range(1, column_count * row_count + 1):
        if itr > image_count:
            break
        file_name = top_k_images[itr - 1][0]
        score = top_k_images[itr - 1][1]
        image = cv2.imread(base_path + file_name)
        ax = display_grid.add_subplot(row_count, column_count, itr)
        ax.set_title(file_name + ' Score: ' + str("{0:.3f}".format(score)))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


# sort the list of tupes in ascending order
def sort_list_of_tuples(tuple_list):
    tuple_list.sort(key=lambda x: x[1])
    return tuple_list
