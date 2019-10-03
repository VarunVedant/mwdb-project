import cv2
import feature_models
import csv
import glob
import time
import os
import config

task_number = input('Choose from the following options: \n'
                        '1. Task1: Compute feature descriptor of a single image\n'
                        '2. Task2: Compute feature descriptors on a folder of images\n'
                        '3. Task3: Compute the top K similar matches for a given image\n')

if int(task_number) == 1:
    feature_model = input('Choose a feature model: \n'
                              '1) Color Moments \n'
                              '2) SIFT: \n')
    # filename would be the unique image identifier. Hence image_id will be the filename of a image
    image_id = input('Enter the file name: ')
    # base_path = '/Users/ramya/PycharmProjects/mwdb_phase1/Hands/'
    base_path = config.IMG_LIST_PATH
    if int(feature_model) == 2:
        # Computing feature descriptors for sift model for task 1.
        start = time.time()
        input_image = cv2.imread(base_path + image_id, 0)
        keypoints = feature_models.sift(input_image, image_id)
        # Iterating over each keypoint to round the distance to 3 decimals. Easy to read.
        for i in keypoints:
            for index in range(1, len(i)):
                element = i[index]
                i[index] = round(element, 3)
        csv_writer = open('sift_task1.csv', 'w')
        writer = csv.writer(csv_writer)
        writer.writerows(keypoints)
        print( 'Number of keypoints detected: ', len(keypoints))
        print( 'Feature descriptors have been saved to sift_task1.csv file')
        print( 'Time taken: ', time.time() - start)
    else:
        # Computing feature descriptors for color moments for task 1.
        start = time.time()
        input_image = cv2.imread(base_path + image_id)
        feature_descriptors = feature_models.color_moments(input_image, image_id)
        csv_writer = open('color_moments_task1.csv', 'w')
        writer = csv.writer(csv_writer)
        writer.writerow(feature_descriptors)
        print( 'Feature descriptor: ', feature_descriptors)
        print( 'Feature descriptor has been saved to color_moments_task1.csv file')
        print( 'Time taken: ', time.time() - start)

elif int(task_number) == 2:
    folder_name = input('Enter the name of the folder: ')
    feature_model = input('Choose a feature model: \n'
                              '1) Color Moments \n'
                              '2) SIFT: \n')
    # base_path = '/Users/ramya/PycharmProjects/mwdb_phase1/'
    base_path = config.PROJ_BASE_PATH
    path = base_path + folder_name + '/*.jpg'

    if int(feature_model) == 2:
        # Computing feature descriptors for sift model for task 2.
        start = time.time()
        # a count variable to keep track of the number of images in the given folder
        count = 0
        for input_file in glob.glob(path):
            count = count + 1
            filename = input_file.split('/')[-1]
            input_image = cv2.imread(input_file, 0)
            keypoints = feature_models.sift(input_image, filename)
            # Iterating over each keypoint to round the distance to 3 decimals. Easy to read.
            for i in keypoints:
                for index in range(1, len(i)):
                    element = i[index]
                    i[index] = round(element, 3)
            # writing descriptors of all the images in input folder into a single output file.
            csv_writer = open('sift_task2.csv', 'a')
            writer = csv.writer(csv_writer)
            writer.writerows(keypoints)
        print( 'Total number of images: ', count)
        print( 'Feature descriptors for all the images in the folder', folder_name, 'have been saved to sift_task2.csv file')
        print( 'Time taken: ', time.time() - start)
    else:
        # Computing feature descriptors for color moments model for task 2.
        start = time.time()
        # a count variable to keep track of the number of images in the given folder
        count = 0
        for input_file in glob.glob(path):
            count = count + 1
            filename = input_file.split('/')[-1]
            input_image = cv2.imread(input_file)
            feature_descriptor = feature_models.color_moments(input_image, filename)
            csv_writer = open('color_moments_task2.csv', 'a')
            writer = csv.writer(csv_writer)
            writer.writerow(feature_descriptor)
        print( 'Total number of images: ', count)
        print ('Feature descriptors for all the images in the folder', folder_name, 'have been saved to color_moments_task2.csv file')
        print( 'Time taken: ', time.time() - start)

elif int(task_number) == 3:
    # filename would be the unique image identifier
    image_id = input('Enter the file name: ')
    feature_model = input('Choose a feature model: \n'
                              '1) Color Moments \n'
                              '2) SIFT: \n')
    # base_path = '/Users/ramya/PycharmProjects/mwdb_phase1/Hands/'
    base_path = config.IMG_LIST_PATH
    k_value = int(input('Enter the value of K: '))

    if int(feature_model) == 2:
        # Computing feature descriptors for sift model for task 3.
        start = time.time()
        input_image = cv2.imread(base_path + image_id, 0)
        keypoints_list_input_image = feature_models.sift(input_image, image_id)
        keypoints_map_input_image = {}
        # each descriptor is of the form <filename, keypoint_xcoord, keypoint_ycoord, scale, orientation, d1 ... d128>
        # Only the descriptor <d1...d128> is added to the map
        for value in keypoints_list_input_image:
            keypoints_map_input_image.setdefault(value[0], []).append(value[5:])
        csv_file = open('sift_task2.csv', 'r')
        csv_reader = csv.reader(csv_file, delimiter=',')
        keypoints_map_all_test_images = {}
        # each line is of the form <filename, keypoint_xcoord, keypoint_ycoord, scale, orientation, d1 ... d128>
        # Only the descriptor <d1...d128> is added to the map
        for line in csv_reader:
            keypoints_map_all_test_images.setdefault(line[0], []).append([float(i) for i in line[5:]])
        ordered_image_list = feature_models.compute_similarity_scores_sift(keypoints_map_input_image,
                                                                              keypoints_map_all_test_images)
        # extracting the top k images from the list
        top_k_images = ordered_image_list[:k_value]
        # a method that plots multiple images in a single window
        feature_models.display_image_grid(k_value, top_k_images, base_path)
        print(top_k_images)
        print( 'Time taken: ', time.time() - start)
    else:
        # Computing feature descriptors for color moments model for task 3.
        start = time.time()
        input_image = cv2.imread(base_path + image_id)
        input_image_descriptor = feature_models.color_moments(input_image, image_id)

        descriptor_map_input_image = {}
        for value in input_image_descriptor:
            descriptor_map_input_image.setdefault(image_id, []).append(value)

        csv_file = open('color_moments_task2.csv', 'r')
        csv_reader = csv.reader(csv_file, delimiter=',')
        descriptor_map_all_test_images = {}
        # each line is of the form <filename, d1 ... >
        # Only the descriptor <d1...> is added to the map
        for line in csv_reader:
            descriptor_map_all_test_images.setdefault(line[0], []).append([float(i) for i in line[1:]])
        distances_list = feature_models.compute_similarity_scores_color_moments(descriptor_map_input_image,
                                                                                descriptor_map_all_test_images)
        # extracting the top k images from the list
        top_k_images = distances_list[:k_value]
        # a method that plots multiple images in a single window
        feature_models.display_image_grid(k_value, top_k_images, base_path)
        print (top_k_images)
        print ('Time taken: ', time.time() - start)
else:
    print("Please choose from the given options")


