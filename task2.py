"""Task 2 Code
    1. Accepts Path of folder with set of images
    2. Accepts the type of model to use (LBP or HOG)
    3. Computes feature descriptors for all images in the folder
    4. Stores the feature descriptor for each image as JSON in a file corresponding to the chosen model.
"""
import feature_descriptor as fd



def task2():
    path = input('\nProvide the folder path: ')
    while True:
        feat_ch = input('\n\nChoose Model from the Menu\n1. LBP\n2. HOG\n3. Exit\nEnter Choice: ')
        if feat_ch == '1':
            fd.feature_descriptor.compute_lbp_vec(path)
        elif feat_ch == '2':
            fd.feature_descriptor.compute_hog_vec(path)
        elif feat_ch == '3':
            break
