"""Task 3 Code
    1. Accepts Image ID, Model (LBP or HOG), value 'k'.
    2. Finds k most similar images to the given image ID.
    3. Displays matching score for each match in the list.
"""
import sys
import feature_descriptor as fd



def main():
    path = input('\nImage ID: ')
    while True:
        feat_ch = input('\n\nChoose Model from the Menu\n1. LBP\n2. HOG\n3. Exit\nEnter Choice: ')
        if feat_ch == '1':
            fd.feature_descriptor.compute_lbp_vec(path)
        elif feat_ch == '2':
            fd.feature_descriptor.compute_hog_vec(path)
        elif feat_ch == '3':
            break
    pass


if __name__ == '__main__':
    sys.exit(main())
