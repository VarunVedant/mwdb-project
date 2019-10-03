"""Task 1 Code
    1. Accepts Image ID from the user
    2. Accepts the type of model to use (LBP or HOG)
    3. Displays Feature Descriptor in Human Readable Form
"""
import feature_descriptor



def task1():
    img_id = input('\nEnter the image ID: ')
    while True:
        feat_ch = input('\n1. LBP\n2. HOG\n3. Exit\nEnter Choice: ')
        if feat_ch == '1':
            feature_descriptor.feature_descriptor.lbp_feat_desc(img_id)
        elif feat_ch == '2':
            feature_descriptor.feature_descriptor.hog_feat_desc(img_id)
        elif feat_ch == '3':
            break
