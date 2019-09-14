"""Main program for Phase 1 Code"""
import sys
import feature_descriptor

def main():
    img_id = input('\nEnter the image ID: ')
    while True:
        feat_ch = input('\n1. LBP\n2. HOG\n3. Exit\nEnter Choice: ')
        print('ur choice', feat_ch, type(feat_ch), feat_ch==1)
        if feat_ch == '1':
            feature_descriptor.feature_descriptor.lbp_feat_desc(img_id)
        elif feat_ch == '2':
            feature_descriptor.feature_descriptor.hog_feat_desc(img_id)
        elif feat_ch == '3':
            break
    pass


if __name__ == '__main__':
    sys.exit(main())
