type_corpus | name_model | shape_data | k_fold | accuracy
-- | -- | -- | -- | --
food | svm | 5619 | 5 | 72.77
env | svm | 1174 | 5 | 64.66
price | svm | 1679 | 5 | 56.33
service | svm | 1058 | 5 | 58.52
all | svm | 9530 | 5 | 67.29
-- | -- | -- | -- | --
food | svm + pca | 5619 | 5 | 76.71
env | svm + pca | 1174 | 5 | 67.63
price | svm + pca | 1679 | 5 | 60.55
service | svm + pca | 1058 | 5 | 66.30
all | svm + pca | 9530 | 5 | 71.59
-- | -- | -- | -- | --
food | lstm | 5619 | 5 | 80.55
env | lstm | 1174 | 5 | 70.17
price | lstm | 1679 | 5 | 68.66
service | lstm | 1058 | 5 | 71.71
all | lstm | 9530 | 5 | 76.19