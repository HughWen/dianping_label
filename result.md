name_corpus | type_corpus | name_model | shape_data | k_fold | accuracy
-- | -- | -- | -- | -- | -- | --
dianping | food | svm | 5619 | 5 | 72.77
dianping | env | svm | 1174 | 5 | 64.66
dianping | price | svm | 1679 | 5 | 56.33
dianping | service | svm | 1058 | 5 | 58.52
dianping | all | svm | 9530 | 5 | 67.29
-- | -- | -- | -- | -- | -- | --
dianping | food | svm + pca | 5619 | 5 | 76.71
dianping | env | svm + pca | 1174 | 5 | 67.63
dianping | price | svm + pca | 1679 | 5 | 60.55
dianping | service | svm + pca | 1058 | 5 | 66.30
dianping | all | svm + pca | 9530 | 5 | 71.59
-- | -- | -- | -- | -- | -- | --
dianping | food | lstm | 5619 | 5 | 80.55
dianping | env | lstm | 1174 | 5 | 70.17
dianping | price | lstm | 1679 | 5 | 68.66
dianping | service | lstm | 1058 | 5 | 71.71
dianping | all | lstm | 9530 | 5 | 76.19
