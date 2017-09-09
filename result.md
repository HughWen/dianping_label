name_corpus| type_corpus | name_model | shape_data | k_fold | accuracy
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
-- | -- | -- | -- | -- | -- | --
dc | appearance | svm | 399 | 5 | 90.03
dc | quality | svm | 643 | 5 | 47.57
dc | cost_performance | svm | 251 | 5 | 82.12
dc | performance | svm | 1349 | 5 | 65.94
dc | price | svm | 586 | 5 | 56.17
dc | all | svm | 3228 | 5 | 64.75
-- | -- | -- | -- | -- | -- | --
dc | appearance | svm + pca | 399 | 5 | 89.80
dc | quality | svm + pca | 643 | 5 | 60.33
dc | cost_performance | svm + pca | 251 | 5 | 81.57
dc | performance | svm + pca | 1349 | 5 | 68.87
dc | price | svm + pca | 586 | 5 | 70.66
dc | all | svm + pca | 3228 | 5 | 71.07
-- | -- | -- | -- | -- | -- | --
dc | appearance | lstm | 399 | 5 | 89.67
dc | quality | lstm | 643 | 5 | 60.56
dc | cost_performance | lstm | 251 | 5 | 84.41
dc | performance | lstm | 1349 | 5 | 70.74
dc | price | lstm | 586 | 5 | 75.90
dc | all | lstm | 3228 | 5 | 73.05
