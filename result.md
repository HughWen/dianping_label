type_corpus | name_model | shape_dataset | validation_split | num_train_pos | num_train_neu | num_train_neg | num_test_pos | num_test_neu | num_test_neg | num_predict_pos | num_predict_neu | num_predict_neg | accuracy
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |--
food | svm (standard) | (5619, 22264) | 0.8 | 3251 | 985 | 259 | 838 | 215 | 71 | 1124 | 0 | 0| 0.7456
food | downsampling  + svm | (990, 8967) | 0.8 | 259 | 257 | 276 | 71 | 73 | 54 | 0 | 0 | 198 | 0.2727
food | upsampling + svm | (11649, 22264) | 0.8 | 3236 | 2878 | 3205 | 853 | 722 | 755 | 2288 | 0 | 42 | 0.3828
food | PCA + svm | (5619, 50) | 0.8 | 3251 | 985 | 259 | 838 | 215 | 71 | 1032 | 92 | 0 | 0.7874
food | PCA + downsampling + svm | (990, 50) | 0.8 | 259 | 257 | 276 | 71 | 73 | 54 | 82 | 60 | 56 | 0.5808
food | PCA + upsampling + svm | (11649, 50) | 0.8 | 3236 | 2878 | 3205 | 853 | 722 | 755 | 910 | 594 | 826 | 0.7099