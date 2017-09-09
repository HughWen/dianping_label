| name_corpus | type_corpus      | name_model | shape_data | k_fold | accuracy |
| ----------- | ---------------- | ---------- | ---------- | ------ | -------- |
| dc          | appearance       | svm        | 399        | 5      | 90.03    |
| dc          | quality          | svm        | 643        | 5      | 47.57    |
| dc          | cost_performance | svm        | 251        | 5      | 82.12    |
| dc          | performance      | svm        | 1349       | 5      | 65.94    |
| dc          | price            | svm        | 586        | 5      | 56.17    |
| dc          | all              | svm        | 3228       | 5      | 64.75    |
| --          | --               | --         | --         | --     | --       |
| dc          | appearance       | svm + pca  | 399        | 5      | 89.80    |
| dc          | quality          | svm + pca  | 643        | 5      | 60.33    |
| dc          | cost_performance | svm + pca  | 251        | 5      | 81.57    |
| dc          | performance      | svm + pca  | 1349       | 5      | 68.87    |
| dc          | price            | svm + pca  | 586        | 5      | 70.66    |
| dc          | all              | svm + pca  | 3228       | 5      | 71.07    |
| --          | --               | --         | --         | --     | --       |
| dc          | appearance       | lstm       | 399        | 5      | 89.67    |
| dc          | quality          | lstm       | 643        | 5      | 60.56    |
| dc          | cost_performance | lstm       | 251        | 5      | 84.41    |
| dc          | performance      | lstm       | 1349       | 5      | 70.74    |
| dc          | price            | lstm       | 586        | 5      | 75.90    |
| dc          | all              | lstm       | 3228       | 5      | 73.05    |
