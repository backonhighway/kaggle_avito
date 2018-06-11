import os


def get_filename(base_dir, prefix1, prefix2):
    col_name = prefix1 + "_" + prefix2 + "_col.csv"
    train_name = prefix1 + "_" + prefix2 + "_train.npz"
    test_name = prefix1 + "_" + prefix2 + "_test.npz"

    cols_file = os.path.join(base_dir, col_name)
    train_file = os.path.join(base_dir, train_name)
    test_file = os.path.join(base_dir, test_name)

    return cols_file, train_file, test_file


def get_lda_filename(base_dir, prefix, train_or_test):
    dense_name = "lda_" + prefix + "_dense_" + train_or_test
    desc_name = "lda_" + prefix + "_desc_" + train_or_test
    title_name = "lda_" + prefix + "_title_" + train_or_test

    dense_file = os.path.join(base_dir, dense_name)
    desc_file = os.path.join(base_dir, desc_name)
    title_file = os.path.join(base_dir, title_name)

    return dense_file, desc_file, title_file