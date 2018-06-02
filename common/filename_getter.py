import os


def get_filename(base_dir, prefix1, prefix2):
    col_name = prefix1 + "_" + prefix2 + "_col.csv"
    train_name = prefix1 + "_" + prefix2 + "_train.npz"
    test_name = prefix1 + "_" + prefix2 + "_test.npz"

    cols_file = os.path.join(base_dir, col_name)
    train_file = os.path.join(base_dir, train_name)
    test_file = os.path.join(base_dir, test_name)

    return cols_file, train_file, test_file