import os


def get_filename(base_dir, prefix):
    col_name = prefix + "_tf_col.csv"
    train_name = prefix + "_tf_train.npz"
    test_name = prefix + "tf_test.npz"

    cols_file = os.path.join(base_dir, col_name)
    train_file = os.path.join(base_dir, train_name)
    test_file = os.path.join(base_dir, test_name)

    return cols_file, train_file, test_file