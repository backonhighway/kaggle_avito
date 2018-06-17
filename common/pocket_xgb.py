import xgboost as xgb
import pandas as pd
from . import pocket_logger


class GoldenXgb:
    def __init__(self):
        self.train_param = {
            'learning_rate': 0.1,
            "max_depth": 10,
            "silent": 1,
            "objective": "reg:logistic",
            "eval_metric": "rmse",
            #"tree_method": "hist",
            #'grow_policy': "lossguide",
            "subsample": 0.75,
            "colsample_bytree": 0.5,
            "seed": 99
        }
        self.target_col_name = "deal_probability"
        self.drop_cols = ["deal_probability"]

    def do_train(self, train_data, test_data, lgb_col):
        tcn = self.target_col_name
        y_train = train_data[tcn]
        y_test = test_data[tcn]
        x_train = train_data[lgb_col]
        x_test = train_data[lgb_col]
        # x_train = train_data.drop(self.drop_cols, axis=1)
        # x_test = test_data.drop(self.drop_cols, axis=1)

        return self.do_train_avito(x_train, x_test, y_train, y_test, lgb_col)

    def do_train_avito(self, x_train, x_test, y_train, y_test, feature_name=None):
        d_train = xgb.DMatrix(x_train, label=y_train, )
        d_test = xgb.DMatrix(x_test, label=y_test, )
        #d_train = xgb.DMatrix(x_train, label=y_train, feature_names=feature_name)
        #d_test = xgb.DMatrix(x_test, label=y_test, feature_names=feature_name)
        watchlist = [(d_test, "eval")]
        num_rounds = 10000
        early_stopping_rounds = 100

        print('Start training...')
        model = xgb.train(
            params=self.train_param,
            dtrain=d_train,
            evals=watchlist,
            num_boost_round=num_rounds,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        print('End training...')
        return model

    @staticmethod
    def do_predict(model, x_test):
        d_test = xgb.DMatrix(x_test)
        return model.predict(d_test, ntree_limit=model.best_ntree_limit)

    @staticmethod
    def show_feature_importance(model):
        print(model.get_fscore())
        #print(model.feature_importances_)

    @staticmethod
    def save_binary(model, filename):
        model.save_model(filename)

    @staticmethod
    def load_model(filename):
        return xgb.Booster(filename)

