import lightgbm as lgb
import pandas as pd
from . import pocket_logger


class GoldenLgb:
    def __init__(self):
        self.train_param = {
            'learning_rate': 0.05,
            'num_leaves': 255,
            'boosting': 'gbdt',
            'application': 'regression',
            'metric': 'rmse',
            'feature_fraction': .3,
            #"max_bin": 511,
            'seed': 99,
            'verbose': 0,
        }
        self.target_col_name = "deal_probability"
        self.category_col = [
            "region", "city", "parent_category_name", "category_name",
            "param_1", "param_2", "param_3",
            "param_all",
            "image_top_1", "user_type"
        ]
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

    def do_train_avito(self, x_train, x_test, y_train, y_test, feature_name):
        lgb_train = lgb.Dataset(x_train, y_train, feature_name=feature_name,
                                categorical_feature=self.category_col)
        lgb_eval = lgb.Dataset(x_test, y_test, feature_name=feature_name,
                               categorical_feature=self.category_col)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    def fuckin(self,x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=10000,
                          early_stopping_rounds=100,
                          )
        print('End training...')
        return model

    def do_train_stack(self, x_train, y_train, feature_name):
        lgb_train = lgb.Dataset(x_train, y_train, feature_name=feature_name,
                                categorical_feature=self.category_col)
        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          verbose_eval=100,
                          num_boost_round=4000,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    def do_train_sk(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=lgb_eval,
                          verbose_eval=100,
                          num_boost_round=30,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    @staticmethod
    def show_feature_importance(model):
        fi = pd.DataFrame({
            "name": model.feature_name(),
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        })
        fi = fi.sort_values(by="importance_split", ascending=False)
        print(fi)
        logger = pocket_logger.get_my_logger()
        logger.info(fi)


