import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from sklearn import metrics
from avito.common import pocket_logger


class HoldoutValidator:
    def __init__(self, model, valid_x, valid_y, max_series):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.max_series = max_series
        print("Initialized validator.")

    def validate(self):
        y_pred = self.model.predict(self.valid_x)
        y_true = self.valid_y

        score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
        self.output_score(score,  "valid score=")

        max_value = self.max_series
        c_pred = np.where(y_pred > max_value, max_value, y_pred)
        score = metrics.mean_squared_error(y_true, c_pred) ** 0.5
        self.output_score(score,  "clipped score=")

    def output_prediction(self, filename):
        self.holdout_df["pred"].to_csv(filename, index=False)

    def output_score(self, score, msg):
        score_msg = msg + "{:.15f}".format(score)
        print(score_msg)
        self.logger.info(score_msg)