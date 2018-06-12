import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUB_DIR = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
PSEUDO_PRED_99 = os.path.join(SUB_DIR, "submission99p.csv")
PSEUDO_PRED = os.path.join(SUB_DIR, "submission_p.csv")
import numpy as np
import pandas as pd


submission = pd.read_csv(PSEUDO_PRED_99)
submission["deal_probability"] = np.clip(submission["deal_probability"], 0.0, 1.0)
print(submission.describe())
submission.to_csv(PSEUDO_PRED, index=False)

