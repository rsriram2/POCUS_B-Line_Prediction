import pyreadr
import pandas as pd

file_path = "/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/NHANES/nhanes_fda_with_r.rds"

result = pyreadr.read_r(file_path)

