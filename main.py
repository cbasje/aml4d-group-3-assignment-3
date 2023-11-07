from preprocessData import preprocessData
from train_test import process_row
import pandas as pd


input_data = (
    pd.read_csv("input_data.csv")
    .fillna(0)
    .astype(
        {
            "smell_predict_hrs": "int32",
            "smell_thr": "int32",
            "train_size": "int32",
            "test_size": "int32",
            "look_back_hrs": "int32",
        }
    )
)
df_sensor, df_smell = preprocessData(
    in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"]
)
# pretty_print(df_sensor, "Display all sensor data and column names")
# pretty_print(df_smell, "Display smell data and column names")

results = pd.DataFrame(
    [process_row(row, df_sensor, df_smell) for i, row in input_data.iterrows()]
)
results.to_csv("output_data.csv")
