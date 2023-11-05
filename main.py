import train_test
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
df = pd.DataFrame([train_test.main(row) for i, row in input_data.iterrows()])
df.to_csv("output_data.csv")
