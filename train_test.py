from computeFeatures import computeFeatures

from util import scorer
from util import calculateScores
from util import createSplits
from util import computeFeatureImportance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate

import pandas as pd


columns = {
    "Wind speed": "3.feed_28.SONICWS_MPH",
    "Wind Direction": "3.feed_28.SONICWD_DEG",
    "Sulphur dioxide": "3.feed_3.SO2_PPM",
    "Hydrogen sulphide": "3.feed_28.H2S_PPM",
}


def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")


def process_row(row, df_sensor, df_smell):
    if not row.get("model"):
        return row

    # Select some variables, which means the columns in the data table.
    if row.get("var_1"):
        variables = [row.get("var_1"), row.get("var_2"), row.get("var_3")]
        wanted_cols = ["DateTime"] + [columns.get(v) for v in variables]
        df_sensor = df_sensor[wanted_cols]

    # pretty_print(df_sensor, "Display selected sensor data and column names")

    df_X, df_Y, _ = computeFeatures(
        df_esdr=df_sensor,
        df_smell=df_smell,
        f_hr=row.get("smell_predict_hrs"),
        b_hr=row.get("look_back_hrs"),
        thr=row.get("smell_thr"),
        add_inter=row.get("add_inter"),
    )
    # pretty_print(df_X, "Display features (X) and column names")
    # pretty_print(df_Y, "Display response (Y) and column names")

    # Split data to have parts for testing and training separate (otherwise it is not representative)
    splits = createSplits(row.get("test_size"), row.get("train_size"), df_X.shape[0])

    # Indicate which model you want to use to predict smell events
    if row.get("model") == "Random Forest":
        model = RandomForestClassifier()
    elif row.get("model") == "Support Vector Machines":
        model = LinearSVC(max_iter=1000)
    elif row.get("model") == "Stochastic Gradient Descent":
        model = SGDClassifier()
    elif row.get("model") == "MLPClassifier":
        model = MLPClassifier(max_iter=500)
    else:
        model = DecisionTreeClassifier()

    # Perform cross-validation to evaluate the model
    print("Use model", model)
    print("Perform cross-validation, please wait...")
    result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)

    scores = calculateScores(result)

    # TODO: Compute and show feature importance weights
    # feature_importance = computeFeatureImportance(df_X, df_Y, scoring="f1")
    # pretty_print(feature_importance,
    #              "Display feature importance based on f1-score")

    return pd.Series({**row, **scores})
