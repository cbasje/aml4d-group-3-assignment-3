from computeFeatures import computeFeatures
from preprocessData import preprocessData

from util import scorer
from util import printScores
from util import createSplits
from util import computeFeatureImportance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate


def pretty_print(df, message):
  print("\n================================================")
  print("%s\n" % message)
  print(df)
  print("\nColumn names below:")
  print(list(df.columns))
  print("================================================\n")


df_sensor, df_smell = preprocessData(
    in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")

# Select some variables, which means the columns in the data table.
# (you can also comment out the following two lines to indicate that you want all variables)
wanted_cols = ["DateTime", "3.feed_28.H2S_PPM", "3.feed_28.SONICWD_DEG"]
df_sensor = df_sensor[wanted_cols]

pretty_print(df_sensor, "Display selected sensor data and column names")

# Indicate the threshold to define a smell event
smell_thr = 40

# Indicate the number of future hours to predict smell events
smell_predict_hrs = 8

# Indicate the number of hours to look back to check previous sensor data
look_back_hrs = 1

# Indicate if you want to add interaction terms in the features (like x1*x2)
add_inter = False

df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor,
                                df_smell=df_smell,
                                f_hr=smell_predict_hrs,
                                b_hr=look_back_hrs,
                                thr=smell_thr,
                                add_inter=add_inter)
pretty_print(df_X, "Display features (X) and column names")
pretty_print(df_Y, "Display response (Y) and column names")

# Indicate how much data you want to use to test the model
test_size = 168

# Indicate how much data you want to use to train the model
train_size = 336

# Split data to have parts for testing and training separate (otherwise it is not representative)
splits = createSplits(test_size, train_size, df_X.shape[0])

# Indicate which model you want to use to predict smell events
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

# Perform cross-validation to evaluate the model
print("Use model", model)
print("Perform cross-validation, please wait...")
result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
printScores(result)

# Compute and show feature importance weights
feature_importance = computeFeatureImportance(df_X, df_Y, scoring="f1")
pretty_print(feature_importance,
             "Display feature importance based on f1-score")
