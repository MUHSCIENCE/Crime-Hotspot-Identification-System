# Loading the necessary libraries

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_df(filenames):
    """
    Returns a dataframe after aggregating all necessary .csv files together

    PARAMETERS:
        filenames : List of .csv files to be aggregated
    """
    dfs = []  # Initialize an empty list to store DataFrames
    for file in filenames:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)  # Append the DataFrame to the list
        print("Finished loading Chicago Crime Dataset file for the year "+file[-8:-4]+".")
    main_df = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames in the list
    print("All data files loaded onto the Main Dataframe.\n\n")

    return main_df



'''Let's write code to automate the creating of our dataset'''

DATA_PATH = "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/"

file_names = ['2015.csv', '2016.csv', '2017.csv', '2018.csv', '2019.csv', '2020.csv', '2021.csv', '2022.csv']
file_names = [DATA_PATH+x for x in file_names]

main_df = create_df(file_names)
orig_shape = main_df.shape
print("The Number of Crimes: " + str(main_df.shape[0]))
print("\nThe Columns: " + str(main_df.shape[1]))

# Information about the main dataframe
main_df.info()

# Visualizing Missing Values
sns.heatmap(data=main_df.isna(), yticklabels=False, cbar=False, cmap="inferno")

# To drop the rows with missing data
main_df = main_df.dropna()
main_df.isna().sum()

# Inspecting the loss of data after such cleaning
print(
    "Data Retained after Cleaning:",
    round(((main_df.shape[0] / orig_shape[0]) * 100), 2),
    "%",
)

# First 10 rows (instances) of our dataset
main_df.head(10)

# What are the features of our dataset?
print(main_df.columns)

""" Function to Clean the 'Date' feature """


def time_convert(date_time):
    s1 = date_time[:11]
    s2 = date_time[11:]

    month = s1[:2]
    date = s1[3:5]
    year = s1[6:10]

    hr = s2[:2]
    mins = s2[3:5]
    sec = s2[6:8]
    time_frame = s2[9:]
    if time_frame == "PM":
        if int(hr) != 12:
            hr = str(int(hr) + 12)
    else:
        if int(hr) == 12:
            hr = "00"

    final_date = datetime(
        int(year), int(month), int(date), int(hr), int(mins), int(sec)
    )
    return final_date


# Using apply() of pandas to apply time_convert on every row of the Date column
main_df["Date"] = main_df["Date"].apply(time_convert)

# Inspect the cleaned "Date" column
main_df["Date"].head()

""" Feature Engineering - Splitting the 'Date' feature into more suitable features for a Time-based analysis"""


# Feature Engineering 1 : Month
def month_col(x):
    return int(x.strftime("%m"))


main_df["Month"] = main_df["Date"].apply(month_col)


# Feature Engineering 2 : Day
def day_col(x):
    return int(x.strftime("%w"))


main_df["Day"] = main_df["Date"].apply(day_col)


# Feature Engineering 3 : Hour
def hour_col(x):
    return int(x.strftime("%H"))


main_df["Hour"] = main_df["Date"].apply(hour_col)

# Inspect the latest version of the dataset

main_df.head()

""" Filter the Top 10 most occuring crimes in the city of Chicago """

"""
STEPS FOLLOWED WHILE DOING THIS :

1. Take in each crime and make a dataset of it
2. Append the sub datasets to each other
"""
top_10 = list(main_df["Primary Type"].value_counts().head(10).index)


def filter_top_10(df):
    dfs = []  # Initialize an empty list to store DataFrames
    for crime in top_10:
        temp = df[df["Primary Type"] == crime]
        dfs.append(temp)  # Append the DataFrame to the list
    df2 = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames in the list
    return df2



df2 = filter_top_10(
    main_df
)  # the dataframe with all the data of only the top 10 crimes
df2.shape

1036588 / 1146382 * 100

df2.head()

# Inspecting a few relevant features

df2[
    [
        "Domestic",
        "Beat",
        "District",
        "Ward",
        "Community Area",
        "FBI Code",
        "Location",
        "X Coordinate",
        "Y Coordinate",
    ]
].head()

""" Grouping """

# Creating our explicit dataset
cri5 = df2.groupby(["Month", "Day", "District", "Hour"], as_index=False).agg(
    {"Primary Type": "count"}
)
cri5 = cri5.sort_values(by=["District"], ascending=False)
cri5.head()

# Renaming our feature
cri6 = cri5.rename(index=str, columns={"Primary Type": "Crime_Count"})
cri6.head()

cri6 = cri6[["Month", "Day", "District", "Hour", "Crime_Count"]]
cri6.head()
print("The shape of our final dataset is:", cri6.shape)

# Viewing the maximum and minimum crime counts
print(
    "Highest Crime Count at any district at any time point:", cri6["Crime_Count"].max()
)
print(
    "Lowest Crime Count at any district at any time point:", cri6["Crime_Count"].min()
)

print(
    "Average no. of crimes per ditrict per time point :",
    round(cri6["Crime_Count"].sum() / cri6.shape[0], 2),
    ".",
)

# Inspecting our own lower and upper bounds to make a target feature "Alarm"

lower = np.mean(cri6["Crime_Count"]) - 0.75 * np.std(cri6["Crime_Count"])
higher = np.mean(cri6["Crime_Count"]) + 0.75 * np.std(cri6["Crime_Count"])
print(lower, higher)

# 0-14 : Low Crime Rate
# 15-33 : Medium Crime Rate
# 34 and above : High Crime Rate

# The above ranges can be made better with the help of a crime analyst. As of now, we have used an intuitive way
# of generating classifications for our target feature; based on aproximating the distribution of the crime counts
# as a Normal curve


# Feature Engineer the above dataset
def crime_rate_assign(x):
    if x <= 14:
        return 0
    elif 14 < x <= 33:
        return 1
    else:
        return 2


cri6["Alarm"] = cri6["Crime_Count"].apply(crime_rate_assign)
cri6 = cri6[["Month", "Day", "Hour", "District", "Crime_Count", "Alarm"]]
cri6.head()

# To store the above dataset as a csv file
cri6.to_csv("C:/Users/Henry/Desktop/Crime_Compress.csv")


# Let's check how good our data is for classification
cri6["Alarm"].value_counts()

print(
    "Low Crime Rate Percentage:",
    round(
        cri6["Alarm"].value_counts()[0] / cri6["Alarm"].value_counts().sum() * 100, 2
    ),
)
print(
    "Medium Crime Rate Percentage:",
    round(
        cri6["Alarm"].value_counts()[1] / cri6["Alarm"].value_counts().sum() * 100, 2
    ),
)
print(
    "High Crime Rate Percentage:",
    round(cri6["Alarm"].value_counts()[2] / cri6["Alarm"].value_counts().sum() * 100.2),
)

# Plotting the Imbalance

x = ["Low (0)", "Medium (1)", "High (2)"]
y = [13600, 23273, 7488]
fig, ax = plt.subplots(figsize=(3, 4))
plt.bar(x, y, color=["green", "blue", "red"], width=0.5)
plt.title('THE IMBALANCE IN THE DATASET')
plt.xlabel("Alarm Rate Classification")
plt.ylabel("Count of Crimes")
plt.title("Class Imbalance")
plt.savefig("C:/Users/Henry/Desktop/")

""" Building our completely unseen final test dataset for the 'GOD TEST 1' """

# Load the Dataset
test_files = ["2010.csv", "2011.csv", "2012.csv", "2013.csv", "2014.csv"]
test_files = [DATA_PATH + x for x in test_files]
test_df = create_df(test_files)

# Drop missing values
test_df = test_df.dropna()

# Using apply() of pandas to apply time_convert on every row of the Date column
test_df["Date"] = test_df["Date"].apply(time_convert)

# Feature Engineering our columns
test_df["Month"] = test_df["Date"].apply(month_col)
test_df["Day"] = test_df["Date"].apply(day_col)
test_df["Hour"] = test_df["Date"].apply(hour_col)

# Compressing
df7 = filter_top_10(test_df)
cri7 = df7.groupby(["Month", "Day", "District", "Hour"], as_index=False).agg(
    {"Primary Type": "count"}
)
cri7 = cri7.sort_values(by=["District"], ascending=False)
cri8 = cri7.rename(index=str, columns={"Primary Type": "Crime_Count"})
cri8 = cri8[["Month", "Day", "District", "Hour", "Crime_Count"]]
cri8["Alarm"] = cri8["Crime_Count"].apply(crime_rate_assign)
cri8 = cri8[["Month", "Day", "Hour", "District", "Crime_Count", "Alarm"]]
print(cri8.head())
print("Class Imbalance\n")
print(cri8["Alarm"].value_counts())

"""Creating the Oversampled balanced dataset"""

from sklearn.utils import resample  # for upsampling

# Set individual classes
cri6_low = cri6[cri6["Alarm"] == 0]
cri6_medium = cri6[cri6["Alarm"] == 1]
cri6_high = cri6[cri6["Alarm"] == 2]

# Upsample the minority classes to size of class 1 (medium)
cri6_low_upsampled = resample(
    cri6_low,
    replace=True,  # sample with replacement
    n_samples=22640,  # to match majority class
    random_state=101,
)

cri6_high_upsampled = resample(
    cri6_high,
    replace=True,  # sample with replacement
    n_samples=22640,  # to match majority class
    random_state=101,
)

# Combine majority class with upsampled minority class
cri6_upsampled = pd.concat([cri6_medium, cri6_low_upsampled, cri6_high_upsampled])

# Using Random Forest for classification (Imbalanced Dataset)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = cri6.iloc[:,0:4].values
y = cri6.iloc[:,5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 101)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", (sklearn.metrics.accuracy_score(y_test, y_pred) * 100), "\n")

cm = pd.crosstab(y_test, y_pred, rownames=['Actual Alarm'], colnames=['Predicted Alarm'])
print("\n----------Confusion Matrix------------------------------------")
print(cm)

# Classification Report
print("\n----------Classification Report------------------------------------")
print(classification_report(y_test,y_pred))

# Unweighted Average Recall
print("\nUAR ->",((cm[0][0])/(cm[0][0]+cm[1][0]+cm[2][0])+(cm[1][1])/(cm[0][1]+cm[1][1]+cm[2][1])+(cm[2][2])/(cm[2][2]+cm[0][2]+cm[1][2]))/3)

# Using Random Forest for classification (Balanced Dataset)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = cri6_upsampled.iloc[:,0:4].values
y = cri6_upsampled.iloc[:,5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 101)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", (sklearn.metrics.accuracy_score(y_test, y_pred) * 100), "\n")

cm = pd.crosstab(y_test, y_pred, rownames=['Actual Alarm'], colnames=['Predicted Alarm'])
print("\n----------Confusion Matrix------------------------------------")
print(cm)

# Classification Report
print("\n----------Classification Report------------------------------------")
print(classification_report(y_test,y_pred))

# Unweighted Average Recall
print("\nUAR ->",((cm[0][0])/(cm[0][0]+cm[1][0]+cm[2][0])+(cm[1][1])/(cm[0][1]+cm[1][1]+cm[2][1])+(cm[2][2])/(cm[2][2]+cm[0][2]+cm[1][2]))/3)

'''God Test 1 : Random Forest'''

X = cri8.iloc[:,0:4].values
y = cri8.iloc[:,5].values

# Testing directly
y_pred = classifier.predict(X)

print("Accuracy:", (sklearn.metrics.accuracy_score(y, y_pred) * 100), "\n")

# Confusion Matrix for evaluating the model
cm = pd.crosstab(y, y_pred, rownames=['Actual Alarm'], colnames=['Predicted Alarm'])
print("\n----------Confusion Matrix------------------------------------")
print(cm)

# Classification Report
print("\n----------Classification Report------------------------------------")
print(classification_report(y,y_pred))

# Unweighted Average Recall
print("\nUAR ->",((cm[0][0])/(cm[0][0]+cm[1][0]+cm[2][0])+(cm[1][1])/(cm[0][1]+cm[1][1]+cm[2][1])+(cm[2][2])/(cm[2][2]+cm[0][2]+cm[1][2]))/3)

# Using Random Forest for classification (Imbalanced Dataset) (using k-fold)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = cri6.iloc[:,0:4].values
y = cri6.iloc[:,5].values

scores = []
for train_index, test_index in skf.split(X, y):
    #print('{} of KFold {}'.format(i,skf.n_splits))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 101)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Model Evaluation
    # print(y_test)
    # print(y_pred)
    scores.append(sklearn.metrics.accuracy_score(y_test, y_pred)*100)
    #print("Accuracy:",(metrics.accuracy_score(y_test, y_pred)*100),"\n")

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Accuracy
print("Accuracy:",np.mean(scores),"\n")

cm = pd.crosstab(y_test, y_pred, rownames=['Actual Alarm'], colnames=['Predicted Alarm'])
print("\n----------Confusion Matrix------------------------------------")
print(cm)

# Classification Report
print("\n----------Classification Report------------------------------------")
print(classification_report(y_test,y_pred))

# Unweighted Average Recall
print("\nUAR ->",((cm[0][0])/(cm[0][0]+cm[1][0]+cm[2][0])+(cm[1][1])/(cm[0][1]+cm[1][1]+cm[2][1])+(cm[2][2])/(cm[2][2]+cm[0][2]+cm[1][2]))/3)