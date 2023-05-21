import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("D:\\project\\dataset.csv")
data1 = pd.read_csv("D:\\project\\Symptom-severity.csv")


# Remove the whitespace from basic dataset
def whitespace_remover(data):
    # iterating over the columns
    for i in data.columns:

        # checking datatype of each columns
        if data[i].dtype == 'object':

            # applying strip function on column
            data[i] = data[i].str.strip()
        else:

            # if condn. is False then it will do nothing.
            pass


# applying whitespace_remover function on dataframe
whitespace_remover(data)
# Change the Dimorphic hemmorhoids(piles) with Dimorphic hemorrhoids(piles)
data.loc[(data['Disease'] == "Dimorphic hemmorhoids(piles)"), 'Disease'] = 'Dimorphic hemorrhoids(piles)'
data = data.fillna(0)
df = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
           'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
           'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
           'Symptom_15', 'Symptom_16', 'Symptom_17']]
# columns in df
col = df.columns
# Unique values of symptom column in data1
cols = data1['Symptom'].unique()
# unique values of entire dataset df
sym = df.values
# Encoding the symptoms with their severity weight
for i in range(len(cols)):
    sym[sym == cols[i]] = data1[data1['Symptom'] == cols[i]]['weight'].values[0]
# Convert the sym array to Dataframe
X = pd.DataFrame(sym, columns=col)
# Replace it with zero,because we don't know its exact weight
X = X.replace('foul_smell_of urine', 0)
# Replace it with zero,because we don't know its exact weight
X = X.replace('dischromic _patches', 0)
# Replace it with zero,because we don't know its exact weight
X = X.replace('spotting_ urination', 0)
# Select the target column
y = data['Disease']
# split the data as train and test data(20% data reserved for testing purpose)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
# Saving model using pickle
pickle.dump(knn, open('pred.pkl', 'wb'))
