import json
import re
import numpy as np


# given panda dataframe return X(examples) and Y(class)
def data_to_examples(df):
    X = []
    y = []
    for idx, row in df.iterrows():
        categories = json.loads(str(row['categories_cut']).replace("\'", '\"'))
        text = str(row['text']).lower()
        text = re.sub('[^a-z]+', ' ', text)
        x = text.split()

        X.append(x)
        # print(x)
        y.append(categories)
    return np.array(X), np.array(y)