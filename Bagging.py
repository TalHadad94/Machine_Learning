import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

def Bagging():
    basicLearner = 100
    decisonTrees = []
    predictions = []
    temp = []

    # Loading the data CSV files.
    data = pd.read_csv("titanikData.csv")
    test = pd.read_csv('titanikTest.csv', names = ["pclass", "age", "gender", "survived"])
    print("Loading data done")

    # Make the Training data Binary or Numerical.
    data['survived_b'] = np.where(data['survived'] == 'yes', 1, 0)
    data['gender_b'] = np.where(data['gender'] == 'female', 1, 0)
    data['age_b'] = np.where(data['age'] == 'child', 1, 0)
    data['pclass'] = pd.Categorical(data['pclass'], ordered = True, categories = ['1st', '2nd', '3rd', 'crew']).codes
    data.drop(columns = ['gender', 'age', 'survived'], inplace = True)

    # Make the Testing data Binary or Numerical.
    test['survived_b'] = np.where(test['survived'] == 'yes', 1, 0)
    test['gender_b'] = np.where(test['gender'] == 'female', 1, 0)
    test['age_b'] = np.where(test['age'] == 'child', 1, 0)
    test['pclass'] = pd.Categorical(test['pclass'], ordered = True, categories = ['1st', '2nd', '3rd', 'crew']).codes
    test.drop(columns = ['gender', 'age', 'survived'], inplace = True)

    # Create the Bagging process - extract 63 percent of the data and make that a model.
    baggingCreate = data.drop_duplicates()
    for i in range(basicLearner):
        sample = baggingCreate.sample(frac = 0.63)
        complementary_size = len(data) - len(sample)
        complementary = sample.sample(n = complementary_size, replace = True)
        temp.append(pd.concat([sample, complementary]))

    # Training process.
    for i in range(basicLearner):
        tree_model = DecisionTreeClassifier(criterion = "entropy", max_depth = 2) #
        X = temp[i].drop(['survived_b'], axis = 1)
        Y = temp[i]['survived_b'].where(temp[i]['survived_b'] == 1, 0)
        model = tree_model.fit(X, Y)
        decisonTrees.append(model)

    # Predict process.
    for i, model in enumerate(decisonTrees):
        predictions.append(model.predict(test.drop(['survived_b'], axis = 1)))
    pred_matrix = np.array(predictions)
    prediction = np.apply_along_axis(mode, 0, pred_matrix)

    # Adding the prediction to the Testing data.
    test['pred'] = prediction
    test['corrects_rows'] = prediction == test.survived_b
    corrects_rows = np.sum(test['corrects_rows'].astype(int))
    n = len(test.pred)

    # Returning the Testing data to words.
    test['survived'] = np.where(test['survived_b'] == 1, 'yes', 'no')
    test['gender'] = np.where(test['gender_b'] == 1, 'female', 'male')
    test['age'] = np.where(test['age_b'] == True, 'child', 'adult')
    test['pclass'] = test['pclass'].replace({0: '1st', 1: '2nd', 2: '3rd', 3: 'crew'})
    test['pred'] = np.where(test['pred'] == 1, 'yes', 'no')
    test.drop(columns = ['survived_b', 'gender_b', 'age_b', 'survived_b', 'corrects_rows'], inplace = True)

    # Printing the Testing data to the Terminal.
    test.to_csv('titanikPredictionBagging.csv', index = False)
    print(test)
    print("")
    print("Prediction: ",corrects_rows * 100 / n,"%")


def mode(a):
    u, c = np.unique(a, return_counts = True)
    return u[c.argmax()]


if __name__ == '__main__':
    Bagging()