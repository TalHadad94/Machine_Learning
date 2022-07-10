import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

def Boosting():
    decisionStumps = []
    predictions = []
    temp = []

    # Loading the data CSV files.
    data = pd.read_csv("titanikData.csv")
    test = pd.read_csv('titanikTest.csv', names = ["pclass", "age", "gender", "survived"]) # Adding headlines.
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
    test['pclass'] = pd.Categorical(test['pclass'], ordered = True,categories = ['1st', '2nd', '3rd', 'crew']).codes
    test.drop(columns = ['gender', 'age', 'survived'], inplace = True)

    # Create the Boosting process - set all weights to be equal.
    m_evaluation = pd.DataFrame(data.survived_b.copy())
    m_evaluation['weights'] = 1 / len(data)
    m_evaluation.drop('survived_b', inplace = True, axis = 1)

    # Training process.
    X = data.drop(['survived_b'], axis = 1)
    Y = data['survived_b'].where(data['survived_b'] == 1, 0)
    iterations = 3
    for i in range(0, iterations):
        tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
        model = tree_model.fit(X, Y, sample_weight=np.array(m_evaluation['weights']))
        decisionStumps.append(model)
        m_evaluation['predictions'] = model.predict(X)
        m_evaluation['is_prediction_incorrect'] = np.where(m_evaluation['predictions'] != data['survived_b'], 1, 0)
        err_rate = np.sum(m_evaluation['weights'] * m_evaluation['is_prediction_incorrect'])
        if err_rate > 0.5:
            continue
        b = err_rate / (1 - err_rate)
        a = 0.5 * np.log(1 / b)
        temp.append(a)
        m_evaluation['weights'] *= np.exp(a * m_evaluation['is_prediction_incorrect']) # Weight calculation.
        m_evaluation['weights'].div(m_evaluation['weights'].sum()) # Rescale weights.

    # Predict process.
    X_test = test.drop(['survived_b'], axis = 1).reindex(range(len(test)))
    temp_predictions = []
    for alpha, model in zip(temp, decisionStumps):
        prediction = alpha * model.predict(X_test)
        temp_predictions.append(prediction)
    predictions = np.sign(np.sum(np.array(temp_predictions), axis = 0))

    # Adding the prediction to the Testing data.
    test['pred'] = predictions
    test['corrects_rows'] = predictions == test.survived_b
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
    test.to_csv('titanikPredictionBoosting.csv', index = False)
    print(test)
    print("")
    print("Prediction: ", corrects_rows * 100 / n,"%")


if __name__ == '__main__':
    Boosting()