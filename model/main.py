import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# import pickle5 as pickle


def create_model(data):
    #
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scaling the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividing data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Testing the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv(
        "data/dataset.csv")

    # Data cleaning, drop 'Unnamed: 32' cause its all NaN
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Encoding the diagnosis variables so that malignant = 1 and benign = 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)


if __name__ == '__main__':
    main()
