from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from util import read_training_data, load_word_dict, read_testing_data

    
if __name__ == "__main__":
    word_mapping = load_word_dict()
    # Using MLPClassifier for distinguishing synonyms and antonyms
    X_train, y_train = read_training_data(word_mapping)
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    # print("Training score: ", clf.score(X_train, y_train))

    # Test result
    X_test, y_test = read_testing_data(word_mapping)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))