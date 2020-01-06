import pickle
import sys

import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    load table from sqlite database and split into features, targets and category names
    :param database_filepath:
    :return: (numpy array of messages, numpy array of targets, list of category names)
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)
    x = df['message'].to_numpy()
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns
    y = y.to_numpy()
    return x, y, category_names


def tokenize(text):
    """
    tokenize text into words and lemmatize these
    :param text: string
    :return: list of strings
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build machine leaning pipeline and grid search object
    :return: sklearn grid search
    """
    pipeline = Pipeline(steps=[
        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # build stopwords list for CountVectorizer parameter, use nltk stopwords list and tokenize with tokenize function
    # and with sklearn in-built tokenizer
    stop_words = stopwords.words('english')
    stopwords_custom = tokenize(' '.join(stop_words))
    stopwords_sklearn = CountVectorizer().build_analyzer()(' '.join(stop_words))
    stopwords_all = list(set(stopwords_custom + stopwords_sklearn))
    parameters = {'tfidf__norm': ['l1', 'l2', None],
                  'count__tokenizer': [tokenize, None],
                 'count__stop_words': [None, stopwords_all],
                 'clf__estimator__n_estimators': [10, 100],
                 'clf__estimator__max_depth': [5, None]}
    cv = GridSearchCV(pipeline, parameters, verbose=10, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    print classification report for a model for each output
    :param model: sklearn model or pipeline
    :param X_test: numpy array of messages
    :param Y_test: numpy array of categories
    :param category_names: list of string containing category names
    :return:
    """
    y_pred = model.predict(X_test)
    for i, cat in enumerate(category_names):
        print(cat)
        print(classification_report(Y_test[:, i], y_pred[:, i]))
        print('###########################')
    return None


def save_model(model, model_filepath):
    """
    save model to pickle file
    :param model: sklearn model
    :param model_filepath: filepath where to save model
    :return: None
    """
    joblib.dump(model, model_filepath)
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model = model.fit( X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
