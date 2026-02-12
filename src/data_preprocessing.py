from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def load_and_preprocess():
    categories = ['talk.politics.misc', 'rec.sport.baseball', 'sci.med', 'comp.graphics']
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    X_train = train_data.data
    y_train = train_data.target

    X_test = test_data.data
    y_test = test_data.target

    return X_train, X_test, y_train, y_test, train_data.target_names

