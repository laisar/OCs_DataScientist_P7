import pandas as pd
import pytest


@pytest.fixture(scope='module')
def data_train():
    """Get customers processed train data to feed into the tests"""
    return pd.read_csv("./data/dataset_predict_compressed.gz", compression='gzip', sep=',')

@pytest.fixture(scope='module')
def data_test():
    """Get customers processed test data to feed into the tests"""
    return pd.read_csv("./data/dataset_predict_compressed.gz", compression='gzip', sep=',')

def test_train_duplicates(data_train):
    """Test if the train duplicated dataframe is empty --> no duplicates"""
    duplicates = data_train[data_train.duplicated()]
    assert duplicates.empty

def test_test_duplicates(data_test):
    """Test if the test duplicated dataframe is empty --> no duplicates"""
    duplicates = data_test[data_test.duplicated()]
    assert duplicates.empty

def test_train_target_col(data_train):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data_train.columns

def test_train_test_sizes(data_train, data_test):
    """Check that train and test dataframe have the same columns (but target and repay)"""
    train_size = data_train.drop(columns=["TARGET", "REPAY"]).shape[1]
    test_size = data_test.drop(columns=["TARGET", "REPAY"]).shape[1]
    assert train_size == test_size

