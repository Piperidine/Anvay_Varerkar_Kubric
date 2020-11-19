import requests
import pandas
import scipy
import numpy as np
import sys
import pandas as pd

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

def transform(dat):
    dat = dat.T.reset_index()
    dat.columns = ['area','price']
    dat = dat.iloc[1:]
    return dat

def get_data():
    TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"
    train = transform(pd.read_csv(TRAIN_DATA_URL))
    test = transform(pd.read_csv(TEST_DATA_URL))
    train_x = train['area']
    train_y = train['price']
    test_x = test['area']
    test_y = test['price']
    return train_x, test_x, train_y, test_y
    
def predict_price(area) -> float:
    train_x, test_x, train_y, test_y = get_data()
    # YOUR IMPLEMENTATION HERE
    np.polyfit(train_x,train_y,1)


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
