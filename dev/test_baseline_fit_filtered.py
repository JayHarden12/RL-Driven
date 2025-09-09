import time
import numpy as np
import pandas as pd

from src.data_loader import get_building_ids, get_building_series
from src.preprocess import add_time_features, merge_weather
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    DS = r'Building Data Genome Project 2 dataset'
    blds = get_building_ids(DS)
    b = blds[0]
    print('building:', b)

    s = (
        get_building_series(b, DS, source='electricity')
        .rename('electricity')
        .to_frame()
        .asfreq('H')
        .interpolate(limit_direction='both')
    )
    feats = merge_weather(add_time_features(s), DS, building_id=b)
    print('feats shape:', feats.shape)

    X = feats[[c for c in feats.columns if c != 'electricity']].select_dtypes(include=[np.number]).fillna(0.0)
    y = feats['electricity'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    t1 = time.time()
    rf.fit(X_train, y_train)
    t2 = time.time()
    print('fit secs:', t2 - t1, 'score:', rf.score(X_test, y_test))


if __name__ == '__main__':
    main()

