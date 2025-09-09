import sys
from pprint import pprint

sys.path.append('.')

from src.data_loader import load_weather


def main():
    print('loading weather...')
    df = load_weather(r'Building Data Genome Project 2 dataset')
    print('loaded:', df.shape)
    print('index tz:', getattr(df.index, 'tz', None))
    print('columns sample:', list(df.columns)[:8])


if __name__ == '__main__':
    main()

