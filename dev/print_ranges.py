import sys
from datetime import timezone

sys.path.append('.')

from src.data_loader import load_meter, load_weather


def fmt_range(df):
    if df.empty:
        return 'empty'
    idx = df.index
    return f"{idx.min()} to {idx.max()}"


def main():
    ds = r'Building Data Genome Project 2 dataset'
    print('Electricity:')
    elec = load_meter('electricity', ds)
    print(fmt_range(elec))

    print('Weather:')
    w = load_weather(ds)
    print(fmt_range(w))


if __name__ == '__main__':
    main()

