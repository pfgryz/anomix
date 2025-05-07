from src.data_loaders import get_all_datasets


def main():
    datasets = get_all_datasets()
    for dataset in datasets:
        x, y = dataset.get_x_y_tuple()
        print("Dataset: ", dataset.name)
        print("X: ", x)
        print("Y: ", y)


if __name__ == "__main__":
    main()
