import argparse
import pandas as pd
from pipeline import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script which utilizes a model trained to predict price."
                    "Script created using automl-gs (https://github.com/minimaxir/automl-gs)")
    parser.add_argument('-d', '--data',  help="Input dataset (must be a .csv)")
    parser.add_argument(
        '-m', '--mode',  help='Mode (either "train" or "predict")')
    parser.add_argument(
        '-s', '--split',  help="Train/Validation Split (if training)",
        default=0.7)
    parser.add_argument(
        '-e', '--epochs',  help="# of Epochs (if training)",
        default=20)
    parser.add_argument(
        '-c', '--context',  help="Context for running script (used during automl-gs training)",
        default='standalone')
    parser.add_argument(
        '-t', '--type',  help="Format for predictions (either csv or json)",
        default='csv')
    args = parser.parse_args()

    cols = ["make",
            "price",
            "body",
            "mileage",
            "engV",
            "registration",
            "year",
            "drive"
            ]
    dtypes = {'make': 'float64', 'price': 'float64', 'body': 'str', 'mileage': 'float64',
              'engV': 'float64', 'registration': 'str', 'year': 'float64', 'drive': 'str'}

    df = pd.read_csv(args.data, parse_dates=True,
                     usecols=cols,
                     dtype=dtypes)

    if args.mode == 'train':
        build_encoders(df)
        encoders = load_encoders()
        model = build_model(encoders)
        model_train(df, encoders, args, model)
    elif args.mode == 'predict':
        encoders = load_encoders()
        model = build_model(encoders)
        model.load_weights('model_weights.hdf5')
        predictions = model_predict(df, model, encoders)
        if args.type == 'csv':
            predictions.to_csv('predictions.csv', index=False)
        if args.type == 'json':
            with open('predictions.json', 'w', encoding='utf-8') as f:
                f.write(predictions.to_json(orient="records"))
