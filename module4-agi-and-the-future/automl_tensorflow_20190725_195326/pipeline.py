from tensorflow.train import cosine_decay, AdamOptimizer
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(encoders):
    """Builds and compiles the model from scratch.

    # Arguments
        encoders: dict of encoders (used to set size of text/categorical inputs)

    # Returns
        model: A compiled model which can be used to train or predict.
    """

    # make
    input_make = Input(shape=(1,), name="input_make")

    # body
    input_body_size = len(encoders['body_encoder'].classes_)
    input_body = Input(
        shape=(input_body_size if input_body_size != 2 else 1,), name="input_body")

    # mileage
    input_mileage = Input(shape=(1,), name="input_mileage")

    # engV
    input_engv = Input(shape=(1,), name="input_engv")

    # registration
    input_registration_size = len(encoders['registration_encoder'].classes_)
    input_registration = Input(shape=(
        input_registration_size if input_registration_size != 2 else 1,), name="input_registration")

    # year
    input_year = Input(shape=(1,), name="input_year")

    # drive
    input_drive_size = len(encoders['drive_encoder'].classes_)
    input_drive = Input(
        shape=(input_drive_size if input_drive_size != 2 else 1,), name="input_drive")

    # Combine all the inputs into a single layer
    concat = concatenate([
        input_make,
        input_body,
        input_mileage,
        input_engv,
        input_registration,
        input_year,
        input_drive
    ], name="concat")

    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense(256, activation='selu', name='hidden_1',
                   kernel_regularizer=l2(1e-3))(concat)
    hidden = AlphaDropout(0.0, name="dropout_1")(hidden)

    for i in range(2-1):
        hidden = Dense(128, activation="selu", name="hidden_{}".format(
            i+2), kernel_regularizer=l2(1e-3))(hidden)
        hidden = AlphaDropout(0.0, name="dropout_{}".format(i+2))(hidden)
    output = Dense(1, name="output", kernel_regularizer=None)(hidden)

    # Build and compile the model.
    model = Model(inputs=[
        input_make,
        input_body,
        input_mileage,
        input_engv,
        input_registration,
        input_year,
        input_drive
    ],
        outputs=[output])
    model.compile(loss="msle",
                  optimizer=AdamWOptimizer(learning_rate=0.1,
                                           weight_decay=0.01))

    return model


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # make
    make_enc = df['make']
    make_encoder = StandardScaler()
    make_encoder_attrs = ['mean_', 'var_', 'scale_']
    make_encoder.fit(df['make'].values.reshape(-1, 1))

    make_encoder_dict = {attr: getattr(make_encoder, attr).tolist()
                         for attr in make_encoder_attrs}

    with open(os.path.join('encoders', 'make_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(make_encoder_dict, outfile, ensure_ascii=False)

    # body
    body_counts = df['body'].value_counts()
    body_perc = max(floor(0.5 * body_counts.size), 1)
    body_top = np.array(body_counts.index[0:body_perc], dtype=object)
    body_encoder = LabelBinarizer()
    body_encoder.fit(body_top)

    with open(os.path.join('encoders', 'body_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(body_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # mileage
    mileage_enc = df['mileage']
    mileage_encoder = StandardScaler()
    mileage_encoder_attrs = ['mean_', 'var_', 'scale_']
    mileage_encoder.fit(df['mileage'].values.reshape(-1, 1))

    mileage_encoder_dict = {attr: getattr(mileage_encoder, attr).tolist()
                            for attr in mileage_encoder_attrs}

    with open(os.path.join('encoders', 'mileage_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(mileage_encoder_dict, outfile, ensure_ascii=False)

    # engV
    engv_enc = df['engV']
    engv_encoder = StandardScaler()
    engv_encoder_attrs = ['mean_', 'var_', 'scale_']
    engv_encoder.fit(df['engV'].values.reshape(-1, 1))

    engv_encoder_dict = {attr: getattr(engv_encoder, attr).tolist()
                         for attr in engv_encoder_attrs}

    with open(os.path.join('encoders', 'engv_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(engv_encoder_dict, outfile, ensure_ascii=False)

    # registration
    registration_counts = df['registration'].value_counts()
    registration_perc = max(floor(0.5 * registration_counts.size), 1)
    registration_top = np.array(
        registration_counts.index[0:registration_perc], dtype=object)
    registration_encoder = LabelBinarizer()
    registration_encoder.fit(registration_top)

    with open(os.path.join('encoders', 'registration_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(registration_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # year
    year_enc = df['year']
    year_encoder = StandardScaler()
    year_encoder_attrs = ['mean_', 'var_', 'scale_']
    year_encoder.fit(df['year'].values.reshape(-1, 1))

    year_encoder_dict = {attr: getattr(year_encoder, attr).tolist()
                         for attr in year_encoder_attrs}

    with open(os.path.join('encoders', 'year_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(year_encoder_dict, outfile, ensure_ascii=False)

    # drive
    drive_counts = df['drive'].value_counts()
    drive_perc = max(floor(0.5 * drive_counts.size), 1)
    drive_top = np.array(drive_counts.index[0:drive_perc], dtype=object)
    drive_encoder = LabelBinarizer()
    drive_encoder.fit(drive_top)

    with open(os.path.join('encoders', 'drive_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(drive_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Target Field: price


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # make
    make_encoder = StandardScaler()
    make_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'make_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        make_attrs = json.load(infile)

    for attr, value in make_attrs.items():
        setattr(make_encoder, attr, value)
    encoders['make_encoder'] = make_encoder

    # body
    body_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'body_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        body_encoder.classes_ = json.load(infile)
    encoders['body_encoder'] = body_encoder

    # mileage
    mileage_encoder = StandardScaler()
    mileage_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'mileage_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        mileage_attrs = json.load(infile)

    for attr, value in mileage_attrs.items():
        setattr(mileage_encoder, attr, value)
    encoders['mileage_encoder'] = mileage_encoder

    # engV
    engv_encoder = StandardScaler()
    engv_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'engv_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        engv_attrs = json.load(infile)

    for attr, value in engv_attrs.items():
        setattr(engv_encoder, attr, value)
    encoders['engv_encoder'] = engv_encoder

    # registration
    registration_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'registration_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        registration_encoder.classes_ = json.load(infile)
    encoders['registration_encoder'] = registration_encoder

    # year
    year_encoder = StandardScaler()
    year_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'year_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        year_attrs = json.load(infile)

    for attr, value in year_attrs.items():
        setattr(year_encoder, attr, value)
    encoders['year_encoder'] = year_encoder

    # drive
    drive_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'drive_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        drive_encoder.classes_ = json.load(infile)
    encoders['drive_encoder'] = drive_encoder

    # Target Field: price

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # make
    make_enc = df['make'].values.reshape(-1, 1)
    make_enc = encoders['make_encoder'].transform(make_enc)

    # body
    body_enc = df['body'].values
    body_enc = encoders['body_encoder'].transform(body_enc)

    # mileage
    mileage_enc = df['mileage'].values.reshape(-1, 1)
    mileage_enc = encoders['mileage_encoder'].transform(mileage_enc)

    # engV
    engv_enc = df['engV'].values.reshape(-1, 1)
    engv_enc = encoders['engv_encoder'].transform(engv_enc)

    # registration
    registration_enc = df['registration'].values
    registration_enc = encoders['registration_encoder'].transform(
        registration_enc)

    # year
    year_enc = df['year'].values.reshape(-1, 1)
    year_enc = encoders['year_encoder'].transform(year_enc)

    # drive
    drive_enc = df['drive'].values
    drive_enc = encoders['drive_encoder'].transform(drive_enc)

    data_enc = [make_enc,
                body_enc,
                mileage_enc,
                engv_enc,
                registration_enc,
                year_enc,
                drive_enc
                ]

    if process_target:
        # Target Field: price
        price_enc = df['price'].values

        return (data_enc, price_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    headers = ['price']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """
    X, y = process_data(df, encoders)

    split = ShuffleSplit(n_splits=1, train_size=args.split,
                         test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        X_train = [field[train_indices, ] for field in X]
        X_val = [field[val_indices, ] for field in X]
        y_train = y[train_indices, ]
        y_val = y[val_indices, ]

    meta = meta_callback(args, X_val, y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              callbacks=[meta],
              batch_size=256)


class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and metrics after each training epoch.

    Metrics metadata is saved in the /metadata folder.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + ['mse', 'mae', 'r_2'])
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r_2 = r2_score(y_true, y_pred)

        metrics = [mse, mae, r_2]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("\nEPOCH_END")
