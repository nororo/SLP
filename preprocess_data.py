from warnings import filterwarnings

filterwarnings("ignore")

from pathlib import Path

import numpy as np
from tqdm import tqdm

tqdm.pandas()

np.random.seed(0)

from preprocessing.prep_dataset import (
    get_transaction,
    preprocess_transaction,
    proc_for_transaction_ml_model,
)

DATA_DIR = Path("../data/")
print("Data directory:", DATA_DIR.resolve())


def main():

    print("=== gen negative sample ===")
    df_transaction_data = get_transaction()
    data, item_frec_cnt_dict = preprocess_transaction(df_transaction_data)

    data = data.query("tran_date <= 365")  # training data
    print(data.shape)

    proc_for_transaction_ml_model(
        data,
        out_filename=DATA_DIR / "proc_data_transaction.parquet",
    )  # save preprocessed data for transaction ml model


if __name__ == "__main__":
    main()
