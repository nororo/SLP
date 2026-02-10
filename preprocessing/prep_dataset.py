import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("../data/")
print("Data directory:", DATA_DIR.resolve())


def agg_unique_keep_order(sr):
    x_list = [x for x in sr]
    return sorted(set(x_list), key=x_list.index)


def drop_list(sr):
    item_list = sr.item_id_list.copy()
    num = sr.random_order
    item_list.pop(num)
    return item_list


def random_sort(data, sort_unit_list=["user_id", "basket_id"]):
    print("=== assign order ===")
    data["order"] = data.groupby(sort_unit_list)["item_id"].cumcount()

    print("=== gen random order ===")
    data["random_order"] = data.groupby(sort_unit_list)["order"].transform(
        lambda x: np.random.permutation(len(x)),
    )

    print("=== random sorting ===")
    sort_cols = sort_unit_list + ["random_order"]
    data = data.sort_values(sort_cols)
    return data


def build_neg_table(count_dict, power=3 / 4, table_size=10000000):
    """-> [0,0,0,0,1,1,1,.....,n_item,n_item,n_item,n_item] (count of idx in the output table is distributed like freq = cnt**power/power_sum)
    https://qiita.com/jyori112/items/21958b13264f14f3b9e8
    """
    powered_sum = sum(count**power for wid, count in count_dict.items())
    table = np.zeros(shape=(table_size,), dtype=np.int32)

    idx = 0
    accum = 0.0

    for wid, count in count_dict.items():
        freq = (count**power) / powered_sum
        accum += freq * table_size
        end_idx = int(accum)
        table[idx : int(accum)] = wid
        idx = end_idx

    return table[table > 0]


# dunnhumby dataset


def get_target_item(df_transaction_data) -> list:
    # Count of discounted items
    df_transaction_data_disc_item = df_transaction_data.query(
        "COUPON_DISC<0",
    ).item_id.value_counts()
    # Items with more than 50 discounts -> target_item
    df_transaction_data_disc_item = df_transaction_data_disc_item[
        df_transaction_data_disc_item > 50
    ]
    target_exchangeable_item: list = list(
        set(df_transaction_data_disc_item.index),
    )  # unique
    print("target items: ", len(target_exchangeable_item))  # 18 items

    return target_exchangeable_item


def get_transaction():
    """Columns:
    RETAIL_DISC (Retail discount)
        Retail loyalty card program discount
        Special price offered to customers with a loyalty card
        Retail incentive discount for customers
    COUPON_DISC (Coupon discount)
        Coupon discount applied by manufacturers
        Discount applied when customers use coupons directly from manufacturers
        This discount amount is refunded to manufacturers by retailers
    COUPON_MATCH_DISC (Coupon matching discount)
        Additional discount provided by retailers for matching manufacturer coupons
        Example: If a manufacturer offers a $0.50 coupon, the retailer may offer an additional $0.50 discount
    """
    df_transaction_data = pd.read_csv(DATA_DIR / "transaction_data.csv")
    col_dict = {
        "household_key": "user_id",
        "BASKET_ID": "basket_id",
        "QUANTITY": "quantity",
        "SALES_VALUE": "price",
        "STORE_ID": "store_id",
        "PRODUCT_ID": "item_id",
        "DAY": "tran_date",
    }

    df_transaction_data = df_transaction_data.rename(columns=col_dict)

    return df_transaction_data


def proc_for_transaction_ml_model(
    data,
    out_filename: Path = Path("proc_data_shopper.parquet"),
):
    # add +1 to user_id, item_id, basket_id, seasonality, store_id to avoid 0 index
    data["user_id"] = data["user_id"] + 1
    data["item_id"] = data["item_id"] + 1
    data["basket_id"] = data["basket_id"] + 1
    data["seasonality"] = data["seasonality"] + 1
    data["store_id"] = data["store_id"] + 1

    data = random_sort(data, sort_unit_list=["user_id", "basket_id"])

    print("=== make context pool===")
    data_g = data.groupby("basket_id").agg(
        item_id_list=("item_id", agg_unique_keep_order),
    )
    data_g = data_g.assign(
        basket_length=data_g.item_id_list.progress_apply(lambda x: len(x)),
    )

    print("=== merge ===")
    data_with_basket = pd.merge(
        data,
        data_g,
        left_on="basket_id",
        right_index=True,
        how="left",
    )

    print("=== make context ===")
    data_with_basket["other_basket_prods"] = data_with_basket.progress_apply(
        drop_list,
        axis=1,
    )

    print("=== save dataset ===")
    data_with_basket.to_parquet(DATA_DIR / "proc_data_transaction.parquet")


def get_major_items(df_transaction_data) -> list:
    """Evaluation dataset item <- major item or exchangeable item"""
    # Number of unique users who purchased each item
    item_cnt_tbl = df_transaction_data.groupby("item_id").agg(
        {"user_id": pd.Series.nunique},
    )
    # Major items with more than 200 unique users
    item_cnt_tbl_200_set = set(item_cnt_tbl.query("user_id>200").index)
    print(len(item_cnt_tbl_200_set))
    # Add target_item
    target_item = get_target_item(df_transaction_data)
    item_set = item_cnt_tbl_200_set | set(target_item)
    return list(item_set)


def preprocess_transaction(df_transaction_data, save_le: bool = False):
    major_item_list = get_major_items(df_transaction_data)
    df_transaction_data = df_transaction_data.query("item_id in @major_item_list")
    item_all = df_transaction_data.groupby("item_id").agg(
        {"basket_id": pd.Series.nunique},
    )

    df_transaction_data["time_frame"] = df_transaction_data["TRANS_TIME"].apply(
        lambda x: (
            "06_10"
            if (x >= 600 and x <= 1059)
            else "11_15"
            if (x >= 1100 and x <= 1559)
            else "16_23"
            if (x >= 1600 and x <= 2359)
            else "24_05"
        ),
    )
    df_transaction_data["day_of_week"] = df_transaction_data["WEEK_NO"] % 7
    df_transaction_data["seasonality_txt"] = (
        df_transaction_data["day_of_week"].astype(str)
        + "_"
        + df_transaction_data["time_frame"]
    )

    le_seasonality = LabelEncoder()
    df_transaction_data["seasonality"] = le_seasonality.fit_transform(
        df_transaction_data["seasonality_txt"],
    )
    if save_le:
        with open(DATA_DIR / "prep_seasonality_le.pkl", "wb") as f:
            pickle.dump(le_seasonality, f)

    le_user = LabelEncoder()
    df_transaction_data["user_id"] = le_user.fit_transform(
        df_transaction_data["user_id"],
    )
    print(df_transaction_data["user_id"].max())
    if save_le:
        with open(DATA_DIR / "prep_user_id_le.pkl", "wb") as f:
            pickle.dump(le_user, f)

    le_item = LabelEncoder()
    df_transaction_data["item_id"] = le_item.fit_transform(
        df_transaction_data["item_id"],
    )
    if save_le:
        with open(DATA_DIR / "prep_item_id_le.pkl", "wb") as f:
            pickle.dump(le_item, f)
    print(df_transaction_data["item_id"].max())

    item_count_g_f = item_all.query("index in @major_item_list")
    item_count_g_f["idx"] = le_item.fit_transform(item_count_g_f.index)
    if save_le:
        item_count_g_f.to_parquet(DATA_DIR / "prep_dunn_item_count_new.parquet")

    le_basket = LabelEncoder()
    df_transaction_data["basket_id"] = le_basket.fit_transform(
        df_transaction_data["basket_id"],
    )
    if save_le:
        with open(DATA_DIR / "prep_basket_id_le.pkl", "wb") as f:
            pickle.dump(le_basket, f)
    print(df_transaction_data["basket_id"].max())

    le_store = LabelEncoder()
    df_transaction_data["store_id"] = le_store.fit_transform(
        df_transaction_data["store_id"],
    )
    if save_le:
        with open(DATA_DIR / "prep_store_id_le.pkl", "wb") as f:
            pickle.dump(le_store, f)
    print(df_transaction_data["store_id"].max())

    item_frec_cnt_dict = item_count_g_f.set_index("idx")["basket_id"].to_dict()

    return df_transaction_data, item_frec_cnt_dict
