from warnings import filterwarnings

filterwarnings("ignore")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import random

np.random.seed(0)


from warnings import filterwarnings

filterwarnings("ignore")


import numpy as np
from tqdm import tqdm

tqdm.pandas()
np.random.seed(0)


from common.prep_dataset import agg_unique_keep_order, random_sort

from copurchase.copurchase import proc_cp_basket, proc_cp_user
from preprocessing.prep_dataset import (
    agg_unique_keep_order,
    build_neg_table,
    get_transaction,
    preprocess_transaction,
    random_sort,
)
from slp.slp import (
    get_shopping_embedding,
    splp_2nd_order_proximity,
    splp_shopping_matching,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_split", default="time")
    parser.add_argument("--result_dir", default="../results")
    parser.add_argument("--input_dir", default="../data")
    parser.add_argument("--experiment_name", default="tmp")
    parser.add_argument("--cfg_path", default="../cfgs/tmp.yaml")
    parser.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="number of parallel processing (1=no parallel processing)",
    )

    args = parser.parse_args()

    return args


# Set data directory and show its absolute path for debugging
DATA_DIR = Path("../data/")
print("Data directory:", DATA_DIR.resolve())


def proc_slp_sop(
    data,
    item_frec_cnt_dict: dict,
    out_filename: Path = Path("proc_data_subst2vec.parquet"),
):
    # add +1 to user_id, item_id, basket_id, seasonality, store_id to avoid 0 index
    item_frec_cnt_dict = {idx + 1: cnt for idx, cnt in item_frec_cnt_dict.items()}

    negative_sample_table = build_neg_table(
        item_frec_cnt_dict,
        power=0.75,
        table_size=1000000000,
    )  # 00
    data["user_id"] = data["user_id"] + 1
    data["item_id"] = data["item_id"] + 1
    data["basket_id"] = data["basket_id"] + 1

    data = random_sort(data, sort_unit_list=["user_id", "basket_id"])
    data_g = data.groupby("basket_id").agg(
        item_id_list=("item_id", agg_unique_keep_order),
        user_id=("user_id", "first"),
        price=("price", "first"),
    )
    data_g = data_g.assign(
        basket_length=data_g.item_id_list.progress_apply(lambda x: len(x)),
    )

    data_g = data_g.query("basket_length>1")
    data_g2 = (
        data_g.groupby("user_id")
        .progress_apply(
            splp_2nd_order_proximity,
        )
        .to_frame()
    )
    data_g2.columns = ["sub_item_id_list"]
    data_g2 = data_g2.query(
        "not sub_item_id_list.isnull()",
    )

    data_g2 = data_g2.explode("sub_item_id_list")
    data_g2["item_1"] = data_g2.sub_item_id_list.apply(lambda x: x["query"])
    data_g2["item_2"] = data_g2.sub_item_id_list.apply(lambda x: x["substitute"])
    data_g2["item_c"] = data_g2.sub_item_id_list.apply(lambda x: x["comp"])

    data_g2 = data_g2.drop("sub_item_id_list", axis=1)
    print("=== save dataset ===")
    data_g2.to_parquet(
        out_filename,
    )


def proc_slp_sm(
    data,
    item_frec_cnt_dict,
    method_name: str = "visit2vec",
    out_filename: Path = Path("proc_data_subst2vec_by_visit2vec.parquet"),
):
    # add +1 to item_id and basket_id to avoid 0 index
    data["user_id"] = data["user_id"] + 1
    data["item_id"] = data["item_id"] + 1
    data["basket_id"] = data["basket_id"] + 1

    # get_shopping_embedding
    emb_basket = get_shopping_embedding()

    # preprocessing
    data = random_sort(data, sort_unit_list=["user_id", "basket_id"])
    data_g = data.groupby("basket_id", as_index=False).agg(
        item_id_list=("item_id", agg_unique_keep_order),
        user_id=("user_id", "first"),
        price=("price", "first"),
    )
    data_g = data_g.assign(
        basket_length=data_g.item_id_list.progress_apply(lambda x: len(x)),
    )
    data_g = data_g.query("basket_length>1")
    print("=== merge visit2vec data ===")
    data_g = data_g.merge(emb_basket, left_on="basket_id", right_index=True)
    del emb_basket
    print("=== gen substitute instance ===")
    data_g2 = (
        data_g.groupby("user_id")
        .progress_apply(
            splp_shopping_matching,
        )
        .to_frame()
    )

    data_g2.columns = ["sub_item_id_list"]
    data_g2 = data_g2.query(
        "not sub_item_id_list.isnull()",
    )

    data_g2 = data_g2.explode("sub_item_id_list")
    data_g2["item_1"] = data_g2.sub_item_id_list.apply(lambda x: x["query"])
    data_g2["item_2"] = data_g2.sub_item_id_list.apply(lambda x: x["substitute"])
    data_g2["item_c"] = data_g2.sub_item_id_list.apply(lambda x: x["comp"])
    data_g2["distance"] = data_g2.sub_item_id_list.apply(lambda x: x["distance"])

    print("=== save dataset ===")

    data_g2.to_parquet(
        out_filename,
    )


def post_process_sampling(
    input_file: Path = Path("proc_data_subst2vec_by_visit2vec.parquet"),
):
    """mainのアウトプットを入力として、(user_id, item_1)についてitem_2を5個サンプリングする処理"""
    import random

    print("=== post process sampling ===")

    print(f"Loading: {input_file}")
    data = pd.read_parquet(input_file)

    print(f"Original data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # (user_id, item_1) group by and sample item_2 up to 5 items
    # item_x keep the items from the original row, other columns (price, basket_id, etc.) keep the first value
    agg_dict = {
        "item_2": ("item_2", lambda x: random.sample(list(x), min(len(x), 5))),
    }

    # keep other columns (item_1, item_2, user_id, item_x以外)
    # item_x keep the items from the original row for each item_2
    other_cols = [
        col
        for col in data.columns
        if col not in ["user_id", "item_1", "item_2", "item_x"]
    ]
    for col in other_cols:
        agg_dict[col] = (col, "first")

    if "item_x" in data.columns:
        agg_dict["item_x"] = ("item_x", lambda x: list(x))

    print("=== groupby and sampling ===")
    data_sampled = data.groupby(["user_id", "item_1"]).agg(**agg_dict).reset_index()

    print(f"After groupby shape: {data_sampled.shape}")

    # explode item_2 and item_x (keep item_x for each item_2)
    print("=== exploding item_2 and item_x ===")
    if "item_x" in data_sampled.columns:
        # check if the length of item_2 and item_x are the same
        # item_2 is sampled 5 items, item_x also get 5 items from the original row
        data_sampled["item_2_item_x"] = data_sampled.apply(
            lambda row: list(zip(row["item_2"], row["item_x"])),
            axis=1,
        )
        data_sampled = data_sampled.explode("item_2_item_x")
        data_sampled["item_2"] = data_sampled["item_2_item_x"].apply(lambda x: x[0])
        data_sampled["item_x"] = data_sampled["item_2_item_x"].apply(lambda x: x[1])
        data_sampled = data_sampled.drop("item_2_item_x", axis=1)
    else:
        data_sampled = data_sampled.explode("item_2")

    print(f"After explode shape: {data_sampled.shape}")

    # 保存
    output_file = input_file.with_name(
        input_file.name.replace(".parquet", "_sampled.parquet"),
    )
    print(f"=== Saving to: {output_file} ===")
    data_sampled.to_parquet(output_file, index=False)

    print("=== post process sampling completed ===")
    return data_sampled


def splp_post_processing(
    input_file: Path = Path("proc_data_subst2vec_by_visit2vec.parquet"),
    threshold_quantile: float = 0.75,
):
    df_transaction_data = get_transaction()

    _, item_frec_cnt_dict = preprocess_transaction(df_transaction_data, save_le=False)
    item_frec_cnt_dict = {idx + 1: cnt for idx, cnt in item_frec_cnt_dict.items()}

    negative_sample_table = build_neg_table(
        item_frec_cnt_dict,
        power=0.75,
        table_size=1000000000,
    )
    # k0
    df_pool_visit2vec = pd.read_parquet(input_file)

    if threshold_quantile < 1:
        threshold = df_pool_visit2vec.distance.quantile(threshold_quantile)
        df_pool_visit2vec = df_pool_visit2vec.query("distance <= @threshold")

    if "distance" in df_pool_visit2vec.columns:
        del df_pool_visit2vec["distance"]

    df_pool_visit2vec = (
        df_pool_visit2vec.groupby(["user_id", "item_1"])
        .agg(
            item_2=("item_2", lambda x: random.sample(list(x), min(len(x), 5))),
            item_c=("item_c", "first"),
        )
        .explode("item_2")
    )
    # (user_id, item_1) pair by pair create list of item_2
    df_pool_visit2vec = df_pool_visit2vec.merge(
        df_pool_visit2vec.groupby(["user_id", "item_1"]).agg(
            subst_items=("item_2", lambda x: list(x)),
        ),
        left_on=["user_id", "item_1"],
        right_index=True,
        how="left",
    )
    print("Converting subst_items to sets...")
    df_pool_visit2vec["subst_items_set"] = df_pool_visit2vec["subst_items"].apply(set)

    batch_neg_pool = np.random.choice(
        negative_sample_table,
        [len(df_pool_visit2vec), 30],
    )

    batch_subst_sets = df_pool_visit2vec["subst_items_set"].values
    batch_item_x = [
        list(set(neg_pool) - subst_set)
        for neg_pool, subst_set in zip(batch_neg_pool, batch_subst_sets)
    ]
    # 結果をDataFrameに格納
    batch_df = pd.DataFrame(
        {
            "neg_pool": [list(row) for row in batch_neg_pool],
            "item_x": batch_item_x,
        },
    )

    df_pool_visit2vec["neg_pool"] = batch_df["neg_pool"].values
    df_pool_visit2vec["item_x"] = batch_df["item_x"].values

    df_pool_visit2vec.reset_index()[
        ["user_id", "item_1", "item_2", "item_x"]
    ].to_parquet(
        input_file.with_name(input_file.name.replace(".parquet", "_k0.parquet")),
    )

    df_pool_visit2vec["item_x"] = df_pool_visit2vec["item_c"].apply(
        lambda x: list(x[0:1]),
    ) + df_pool_visit2vec["neg_pool"].apply(lambda x: list(x[0:19]))

    df_pool_visit2vec.reset_index()[
        ["user_id", "item_1", "item_2", "item_x"]
    ].to_parquet(
        input_file.with_name(input_file.name.replace(".parquet", "_k1.parquet")),
    )


def main():
    n_parallel = 8

    print("=== gen negative sample ===")
    df_transaction_data = get_transaction()
    data, item_frec_cnt_dict = preprocess_transaction(df_transaction_data)

    data = data.query("tran_date <= 365")  # training data
    print(data.shape)

    print("=== proc_prod2vec_user ===")
    method_name = "prod2vec_user"
    (DATA_DIR / f"{method_name}").mkdir(parents=True, exist_ok=True)
    proc_cp_user(
        data,
        item_frec_cnt_dict,
        neg_sample_size=20,
        out_filename=DATA_DIR / f"{method_name}" / "proc_data_prod2vec_usr.parquet",
        n_parallel=n_parallel,
    )
    post_process_sampling(
        DATA_DIR / f"{method_name}" / "proc_data_prod2vec_usr.parquet",
    )
    print("=== proc_prod2vec_basket ===")
    method_name = "prod2vec_bsk"
    (DATA_DIR / f"{method_name}").mkdir(parents=True, exist_ok=True)
    proc_cp_basket(
        data,
        item_frec_cnt_dict,
        neg_sample_size=20,
        out_filename=DATA_DIR / f"{method_name}" / "proc_data_prod2vec_bsk.parquet",
    )
    post_process_sampling(
        DATA_DIR / f"{method_name}" / "proc_data_prod2vec_bsk.parquet",
    )
    method_name = "subst2vec_sop"
    (DATA_DIR / f"{method_name}").mkdir(parents=True, exist_ok=True)
    print("=== proc_subst2vec ===")
    proc_slp_sop(
        data,
        item_frec_cnt_dict,
        out_filename=DATA_DIR / f"{method_name}" / "proc_data_subst2vec_sop.parquet",
    )
    splp_post_processing(
        DATA_DIR / f"{method_name}" / "proc_data_subst2vec_sop.parquet",
        threshold_quantile=1,
    )
    print("=== proc_subst2vec_by_visit2vec 2order k0 ===")
    method_name = "subst2vec_visit2vec_b64"
    (DATA_DIR / f"{method_name}").mkdir(parents=True, exist_ok=True)
    proc_slp_sm(
        data,
        item_frec_cnt_dict,
        out_filename=DATA_DIR
        / f"{method_name}"
        / "proc_data_subst2vec_by_visit2vec.parquet",
    )
    splp_post_processing(
        DATA_DIR / f"{method_name}" / "proc_data_subst2vec_by_visit2vec.parquet",
        threshold_quantile=0.75,
    )


if __name__ == "__main__":
    main()
