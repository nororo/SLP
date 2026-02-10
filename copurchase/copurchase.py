from warnings import filterwarnings

filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

tqdm.pandas()
np.random.seed(0)

from itertools import permutations

from common.prep_dataset import agg_unique_keep_order, build_neg_table, random_sort


def process_user_data(user_data, negative_sample_table, neg_sample_size=10):
    user_id, data_row = user_data

    item_id_list = data_row["item_id_list"]
    history_length = data_row["history_length"]

    if history_length <= 1:
        return None

    comb_list = list(permutations(item_id_list, 2))
    num_combs = len(comb_list)

    batch_neg_pool = np.random.choice(negative_sample_table, [num_combs, 30])

    item_id_set = set(item_id_list)

    results = []
    for idx, comb in enumerate(comb_list):
        item_1, item_2 = comb

        item_x = list(set(batch_neg_pool[idx]) - item_id_set)[:neg_sample_size]

        results.append(
            {
                "user_id": user_id,
                "item_1": item_1,
                "item_2": item_2,
                "item_x": item_x,
            },
        )

    return results


def proc_cp_user(
    data,
    item_frec_cnt_dict,
    neg_sample_size=10,
    out_filename: Path = Path("proc_data_prod2vec_usr.parquet"),
    n_parallel=1,
):
    # add +1 to item_id and basket_id to avoid 0 index
    item_frec_cnt_dict = {idx + 1: cnt for idx, cnt in item_frec_cnt_dict.items()}
    negative_sample_table = build_neg_table(
        item_frec_cnt_dict,
        power=0.75,
        table_size=1000000000,
    )
    data["user_id"] = data["user_id"] + 1
    data["item_id"] = data["item_id"] + 1
    data["basket_id"] = data["basket_id"] + 1
    data = random_sort(data, sort_unit_list=["user_id"])
    data_g = data.groupby("user_id").agg(
        item_id_list=("item_id", agg_unique_keep_order),
    )
    data_g = data_g.assign(
        history_length=data_g.item_id_list.progress_apply(lambda x: len(x)),
    )
    data_g = data_g.query("history_length>1")

    if n_parallel > 1:
        user_ids = list(data_g.index)

        results = Parallel(n_jobs=n_parallel, verbose=1)(
            delayed(process_user_data)(
                (user_id, data_g.loc[user_id]),
                negative_sample_table=negative_sample_table,
                neg_sample_size=neg_sample_size,
            )
            for user_id in user_ids
        )

        all_results = []
        for result in results:
            if result is not None:
                all_results.extend(result)

        data_g = pd.DataFrame(all_results)
    else:
        data_g["item_id_comb"] = data_g.item_id_list.progress_apply(
            lambda x: list(permutations(x, 2)),
        )
        data_g = data_g.explode("item_id_comb")

        data_g["item_1"] = data_g.item_id_comb.apply(lambda x: x[0])
        data_g["item_2"] = data_g.item_id_comb.apply(lambda x: x[1])
        data_g["item_x"] = data_g.progress_apply(
            lambda x: [
                neg_item
                for neg_item in np.random.choice(negative_sample_table, 30)
                if neg_item not in x.item_id_list
            ][:neg_sample_size],
            axis=1,
        )

        data_g = data_g.drop(["item_id_list", "history_length", "item_id_comb"], axis=1)
        data_g = data_g.reset_index()

    print("=== save dataset ===")
    data_g.to_parquet(out_filename, index=False)


def proc_cp_basket(
    data,
    item_frec_cnt_dict,
    neg_sample_size=10,
    out_filename: Path = Path("proc_data_prod2vec_bsk.parquet"),
):
    # add +1 to item_id and basket_id to avoid 0 index
    item_frec_cnt_dict = {idx + 1: cnt for idx, cnt in item_frec_cnt_dict.items()}
    negative_sample_table = build_neg_table(
        item_frec_cnt_dict,
        power=0.75,
        table_size=1000000000,
    )
    data["user_id"] = data["user_id"] + 1
    data["item_id"] = data["item_id"] + 1
    data["basket_id"] = data["basket_id"] + 1
    # random sort by user_id and basket_id
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
    data_g["item_id_comb"] = data_g.item_id_list.progress_apply(
        lambda x: list(permutations(x, 2)),
    )

    data_g = data_g.explode("item_id_comb")

    data_g["item_1"] = data_g.item_id_comb.apply(lambda x: x[0])
    data_g["item_2"] = data_g.item_id_comb.apply(lambda x: x[1])
    data_g["item_x"] = data_g.progress_apply(
        lambda x: [
            neg_item
            for neg_item in np.random.choice(negative_sample_table, 30)
            if neg_item not in x.item_id_list
        ][:neg_sample_size],
        axis=1,
    )

    data_g = data_g.drop(["item_id_list", "basket_length", "item_id_comb"], axis=1)

    print("=== save dataset ===")
    data_g.reset_index().to_parquet(out_filename)
