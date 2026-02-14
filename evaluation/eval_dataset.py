# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set data directory and show its absolute path for debugging
DATA_DIR = Path("../../data/")
RESULT_DIR = Path("../../results/")
print("Data directory:", DATA_DIR.resolve())
print("Result directory:", RESULT_DIR.resolve())

from lib.dunnhumby_dataset import get_major_items, get_target_item, get_transaction


def get_coupon_red_user_day_key(
    df_transaction_data: pd.DataFrame,
    target_item: str,
) -> pd.DataFrame:
    # make user_day key for getting purchased items after 1 week of coupon use
    user_day_pair = df_transaction_data.query(
        "COUPON_DISC<0 and item_id==@target_item",
    )[["user_id", "tran_date"]].drop_duplicates()
    return user_day_pair


def get_post_pre_diff(
    df_transaction_data_comb,
    major_item_list: list,
    base_column_name: str = "pre_4w_term_start",
    pre_weeks: int = 4,
):
    # extract target post tran (major item)
    df_transaction_data_comb_post = df_transaction_data_comb.query(
        "tran_date >= post_term_start and tran_date <= post_term_end and item_id in @major_item_list",
    )
    # basket x cpn_use
    assert (
        df_transaction_data_comb_post.duplicated(
            subset=["user_id", "item_id", "basket_id", "user_cpn_day_key"],
        ).sum()
        == 0
    )
    # Y | T=1
    post_purchase = df_transaction_data_comb_post.groupby(
        ["user_cpn_day_key", "item_id"],
    ).agg(
        basket_cnt_post=("basket_id", pd.Series.nunique),
    )

    # extract target pre tran (major item)
    # for query replace column name
    df_transaction_data_comb = df_transaction_data_comb.assign(
        pre_term_start=df_transaction_data_comb[base_column_name],
    )
    df_transaction_data_comb_pre4w = df_transaction_data_comb.query(
        "tran_date >= pre_term_start and tran_date <= pre_term_end and item_id in @major_item_list",
    )
    assert (
        df_transaction_data_comb_pre4w.duplicated(
            subset=["user_id", "item_id", "basket_id", "user_cpn_day_key"],
        ).sum()
        == 0
    )

    # Y_pre | T=1
    pre_purchase = (
        df_transaction_data_comb_pre4w.groupby(
            ["user_cpn_day_key", "item_id"],
        ).agg(
            basket_cnt_pre=("basket_id", pd.Series.nunique),
        )
        / pre_weeks
    )
    diff_purchase = pd.merge(
        post_purchase,
        pre_purchase,
        left_index=True,
        right_index=True,
        how="outer",
    ).fillna(0)

    diff_purchase = diff_purchase.assign(
        diff_post_pre=(diff_purchase.basket_cnt_post - diff_purchase.basket_cnt_pre),
    )
    diff_purchase_g = (
        diff_purchase.reset_index()
        .groupby("item_id")
        .agg({"diff_post_pre": "sum", "basket_cnt_pre": "sum"})
        .sort_values("diff_post_pre")
    )
    return diff_purchase_g


def get_t1(
    df_transaction_data: pd.DataFrame,
    major_item_list: list,
    target_item: str,
    eval_start_date: int = 366,
):
    cpn_red_user_day_pair = get_coupon_red_user_day_key(
        df_transaction_data,
        target_item,
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

    print("cpn_red_user_day_pair: ", len(cpn_red_user_day_pair))
    cpn_red_user_day_pair = cpn_red_user_day_pair.assign(
        user_cpn_day_key=cpn_red_user_day_pair.user_id.astype(str)
        + "_"
        + (cpn_red_user_day_pair.tran_date).astype(str),
        post_term_start=cpn_red_user_day_pair.tran_date,
        post_term_end=cpn_red_user_day_pair.tran_date + 6,
        pre_term_end=cpn_red_user_day_pair.tran_date - 1,
        pre_1w_term_start=cpn_red_user_day_pair.tran_date - 7,
        pre_2w_term_start=cpn_red_user_day_pair.tran_date - 14,
        pre_4w_term_start=cpn_red_user_day_pair.tran_date - 7 * 4,
        pre_8w_term_start=cpn_red_user_day_pair.tran_date - 7 * 8,
        weekday=cpn_red_user_day_pair.tran_date % 7,
    )

    # add coupon redempt date to transaction -> (transaction(user) x coupon_redempt)
    df_transaction_data_comb = pd.merge(
        df_transaction_data,
        cpn_red_user_day_pair.query("tran_date>=@eval_start_date").drop(
            "tran_date",
            axis=1,
        ),
        left_on="user_id",
        right_on="user_id",
        how="inner",
    )
    print("size df_transaction_data_comb: ", len(df_transaction_data_comb))

    simple_post_pre_diff = get_post_pre_diff(
        df_transaction_data_comb,
        major_item_list,
        base_column_name="pre_4w_term_start",
        pre_weeks=4,
    )
    return simple_post_pre_diff


def main():
    df_transaction_data = get_transaction()
    # quantity hist (<20)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(2, 1, 1)
    np.log(df_transaction_data.query("quantity<=20").quantity.value_counts()).plot.bar(
        ax=ax,
    )
    # quantity hist (>=20)
    ax = fig.add_subplot(2, 1, 2)
    df_transaction_data.query("quantity>20").quantity.hist(bins=200, ax=ax)
    plt.show()
    # -> quantity is in liters and other units -> basket_id unique cnt -> Y
    # -> quantity is in liters and other units -> basket_id unique cnt -> Y
    target_exchangeable_item = get_target_item(
        df_transaction_data.query("tran_date>=366"),
        coupon_cnt_thr=20,
    )
    major_item_list = get_major_items(df_transaction_data)
    for target_item in target_exchangeable_item:
        dataset_t1 = get_t1(
            df_transaction_data,
            major_item_list,
            target_item,
            eval_start_date=366,
        )
        dataset_t1 = dataset_t1.rename(columns={"diff_post_pre": "pred"})

        dataset_t1[["pred"]].to_csv(
            RESULT_DIR / "eval_did" / f"did_7days_{target_item!s}.csv",
        )


# %%
if __name__ == "__main__":
    main()

# %%
