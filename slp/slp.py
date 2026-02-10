# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
np.random.seed(0)

import itertools

import pandera as pa
from pandera.typing import Series
from tqdm import tqdm


def safe_flatten_lists(series):
    """Safely flatten a Series of lists"""
    return list(itertools.chain.from_iterable(series.dropna()))


def get_shopping_embedding(use_cls_embedding: bool = True, normalize: bool = False):
    filename = "features.parquet"

    emb_basket = pd.read_parquet(filename)

    emb_basket = emb_basket.set_index("basket_id").drop(
        columns=["user_id", "num_history_baskets"],
    )
    if use_cls_embedding:
        cls_embedding_cols = ["embedding_" + str(num) for num in range(64)]
        emb_basket = emb_basket[cls_embedding_cols]
    if normalize:
        emb_basket = (emb_basket - emb_basket.mean(axis=1)) / emb_basket.std(axis=1)
    print(emb_basket.shape)
    return emb_basket


class TransactionGroupedByUser(pa.DataFrameModel):
    """Transaction grouped by user"""

    user_id: Series[int]
    item_id_list: Series[list[int]]
    basket_id: Series[int]


class TransactionGroupedByUserWithEmbedding(pa.DataFrameModel):
    """Transaction grouped by user"""

    user_id: Series[int]
    item_id_list: Series[list[int]]
    basket_id: Series[int]
    embedding_0: Series[np.float32]


@pa.check_io(
    baskets_without_query_df=TransactionGroupedByUserWithEmbedding,
    baskets_with_query_look=TransactionGroupedByUserWithEmbedding,
)
def calc_distance(
    baskets_without_query_df: TransactionGroupedByUserWithEmbedding,
    baskets_with_query_look: TransactionGroupedByUserWithEmbedding,
) -> tuple[int | None, float] | None:
    """Calculate distance between baskets without query and baskets with query"""
    embedding_cols = [
        col for col in baskets_with_query_look.columns if col.startswith("embedding_")
    ]
    if len(embedding_cols) > 0:
        emb_basket_with_query = baskets_with_query_look[
            embedding_cols + ["basket_id"]
        ].set_index("basket_id")
        emb_basket_with_query = emb_basket_with_query.mean(axis=0)
    else:
        print(
            "DEBUG: No embedding columns found for basket",
        )
    if len(embedding_cols) > 0:
        emb_baskets_without_query = baskets_without_query_df[
            embedding_cols + ["basket_id"]
        ].set_index("basket_id")
    else:
        print("DEBUG: No embedding columns found for baskets without query")
        return None

    if len(emb_baskets_without_query) > 1:
        query_emb = emb_basket_with_query.values
        baskets_emb = emb_baskets_without_query.values

        if query_emb.ndim == 1:
            query_emb = query_emb[
                np.newaxis,
                :,
            ]

        distances = np.linalg.norm(
            baskets_emb - query_emb,
            axis=1,
        )
        basket_id_nearest = emb_baskets_without_query.index[np.argmin(distances)]
        min_distance = np.min(distances)
        matched_distance = float(min_distance)
        basket_id_nearest = emb_baskets_without_query.index[np.argmin(distances)]

    elif len(emb_baskets_without_query) == 1:
        # 1 item case
        query_emb = emb_basket_with_query.values
        basket_emb = emb_baskets_without_query.values[0]

        if query_emb.ndim == 1:
            distance = np.linalg.norm(basket_emb - query_emb)
        else:
            distance = np.linalg.norm(basket_emb - query_emb[0])

        matched_distance = float(distance)
        basket_id_nearest = emb_baskets_without_query.index[0]

    else:
        basket_id_nearest = None

    return basket_id_nearest, matched_distance


@pa.check_io(df=TransactionGroupedByUser)
def slp_shopping_matching(
    df: TransactionGroupedByUser,
    second_order_prox: bool = False,
) -> list[dict] | None:
    """Substitutes products label predictor using visit2vec
    Suppossed to be used with "group by" user_id
    """
    # mapping of item to baskets
    item_to_baskets = {}  # item_id: str -> basket_id: set[str]
    for basket_id, item_list in zip(df["basket_id"], df["item_id_list"]):
        for item in item_list:
            if item not in item_to_baskets:
                item_to_baskets[item] = set()
            item_to_baskets[item].add(basket_id)

    sr = df.item_id_list  # user
    basket_item_list = list(set(safe_flatten_lists(sr)))

    rcd_list = []

    if len(sr) > 1:
        for query_item in basket_item_list:
            # use mapping of item to baskets
            if query_item in item_to_baskets:
                baskets_with_query_ids: set[int] = item_to_baskets[query_item]
                baskets_with_query = df[df["basket_id"].isin(baskets_with_query_ids)]
            else:
                # fallback: use item_id_list
                mask = df.item_id_list.apply(lambda x: query_item in x)
                baskets_with_query = df.loc[mask, :]

            # basket_id of records_including_query
            assert set(baskets_with_query.basket_id) == baskets_with_query_ids

            # 1st-order items
            set_items_with_query = set(
                safe_flatten_lists(baskets_with_query.item_id_list),
            ) - set(
                [query_item],
            )
            matched_item_list = []
            # ins
            for basket_with_query in baskets_with_query_ids:
                # retrived pool basket
                baskets_with_query_look = df.query("basket_id == @basket_with_query")
                # item_id_list of retrieved pool basket
                item_id_list_with_query = safe_flatten_lists(
                    baskets_with_query_look.item_id_list,
                )

                # retrieve from baskets without query item with nearest embedding
                if second_order_prox:
                    baskets_with_query_1st = []
                    for query_item_1st in item_id_list_with_query:
                        if query_item_1st != query_item:
                            baskets_with_query_1st.extend(
                                item_to_baskets[query_item_1st],
                            )
                    set_baskets_with_query_1st = set(baskets_with_query_1st)

                    baskets_without_query_ids = set(
                        df.query(
                            "basket_id not in @baskets_with_query_ids and basket_id in @set_baskets_with_query_1st",
                        ).basket_id,
                    )
                else:
                    # retrieve from baskets without query item with nearest embedding
                    baskets_without_query_ids = set(
                        df.query(
                            "basket_id not in @baskets_with_query_ids",
                        ).basket_id,
                    )

                # if no baskets without query item, skip
                if len(baskets_without_query_ids) == 0:
                    continue

                # get baskets without query item

                baskets_without_query_df = df.query(
                    "basket_id in @baskets_without_query_ids",
                )
                basket_id_nearest, matched_distance = calc_distance(
                    baskets_without_query_df,
                    baskets_with_query_look,
                )
                if basket_id_nearest is None:
                    continue

                if basket_id_nearest is not None:
                    nearest_baskets = df.query("basket_id == @basket_id_nearest")
                    nearest_basket_items = list(
                        set(safe_flatten_lists(nearest_baskets.item_id_list)),
                    )
                    substitute_set = (
                        set(nearest_basket_items) - set_items_with_query
                    ) - set(
                        [query_item],
                    )
                    for sub in substitute_set:
                        rcd_list.append(
                            {
                                "query": query_item,
                                "substitute": sub,
                                "comp": list(set_items_with_query),
                                "distance": matched_distance,
                            },
                        )

    if rcd_list == []:
        print(
            f"DEBUG: rcd_list is empty for user with {len(sr)} baskets and {len(basket_item_list)} unique items",
        )
        return None
    return rcd_list


# %%


def slp_2nd_order_proximity(
    df: TransactionGroupedByUser,
):
    sr = df.item_id_list
    basket_item_list = list(set(sr.sum()))
    assert len(set(basket_item_list)) == len(set(sr.sum())), "duplicated item"
    subst_list = []
    rcd_list = []
    cop_item_dict = {}
    if len(sr) > 1:
        for query_item in basket_item_list:
            basket_list_including_query = [
                basket_items for basket_items in sr if query_item in basket_items
            ]

            basket_item_set_with_query = set(
                sum(basket_list_including_query, []),
            ) - set([query_item])
            substitute_rep = [
                basket_items
                for basket_items in sr
                if (query_item not in basket_items)
                & (basket_item_set_with_query & set(basket_items) != set())
            ]
            substitute_set = (
                set(sum(substitute_rep, [])) - basket_item_set_with_query
            ) - set([query_item])
            cop_item_dict.update({query_item: list(basket_item_set_with_query)})
            if len(substitute_set) > 0:
                subst_list = subst_list + [
                    (query_item, sub) for sub in list(substitute_set)
                ]

        for item1, item2 in subst_list:
            rcd_list.append(
                {
                    "query": item1,
                    "substitute": item2,
                    "comp": cop_item_dict[item1],
                    "substitute_list": subst_list,
                },
            )
    if rcd_list == []:
        return None
    return rcd_list
