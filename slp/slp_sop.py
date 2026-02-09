import random


def substitutes_products_estimator_2nd_order(
    df,
    negative_sample_table,
    k_cop_neg: int = 2,
    neg_sample_size=10,
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
            # if COUNT_TH > 1:
            #    basket_list_sr = pd.Series(basket_list_including_query).value_counts()
            #    basket_list_sr = basket_list_sr[basket_list_sr >= COUNT_TH]
            #    basket_list_including_query = basket_list_sr.index.tolist()

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
        pos_pair = random.sample(
            subst_list,
            k=min(len(basket_item_list), len(subst_list)),
        )

        for item1, item2 in pos_pair:
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
