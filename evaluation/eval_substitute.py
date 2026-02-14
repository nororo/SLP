# %%
"""This scripts run in ipython notebook."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

RESULT_DIR = Path("../results/dunn/pretrained_model")

filename = "../data/tem_tbl_g.csv"
item_tbl = pd.read_csv(filename).iloc[
    :857,
    :857,
]  # 857 is the max item_id in training data
item_tbl
print("item_tbl size:", len(item_tbl))

EVAL_DATA_DIR = Path("../results/eval_did")
from lib.dunnhumby_dataset import get_target_item, get_transaction

# %%
from sklearn.metrics import ndcg_score

EVAL_DATA_DIR = Path("../results/eval_did")
# exp0527


def get_inout_corss_product_matrix(embeddings_dict):
    w_input = embeddings_dict["input_emb"]
    w_output = embeddings_dict["output_emb"]
    p_k_A = expit(np.dot(w_input, w_output.T))
    comp = 0.5 * (p_k_A + p_k_A.T)
    return comp


def get_kld_matrix(embeddings_dict, epsilon=1e-8):
    w_input = embeddings_dict["input_emb"]
    w_output = embeddings_dict["output_emb"]
    p_k_A = softmax(np.dot(w_input, w_output.T), axis=1)
    kld = 0.5 * (
        p_k_A * (np.log(p_k_A + epsilon) - np.log(p_k_A.T + epsilon))
        + p_k_A.T * (np.log(p_k_A.T + epsilon) - np.log(p_k_A + epsilon))
    )
    return kld


def get_input_correlation(embeddings_dict):
    w_input = embeddings_dict["input_emb"]
    corr = np.corrcoef(w_input)
    return corr


def regression_out(mat_comp: pd.DataFrame, mat_subst: pd.DataFrame) -> pd.DataFrame:
    """Calculate the residual of the substitution matrix by regression"""
    mat_comp = mat_comp.unstack()
    mat_comp.index.names = ["item_1", "item_2"]
    x = mat_comp.reset_index().query("item_1 != item_2")
    x.columns = ["item_1", "item_2", "score"]
    x["score_centered"] = x["score"] - x["score"].mean()
    mat_subst = mat_subst.unstack()
    mat_subst.index.names = ["item_1", "item_2"]
    y = mat_subst.reset_index().query("item_1 != item_2")
    y[0] = y[0].fillna(0)
    y["score_centered"] = y[0] - y[0].mean()

    model = LinearRegression()
    model.fit(
        x[["score_centered"]],
        y["score_centered"].values,
    )
    residual = (
        y["score_centered"].values
        - model.predict(x[["score_centered"]])
        + x["score"].mean()
    )
    residual_df = pd.DataFrame(
        residual,
        index=y.index,
        columns=["residual"],
    )
    residual_df
    residual_df["item_1"] = y["item_1"]
    residual_df["item_2"] = y["item_2"]
    residual_df = pd.concat(
        [residual_df, mat_subst.reset_index().query("item_1 == item_2")],
        axis=0,
    )
    residual_df["residual"] = residual_df["residual"].fillna(0)
    residual_mat = residual_df.pivot(
        index="item_1",
        columns="item_2",
        values="residual",
    )
    return residual_mat


# %% load
def get_emb_prod2vec(filename: str, item_tbl: pd.DataFrame):
    s_item = len(item_tbl)
    with open(filename, "rb") as f:
        embeddings_dict = pickle.load(f)
    prod2vec_cross_product_matrix = pd.DataFrame(
        get_inout_corss_product_matrix(embeddings_dict)[1 : s_item + 1, 1 : s_item + 1],
        index=item_tbl.item_id,
        columns=item_tbl.item_id,
    )
    prod2vec_kld_matrix = pd.DataFrame(
        get_kld_matrix(embeddings_dict)[1 : s_item + 1, 1 : s_item + 1],
        index=item_tbl.item_id,
        columns=item_tbl.item_id,
    )

    assert prod2vec_cross_product_matrix.shape == (s_item, s_item)
    assert prod2vec_kld_matrix.shape == (s_item, s_item)

    return prod2vec_cross_product_matrix, prod2vec_kld_matrix


# %%


class EvalEmb:
    def __init__(
        self,
        cls_thr_max: int = 8192 * 2,
        cls_thr_min: int = 8192 * 1,
        thr_vol: int = 5,
        coupon_cnt_thr: int = 100,
        eval_data="did",
    ):
        # load
        df_transaction_data = get_transaction()
        target_exchangeable_item = get_target_item(
            df_transaction_data.query("tran_date>=366"),
            coupon_cnt_thr=coupon_cnt_thr,
        )
        self.n_item_eval = len(target_exchangeable_item)
        df_transaction_data = get_transaction()
        item_purchase_count = (
            df_transaction_data.query("tran_date>=366")
            .groupby("item_id")
            .agg(
                {"basket_id": pd.Series.nunique},
            )
        )

        eval_data_dict = {}
        for itr in range(self.n_item_eval):
            item_id = target_exchangeable_item[itr]

            filename = EVAL_DATA_DIR / f"did_7days_{item_id!s}.csv"
            eval_data = pd.read_csv(filename, index_col=0)

            eval_data = eval_data.merge(
                item_purchase_count,
                left_index=True,
                right_index=True,
                how="left",
            )
            eval_data.rename(columns={"basket_id": "item_purchase_count"}, inplace=True)

            eval_data_dict[item_id] = eval_data
        self.eval_data_dict = eval_data_dict

    def eval_vec_mrr(
        self,
        emb,
        top_k_ans: int = 999999,
        pred_k: int = 100,
        thr: float = 0,
    ):
        """Calculate Mean Reciprocal Rank (MRR)."""
        eval_score_dict = {}
        cnt_rep_item_dict = {}
        cnt_neg_dict = {}
        eval_item_ids = list(set(emb.columns) & set(self.eval_data_dict.keys()))
        for item_id in eval_item_ids:
            eval_data = self.eval_data_dict[item_id]
            eval_vector = emb[[item_id]]
            prod_repr = set(eval_vector.index) & set(eval_data.index)
            cnt_rep_item_dict[item_id] = len(prod_repr)
            if item_id not in prod_repr:
                continue

            eval_data_t = pd.merge(
                eval_data.query("index in @prod_repr"),
                eval_vector.query("index in @prod_repr"),
                left_index=True,
                right_index=True,
                how="left",
            )
            eval_data_t["rank_eval"] = eval_data_t[item_id]

            eval_column = "pred"
            eval_data_t_1 = eval_data_t.assign(
                rank_pred=eval_data_t[item_id].rank(ascending=False).astype(int),
                relevancy=(eval_data_t[eval_column] * (-1)).clip(0),
                relevancy_rank=(eval_data_t[eval_column] * (-1))
                .rank(ascending=False)
                .astype(int),
            )

            relevant_items = eval_data_t_1.query(
                "relevancy_rank <= @top_k_ans and relevancy > @thr",
            )

            if len(relevant_items) == 0:
                eval_score_dict[item_id] = 0.0
            cnt_neg_dict[item_id] = (eval_data_t_1.relevancy > 0).sum()
            sorted_by_pred = eval_data_t_1.sort_values("rank_pred")

            reciprocal_ranks = []
            for idx in relevant_items.index:
                pred_rank = sorted_by_pred.index.get_loc(idx) + 1  # 1-indexed

                if pred_rank <= pred_k:
                    reciprocal_ranks.append(1.0 / pred_rank)
                else:
                    reciprocal_ranks.append(0.0)

            mrr = np.max(reciprocal_ranks) if reciprocal_ranks else 0.0
            eval_score_dict[item_id] = mrr
        return eval_score_dict, cnt_neg_dict, cnt_rep_item_dict

    def eval_vec_map(
        self,
        emb,
        top_k_ans: int = 999999,
        pred_k: int = 100,
        thr: float = 0,
    ):
        """Calculate Mean Average Precision (MAP)."""
        eval_score_dict = {}
        cnt_neg_dict = {}
        cnt_rep_item_dict = {}
        eval_item_ids = list(set(emb.columns) & set(self.eval_data_dict.keys()))
        for item_id in eval_item_ids:
            eval_data = self.eval_data_dict[item_id]
            eval_vector = emb[[item_id]]
            prod_repr = set(eval_vector.index) & set(eval_data.index)
            cnt_rep_item_dict[item_id] = len(prod_repr)
            if item_id not in prod_repr:
                continue
            eval_data_t = pd.merge(
                eval_data.query("index in @prod_repr"),
                eval_vector.query("index in @prod_repr"),
                left_index=True,
                right_index=True,
                how="left",
            )
            eval_data_t["rank_eval"] = eval_data_t[item_id]

            eval_column = "pred"
            eval_data_t_1 = eval_data_t.assign(
                rank_pred=eval_data_t[item_id].rank(ascending=False).astype(int),
                relevancy=(eval_data_t[eval_column] * (-1)).clip(0),
                relevancy_rank=(eval_data_t[eval_column] * (-1))
                .rank(ascending=False)
                .astype(int),
            )
            # get relevant items (relevancy_rank is within top_k_ans)
            relevant_items = set(
                eval_data_t_1.query(
                    "relevancy_rank <= @top_k_ans and relevancy > @thr",
                ).index,
            )
            if len(relevant_items) == 0:
                eval_score_dict[item_id] = 0.0

            # sort by prediction rank and consider only within pred_k
            sorted_by_pred = eval_data_t_1.sort_values("rank_pred").head(pred_k)

            num_hits = 0
            sum_precisions = 0.0

            for rank, (idx, _) in enumerate(sorted_by_pred.iterrows(), start=1):
                if idx in relevant_items:
                    num_hits += 1
                    precision_at_rank = num_hits / rank
                    sum_precisions += precision_at_rank

            average_precision = (
                sum_precisions / len(relevant_items) if len(relevant_items) > 0 else 0.0
            )
            eval_score_dict[item_id] = average_precision

        return eval_score_dict, cnt_neg_dict, cnt_rep_item_dict

    def eval_vec_mean_matching_accuracy(
        self,
        emb,
        top_k_ans: int = 200,
        pred_k: int = 200,
        thr: float = 0,
    ) -> float:
        """VMPPO: Calculate pairwise matching accuracy based on covariate adjustment."""
        eval_score_dict = {}
        cnt_neg_dict = {}
        cnt_rep_item_dict = {}
        eval_item_ids = list(set(emb.columns) & set(self.eval_data_dict.keys()))
        for item_id in eval_item_ids:
            eval_data = self.eval_data_dict[item_id]
            eval_vector = emb[[item_id]]
            prod_repr = set(eval_vector.index) & set(eval_data.index)
            cnt_rep_item_dict[item_id] = len(prod_repr)
            if item_id not in prod_repr:
                continue
            eval_data_t = pd.merge(
                eval_data.query("index in @prod_repr"),
                eval_vector.query("index in @prod_repr"),
                left_index=True,
                right_index=True,
                how="left",
            )
            eval_data_t["rank_eval"] = eval_data_t[item_id]

            eval_column = "pred"
            eval_data_t_1 = eval_data_t.assign(
                rank_pred=eval_data_t[item_id].rank(ascending=False).astype(int),
                relevancy=(eval_data_t[eval_column] * (-1)).clip(0),
                relevancy_rank=(eval_data_t[eval_column] * (-1))
                .rank(ascending=False)
                .astype(int),
            )
            min_score = max(
                0,
                eval_data_t_1.query(
                    "relevancy_rank <= @top_k_ans",
                ).relevancy.min(),
                0,
            )
            eval_data_sorted = eval_data_t.sort_values(
                "item_purchase_count",
            ).reset_index(drop=False)

            correct_predictions = 0
            total_pairs = 0

            for i in range(len(eval_data_sorted) - 1):
                item_i = eval_data_sorted.iloc[i]
                item_j = eval_data_sorted.iloc[i + 1]

                beta_i = max(min_score, -item_i[eval_column])
                beta_j = max(min_score, -item_j[eval_column])

                emb_score_i = item_i[item_id]
                emb_score_j = item_j[item_id]

                if beta_i == beta_j:
                    continue

                beta_larger_is_i = beta_i > beta_j

                emb_predicts_i = emb_score_i > emb_score_j

                if beta_larger_is_i == emb_predicts_i:
                    correct_predictions += 1

                total_pairs += 1

            accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0.0
            eval_score_dict[item_id] = accuracy
        return eval_score_dict, cnt_neg_dict, cnt_rep_item_dict

    def eval_vec_ndcg(
        self,
        emb,
        pred_k: int = 100,
        gain_type: str = "raw",
        top_k_ans: int = 2000,
    ):
        eval_score_dict = {}
        cnt_neg_dict = {}
        eval_item_ids = list(set(emb.columns) & set(self.eval_data_dict.keys()))
        for item_id in eval_item_ids:
            eval_data = self.eval_data_dict[item_id]
            eval_vector = emb[[item_id]]
            prod_repr = set(eval_vector.index) & set(eval_data.index)
            if item_id not in prod_repr:
                continue

            eval_data_t = pd.merge(
                eval_data.query("index in @prod_repr"),
                eval_vector.query("index in @prod_repr"),
                left_index=True,
                right_index=True,
                how="left",
            )

            eval_data_t["rank_eval"] = eval_data_t[item_id]

            eval_column = "pred"
            eval_data_t_1 = eval_data_t.assign(
                rank_pred=eval_data_t[item_id].rank(ascending=False).astype(int),
                relevancy=(eval_data_t[eval_column] * (-1)).clip(0),
                relevancy_rank=(eval_data_t[eval_column] * (-1))
                .rank(ascending=False)
                .astype(int),
            )
            eval_data_t_1 = eval_data_t_1.query(
                "relevancy_rank <= @top_k_ans",
            )
            # eval_score = {}

            # eval_score['correlation_vitamin_drink_new'], pvalue = spearmanr(eval_data_t_1['relevancy_rank'].values, eval_data_t_1['rank'].values)
            if gain_type == "rank":
                # NDCGには予測スコア（高いほど良い）を渡す必要があるため、ランクを逆転させる
                max_rank = eval_data_t_1["rank_pred"].max()
                pred_score = max_rank + 1 - eval_data_t_1["rank_pred"]
                ans_max_rank = eval_data_t_1["relevancy_rank"].max()
                ans_pred_score = ans_max_rank + 1 - eval_data_t_1["relevancy_rank"]

                eval_score = ndcg_score(
                    ans_pred_score.values.reshape(1, -1),
                    pred_score.values.reshape(1, -1),
                    k=pred_k,
                )
                eval_score_dict[item_id] = eval_score

            elif gain_type == "raw":
                # NDCGには予測スコア（高いほど良い）を渡す必要があるため、ランクを逆転させる
                max_rank = eval_data_t_1["rank_pred"].max()
                pred_score = max_rank + 1 - eval_data_t_1["rank_pred"]
                eval_score_le0 = ndcg_score(
                    eval_data_t_1["relevancy"].values.reshape(1, -1),
                    pred_score.values.reshape(1, -1),
                    k=pred_k,
                )
                eval_score_dict[item_id] = eval_score_le0
            cnt_neg_dict[item_id] = (eval_data_t_1.relevancy > 0).sum()

        return eval_score_dict, cnt_neg_dict


EvalEmb_obj = EvalEmb(coupon_cnt_thr=20)


def eval_embedding(
    EvalEmb_obj,
    rst_dict,
    pred_k: int = 50,
    thr: float = 0,
    gain_type="raw",
    top_k_ans=2000,
    random_rst_df=None,
):
    rst_df_all = []
    _, cnt_neg_list = EvalEmb_obj.eval_vec_ndcg(
        rst_dict["prod2vec_user_subst"],
        pred_k=pred_k,
        gain_type=gain_type,
        top_k_ans=top_k_ans,
    )
    rst_df_all.append(
        pd.Series(
            cnt_neg_list,
            name="cnt_neg",
        ),
    )
    for key, value in rst_dict.items():
        print(key)
        eval_score_dict, cnt_neg_list = EvalEmb_obj.eval_vec_ndcg(
            value,
            pred_k=pred_k,
            gain_type=gain_type,
            top_k_ans=top_k_ans,
        )
        rst_df_all.append(
            pd.Series(
                eval_score_dict,
                name=key,
            ),
        )
    rst_df = pd.concat(rst_df_all, axis=1)
    rst_df = rst_df.T
    rst_df["mean"] = rst_df.mean(
        axis=1,
    )
    return rst_df


# %%
def get_random_results(
    method="ndcg",
    rnd_itr_num=10,
    top_k_ans=2000,
    pred_k=4000,
    coupon_cnt_thr=80,
):
    np.random.seed(42)
    random_rst_df_list = []
    EvalEmb_obj = EvalEmb(
        coupon_cnt_thr=coupon_cnt_thr,
    )
    for itr in tqdm(range(rnd_itr_num)):
        random_subst = pd.DataFrame(
            np.random.randn(len(item_tbl), len(item_tbl)),
            index=item_tbl.item_id,
            columns=item_tbl.item_id,
        )
        if method == "ndcg":
            random_rst_df, _ = EvalEmb_obj.eval_vec_ndcg(
                random_subst,
                pred_k=pred_k,
                gain_type="raw",
                top_k_ans=top_k_ans,
            )
            random_rst_df_list.append(random_rst_df)
        elif method == "mrr":
            random_rst_df, _, _ = EvalEmb_obj.eval_vec_mrr(
                random_subst,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            random_rst_df_list.append(random_rst_df)
        elif method == "map":
            random_rst_df, _, _ = EvalEmb_obj.eval_vec_map(
                random_subst,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            random_rst_df_list.append(random_rst_df)
        elif method == "mean_matching_accuracy":
            random_rst_df, _, _ = EvalEmb_obj.eval_vec_mean_matching_accuracy(
                random_subst,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            random_rst_df_list.append(random_rst_df)

    random_rst_df = pd.DataFrame(random_rst_df_list)
    random_rst_df["average"] = random_rst_df.mean(
        axis=1,
    )  # macro average of rates (simple average of the rates)
    return random_rst_df


def main():

    rst_dict = {}

    filename = RESULT_DIR / "dunn_prod2vec_user_output/embeddings_dict.pickle.gz"
    rst_dict["prod2vec_user_subst"], _ = get_emb_prod2vec(filename, item_tbl)
    filename = RESULT_DIR / "dunn_prod2vec_basket_output/embeddings_dict.pickle.gz"

    (
        rst_dict["prod2vec_bsk_comp"],
        rst_dict["prod2vec_bsk_subst"],
    ) = get_emb_prod2vec(filename=filename, item_tbl=item_tbl)

    rst_dict["prod2vec_bsk_residual"] = regression_out(
        rst_dict["prod2vec_bsk_comp"],
        rst_dict["prod2vec_bsk_subst"],
    )

    for k in [0]:
        filename = RESULT_DIR / f"dunn_subst2vec_k{k}_output/embeddings_dict.pickle.gz"
        rst_dict[f"subst2vec_subst_k{k}"], _ = get_emb_prod2vec(
            filename=filename,
            item_tbl=item_tbl,
        )

    for k in [0]:
        filename = (
            RESULT_DIR
            / f"dunn_subst2vec_visit2vec_k{k}_output/embeddings_dict.pickle.gz"
        )

        rst_dict[f"subst2vec_visit2vec_k{k}"], _ = get_emb_prod2vec(
            filename=filename,
            item_tbl=item_tbl,
        )

    RESULT_EXPORT_DIR = Path("../results/dunn/results_tbl")
    top_k_ans = 2000
    pred_k = 4000
    cnt_ans_thr = 20
    top_k_ans = 300
    for coupon_cnt_thr in [40, 80]:
        EvalEmb_obj = EvalEmb(
            coupon_cnt_thr=coupon_cnt_thr,
        )
        random_rst_df = get_random_results(
            method="ndcg",
            rnd_itr_num=100,
            top_k_ans=top_k_ans,
            pred_k=pred_k,
            coupon_cnt_thr=coupon_cnt_thr,
        )

        rst_df = eval_embedding(
            EvalEmb_obj,
            rst_dict,
            pred_k=pred_k,
            gain_type="raw",
            random_rst_df=random_rst_df,
            top_k_ans=top_k_ans,
        )

        selected_cols_fig_1 = [
            "cnt_neg",
            "popularity",
            "prod2vec_user_subst",
            "prod2vec_bsk_comp",
            # "subst2vec_subst_k0",
            "subst2vec_visit2vec_k0",
            # "subst2vec_visit2vec_k1",
            "subst2vec_subst_k0",
            # "subst2vec_subst_k5",
            # "subst2vec_subst_k0_opt",
            "random_baseline",
            "random_baseline+std",
            "random_baseline-std",
            "random_std",
        ]
        mask_columns = list(
            set(rst_df.T.query("cnt_neg > @cnt_ans_thr").index) | set(["mean"]),
        )

        rst_df.loc["random_baseline", :] = random_rst_df["average"].mean()
        rst_df.loc["random_baseline+std", :] = (
            random_rst_df["average"].mean() + random_rst_df["average"].std()
        )
        rst_df.loc["random_baseline-std", :] = (
            random_rst_df["average"].mean() - random_rst_df["average"].std()
        )
        rst_df.loc["random_std", :] = random_rst_df["average"].std()
        print(mask_columns)
        print("======== fig 1 ========")
        rst_df.to_csv(RESULT_EXPORT_DIR / f"coupon_cnt_thr{coupon_cnt_thr}_ndcg.csv")
        display(
            rst_df.loc[selected_cols_fig_1, mask_columns].sort_values(
                by="mean",
                ascending=False,
            ),
        )
        selected_cols_fig_2 = [
            "prod2vec_user_subst",
            "prod2vec_bsk_residual",
            "subst2vec_visit2vec_k0",
            "subst2vec_subst_k0",
            "shopper_subst",
            "shopper_residual",
            "decgcn_subst",
            "pmsc_subst",
            "spem_subst",
        ]
        display(
            rst_df.loc[selected_cols_fig_2, mask_columns].sort_values(
                by="mean",
                ascending=False,
            ),
        )
    # %% mrr

    def eval_embedding_mrr(
        EvalEmb_obj,
        rst_dict,
        pred_k: int = 50,
        top_k_ans: int = 100,
        thr: float = 0,
    ):
        rst_df_all = []
        _, cnt_neg_list, cnt_rep_item_dict = EvalEmb_obj.eval_vec_mrr(
            rst_dict["prod2vec_user_subst"],
            pred_k=pred_k,
            top_k_ans=top_k_ans,
        )
        rst_df_all.append(
            pd.Series(
                cnt_neg_list,
                name="cnt_neg",
            ),
        )
        for key, value in rst_dict.items():
            eval_score_dict, _, _ = EvalEmb_obj.eval_vec_mrr(
                value,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            rst_df_all.append(
                pd.Series(
                    eval_score_dict,
                    name=key,
                ),
            )
        rst_df = pd.concat(rst_df_all, axis=1)
        rst_df = rst_df.T
        rst_df["mean"] = rst_df.mean(
            axis=1,
        )
        return rst_df, cnt_rep_item_dict

    top_k_ans = 2000
    pred_k = 4000
    cnt_ans_thr = 20
    for coupon_cnt_thr in [40, 80]:
        EvalEmb_obj = EvalEmb(
            coupon_cnt_thr=coupon_cnt_thr,
        )
        for top_k_ans in [1, 2, 4, 8]:
            rst_df, _ = eval_embedding_mrr(
                EvalEmb_obj=EvalEmb_obj,
                rst_dict=rst_dict,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            random_rst_df = get_random_results(
                method="mrr",
                rnd_itr_num=100,
                top_k_ans=top_k_ans,
                pred_k=pred_k,
                coupon_cnt_thr=coupon_cnt_thr,
            )
            mask_columns = list(
                set(rst_df.T.query("cnt_neg > @cnt_ans_thr").index) | set(["mean"]),
            )
            rst_df.loc["random_baseline", :] = random_rst_df["average"].mean()
            rst_df.loc["random_baseline+std", :] = (
                random_rst_df["average"].mean() + random_rst_df["average"].std()
            )
            rst_df.loc["random_baseline-std", :] = (
                random_rst_df["average"].mean() - random_rst_df["average"].std()
            )
            rst_df.loc["random_std", :] = random_rst_df["average"].std()
            rst_df.to_csv(
                RESULT_EXPORT_DIR
                / f"coupon_cnt_thr{coupon_cnt_thr}_mrr_top_k_ans{top_k_ans}.csv",
            )
            print(mask_columns)
            print("======== fig1 map ========")
            display(
                rst_df.loc[selected_cols_fig_1, mask_columns].sort_values(
                    by="mean",
                    ascending=False,
                ),
            )
            print("======== fig2 MAP ========")
            display(
                rst_df.loc[selected_cols_fig_2, mask_columns].sort_values(
                    by="mean",
                    ascending=False,
                ),
            )

    # %% map

    def eval_embedding_map(
        EvalEmb_obj,
        rst_dict,
        pred_k: int = 50,
        top_k_ans: int = 100,
        thr: float = 0,
    ):
        rst_df_all = []
        _, cnt_neg_list, cnt_rep_item_dict = EvalEmb_obj.eval_vec_map(
            rst_dict["prod2vec_user_subst"],
            pred_k=pred_k,
            top_k_ans=top_k_ans,
        )
        rst_df_all.append(
            pd.Series(
                cnt_neg_list,
                name="cnt_neg",
            ),
        )
        rst_df_all.append(
            pd.Series(
                cnt_rep_item_dict,
                name="cnt_rep_item",
            ),
        )
        for key, value in rst_dict.items():
            eval_score_dict, _, _ = EvalEmb_obj.eval_vec_map(
                value,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            rst_df_all.append(
                pd.Series(
                    eval_score_dict,
                    name=key,
                ),
            )
        rst_df = pd.concat(rst_df_all, axis=1)
        rst_df = rst_df.T
        rst_df["mean"] = rst_df.mean(
            axis=1,
        )
        return rst_df

    pred_k = 4000
    cnt_ans_thr = 20
    for coupon_cnt_thr in [20, 40]:
        EvalEmb_obj = EvalEmb(
            coupon_cnt_thr=coupon_cnt_thr,
        )

        for top_k_ans in [8, 16, 32, 64]:
            rst_df = eval_embedding_map(
                EvalEmb_obj,
                rst_dict,
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
            random_rst_df = get_random_results(
                method="map",
                rnd_itr_num=100,
                top_k_ans=top_k_ans,
                pred_k=pred_k,
                coupon_cnt_thr=coupon_cnt_thr,
            )

            mask_columns = list(
                set(rst_df.T.query("cnt_neg > @cnt_ans_thr").index) | set(["mean"]),
            )
            rst_df.loc["random_baseline", :] = random_rst_df["average"].mean()
            rst_df.loc["random_baseline+std", :] = (
                random_rst_df["average"].mean() + random_rst_df["average"].std()
            )
            rst_df.loc["random_baseline-std", :] = (
                random_rst_df["average"].mean() - random_rst_df["average"].std()
            )
            rst_df.loc["random_std", :] = random_rst_df["average"].std()
            rst_df.to_csv(
                RESULT_EXPORT_DIR
                / f"coupon_cnt_thr{coupon_cnt_thr}_map_top_k_ans{top_k_ans}.csv",
            )
            print(mask_columns)
            print("======== fig1 map ========")
            display(
                rst_df.loc[selected_cols_fig_1, mask_columns].sort_values(
                    by="mean",
                    ascending=False,
                ),
            )
            print("======== fig2 MAP ========")
            display(
                rst_df.loc[selected_cols_fig_2, mask_columns].sort_values(
                    by="mean",
                    ascending=False,
                ),
            )

    # %% mean matching accuracy
    def eval_embedding_mean_matching_accuracy(
        EvalEmb_obj,
        rst_dict,
        pred_k: int = 50,
        top_k_ans: int = 100,
        thr: float = 0,
    ):
        rst_df_all = []
        _, cnt_neg_list, cnt_rep_item_dict = (
            EvalEmb_obj.eval_vec_mean_matching_accuracy(
                rst_dict["prod2vec_user_subst"],
                pred_k=pred_k,
                top_k_ans=top_k_ans,
            )
        )
        rst_df_all.append(
            pd.Series(
                cnt_neg_list,
                name="cnt_neg",
            ),
        )
        for key, value in rst_dict.items():
            eval_score_dict, cnt_neg_list, cnt_rep_item_dict = (
                EvalEmb_obj.eval_vec_mean_matching_accuracy(
                    value,
                    pred_k=pred_k,
                    top_k_ans=top_k_ans,
                )
            )
            rst_df_all.append(
                pd.Series(
                    eval_score_dict,
                    name=key,
                ),
            )
        rst_df = pd.concat(rst_df_all, axis=1)
        rst_df = rst_df.T
        rst_df["mean"] = rst_df.mean(
            axis=1,
        )
        return rst_df

    top_k_ans = 300
    pred_k = 1000
    cnt_ans_thr = 20
    for coupon_cnt_thr in [40, 80]:
        EvalEmb_obj = EvalEmb(
            coupon_cnt_thr=coupon_cnt_thr,
        )
        random_rst_df = get_random_results(
            method="mean_matching_accuracy",
            rnd_itr_num=100,
            top_k_ans=top_k_ans,
            pred_k=pred_k,
            coupon_cnt_thr=coupon_cnt_thr,
        )
        rst_df = eval_embedding_mean_matching_accuracy(
            EvalEmb_obj,
            rst_dict,
            pred_k=pred_k,
            top_k_ans=top_k_ans,
        )
        mask_columns = list(
            set(rst_df.T.query("cnt_neg > @cnt_ans_thr").index) | set(["mean"]),
        )
        rst_df.loc["random_baseline", :] = random_rst_df["average"].mean()
        rst_df.loc["random_baseline+std", :] = (
            random_rst_df["average"].mean() + random_rst_df["average"].std()
        )
        rst_df.loc["random_baseline-std", :] = (
            random_rst_df["average"].mean() - random_rst_df["average"].std()
        )
        rst_df.loc["random_std", :] = random_rst_df["average"].std()
        rst_df.to_csv(
            RESULT_EXPORT_DIR
            / f"coupon_cnt_thr{coupon_cnt_thr}_mean_matching_accuracy.csv",
        )
        print(mask_columns)
        print("======== fig1 mean matching accuracy ========")
        display(
            rst_df.loc[selected_cols_fig_1, mask_columns].sort_values(
                by="mean",
                ascending=False,
            ),
        )
        print("======== fig2 mean matching accuracy ========")
        display(
            rst_df.loc[selected_cols_fig_2, mask_columns].sort_values(
                by="mean",
                ascending=False,
            ),
        )


# %%
