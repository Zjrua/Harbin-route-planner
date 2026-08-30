"""架构必要性消融（论证架构简洁性：每个组件是否可省）.

问题：当前管线 = 二阶区域转移 + 混合距离衰减 + log-linear 池化(固定λ)。
哪些组件对命中率有不可替代的贡献？哪些可以砍掉？

消融矩阵（测试份 372 点，全部用拟合份估计的参数）：
1. 距离衰减形式 ∈ {none, exp, power, mix}（region 固定开）
2. 区域转移 ∈ {off, 一阶, 二阶}（decay 固定 mix）
3. 区域数 K ∈ {4, 6, 8}（decay=mix, 二阶）
4. 平滑 α ∈ {0.05, 0.1, 0.5}（其余默认）
5. 融合层：decay ∈ {exp, mix} 对融合(fixed λ=0.55)的影响
   ——若 mix 对融合无增益，管线可简化为 exp 衰减

用法:
    ./.venv/Scripts/python.exe scripts/run_ablation_arch.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_fusion_eval import compute_artifacts, fuse, mcnemar_exact, wilson_ci
from src.behavior_prior import SecondOrderMarkov, region_sequences

OUT_JSON = "output/eval_ablation_arch.json"


def prior_hits(art, mk, decay_form, use_region=True, split="test"):
    """给定转移矩阵与衰减形式，算测试份先验命中率与逐点命中向量."""
    pools = art["pools"][split]
    region_of_poi = art["region_of_poi"]
    from src.behavior_prior import DecayModel
    dist = art["dist"]
    # 拟合份池上重拟合该形式的衰减（数据纪律：只用拟合份）
    fit_pools = [([float(dist[p["prefix"][-1]][c]) for c in p["cands"]],
                  p["cands"].index(p["true_next"])) for p in art["pools"]["fit"]]
    dec = DecayModel(form=decay_form).fit(fit_pools) if decay_form != "none" else None
    hits = []
    for pt in pools:
        last, prev = pt["prefix"][-1], pt["prefix"][-2]
        ds = np.array([float(dist[last][c]) for c in pt["cands"]])
        if use_region:
            cr = np.array([region_of_poi[c] for c in pt["cands"]])
            lp = mk.log_score(region_of_poi[prev], region_of_poi[last], cr)
        else:
            lp = np.zeros(len(ds))
        if dec is not None:
            lp = lp + dec.log_score(ds)
        hits.append(pt["cands"][int(np.argmax(lp))] == pt["true_next"])
    return np.array(hits), dec


def first_order_mk(art):
    """一阶区域转移（同 α），用于阶数消融."""
    seqs = art["fit_seqs"]
    K = 8
    C = np.zeros((K, K))
    for s in seqs:
        for a, b in zip(s, s[1:]):
            C[a, b] += 1
    P = C + art["best_alpha"]
    P = P / P.sum(axis=1, keepdims=True)
    class M1:
        def log_score(self, prev, cur, cand_regions):
            row = P[cur] + 1e-12
            row = row / row.sum()
            return np.log(row[cand_regions])
    return M1()


def summarize(hits):
    k = int(hits.sum())
    lo, hi = wilson_ci(k, len(hits))
    return {"hit": round(k / len(hits), 4), "k": k, "n": len(hits),
            "ci95": [round(lo, 4), round(hi, 4)]}


def main():
    art = compute_artifacts(skip_llm=True)
    out = {}

    base_mk = SecondOrderMarkov(alpha=art["best_alpha"]).fit(art["fit_seqs"])
    base_hits, _ = prior_hits(art, base_mk, "mix")

    # ==== 1. 距离衰减形式 ====
    dec_tbl = {}
    for form in ("none", "exp", "power", "mix"):
        h, dec = prior_hits(art, base_mk, form)
        dec_tbl[form] = summarize(h)
    # region off（纯距离）
    h, _ = prior_hits(art, base_mk, "mix", use_region=False)
    dec_tbl["region_off_pure_dist"] = summarize(h)
    out["decay_form"] = dec_tbl

    # ==== 2. 阶数 ====
    ord_tbl = {}
    m1 = first_order_mk(art)
    h1, _ = prior_hits(art, m1, "mix")
    ord_tbl["first_order"] = summarize(h1)
    ord_tbl["second_order"] = summarize(base_hits)
    b = int(np.sum(base_hits & ~h1)); c = int(np.sum(~base_hits & h1))
    ord_tbl["second_vs_first_mcnemar"] = {"b": b, "c": c,
                                          "p": round(mcnemar_exact(b, c), 5)}
    out["order"] = ord_tbl

    # ==== 3. K（需要重划分区域） ====
    import pandas as pd
    from src.behavior_prior import RegionMap
    meta = pd.read_csv("data/processed/poi_metadata.csv")
    clusters = np.load("data/processed/clusters.npy", allow_pickle=True)
    cluster_id = np.load("data/processed/cluster_id.npy")
    routes = art["routes"]
    splits = art["splits"]
    k_tbl = {}
    for K in (4, 6, 8):
        if K == 8:
            mk = base_mk
            h = base_hits
        else:
            rmap = RegionMap(K=K).fit(meta, clusters)
            rop = rmap.transform(cluster_id)
            seqs = region_sequences(routes, splits["fit"], rop)
            mk = SecondOrderMarkov(K=K, alpha=art["best_alpha"]).fit(seqs)
            h = []
            for pt in art["pools"]["test"]:
                last, prev = pt["prefix"][-1], pt["prefix"][-2]
                ds = np.array([float(art["dist"][last][c]) for c in pt["cands"]])
                cr = np.array([rop[c] for c in pt["cands"]])
                lp = mk.log_score(rop[prev], rop[last], cr) \
                    + art["decay"].log_score(ds)
                h.append(pt["cands"][int(np.argmax(lp))] == pt["true_next"])
            h = np.array(h)
        k_tbl[K] = summarize(h)
    out["K"] = k_tbl

    # ==== 4. α ====
    a_tbl = {}
    for a in (0.05, 0.1, 0.5):
        mk = SecondOrderMarkov(alpha=a).fit(art["fit_seqs"])
        h, _ = prior_hits(art, mk, "mix")
        a_tbl[a] = summarize(h)
    out["alpha"] = a_tbl

    # ==== 5. 融合层的衰减形式（fixed λ=0.55） ====
    from run_fusion_eval import score_split
    test_res, test_lp, test_X, test_y = score_split(art, "test")
    test_ll = np.log(np.clip(art["llm_probs"]["test"], 1e-12, 1))
    fus_tbl = {}
    for form in ("exp", "mix"):
        lp_decay_only = []
        # 用该形式衰减重建 log_prior（区域部分不变：从 test_lp 中减 mix 加新形式）
        # 简化：直接重算先验
        mk = base_mk
        h, dec = prior_hits(art, mk, form)
        # 重算逐点 log_prior
        lps = []
        for pt in art["pools"]["test"]:
            last, prev = pt["prefix"][-1], pt["prefix"][-2]
            ds = np.array([float(art["dist"][last][c]) for c in pt["cands"]])
            cr = np.array([art["region_of_poi"][c] for c in pt["cands"]])
            lp = mk.log_score(art["region_of_poi"][prev],
                              art["region_of_poi"][last], cr) + dec.log_score(ds)
            lp = lp - lp.max()
            lps.append(lp)
        lps = np.array(lps)
        P = fuse(lps, test_ll, 0.55)
        hits = P.argmax(axis=1) == test_y
        fus_tbl[form] = summarize(hits)
    out["fusion_decay"] = fus_tbl

    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"保存至: {OUT_JSON}")


if __name__ == "__main__":
    main()
