"""行为先验模型（毕设 §3.1）：分层马尔可夫 + 距离衰减 + 异质混合.

三个组件（全部只在拟合份 84 条路线上估计，见 output/test_split_power.json）：
1. RegionMap：90 簇质心 Ward 合并到 K=8 区域（前置实验 #2 裁定的粒度）。
2. SecondOrderMarkov：二阶区域转移 P(region_next | region_prev, region_cur)，
   行向 Dirichlet 平滑（前置实验 #3 裁定二阶）。
3. DecayModel：距离衰减 f(d)，条件 logit（候选池离散选择）拟合，
   三种形式选型：exp(-d/ρ) / 幂律 d^(-β) / 混合 w·exp + (1-w)·幂律。
4. HopMixture：转移距离的两分量混合（就近 lognormal + 跨区 log-gamma 重尾），
   EM 估计——给 λ(s) 提供"就近模式后验"特征（双峰检验结论的参数化落地）。

先验得分（下一站候选打分）：
    P_prior(c) ∝ P2(reg(c) | reg(prev), reg(cur)) · f(d(cur, c))
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import logsumexp

K = 8
DEFAULT_ALPHA = 0.5


# ==== 1. 区域划分 ====
class RegionMap:
    def __init__(self, K=K):
        self.K = K

    def fit(self, meta, clusters):
        centroids = np.array([[meta.iloc[cl]["lng"].mean(), meta.iloc[cl]["lat"].mean()]
                              for cl in clusters])
        Z = linkage(centroids, method="ward")
        roc = fcluster(Z, t=self.K, criterion="maxclust") - 1
        self.region_of_cluster_ = roc
        self.centroids_ = centroids
        return self

    def transform(self, cluster_id):
        return np.array([self.region_of_cluster_[c] for c in cluster_id])


# ==== 2. 二阶区域转移 ====
class SecondOrderMarkov:
    def __init__(self, K=K, alpha=DEFAULT_ALPHA):
        self.K, self.alpha = K, alpha

    def fit(self, region_seqs):
        C = np.zeros((self.K, self.K, self.K))
        for s in region_seqs:
            for a, b, c in zip(s, s[1:], s[2:]):
                C[a, b, c] += 1
        self.counts_ = C
        return self

    def probs(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        P = self.counts_ + alpha
        tot = P.sum(axis=-1, keepdims=True)
        return np.where(tot == 0, 1.0 / self.K, P / np.where(tot == 0, 1, tot))

    def entropy(self, prev, cur, alpha=None):
        """当前状态的转移熵（λ(s) 特征）."""
        P = self.probs(alpha)[prev, cur]
        return float(-np.sum(P * np.log(P + 1e-12)))

    def log_score(self, prev, cur, cand_regions):
        return np.log(self.probs()[prev, cur, cand_regions] + 1e-12)

    def heldout_loglik(self, region_seqs, alpha=None):
        P = self.probs(alpha)
        return float(sum(np.log(P[a, b, c] + 1e-12)
                         for s in region_seqs
                         for a, b, c in zip(s, s[1:], s[2:])))


# ==== 3. 距离衰减（条件 logit 选型） ====
class DecayModel:
    """在候选池离散选择任务上 MLE 拟合 f(d)。

    pool: 每个预测点 (候选距离数组, 真实下一站在池中的下标)。
    三种形式：'exp' f=exp(-d/ρ)；'power' f=d^(-β)（d 截断下限 0.3km）；
    'mix' w·exp + (1-w)·幂律。选择按同一拟合数据的 AIC + 持出 log-loss。
    """

    D_MIN = 0.3  # 幂律发散保护

    def __init__(self, form="exp"):
        self.form = form

    def _logf(self, d, params):
        if self.form == "exp":
            rho, = params
            return -d / rho
        if self.form == "power":
            beta, = params
            return -beta * np.log(np.maximum(d, self.D_MIN))
        w, rho, beta = params
        # log(w·exp(-d/ρ) + (1-w)·d^(-β)) 用 logsumexp
        return np.logaddexp(np.log(w + 1e-12) - d / rho,
                            np.log(1 - w + 1e-12)
                            - beta * np.log(np.maximum(d, self.D_MIN)))

    def fit(self, pools, grid=41):
        """pools: [(dist_vec[K], true_idx)]. 网格+局部细化搜索 MLE."""
        best = None
        if self.form == "exp":
            for rho in np.geomspace(0.2, 500, grid):
                ll = self._pool_ll(pools, (rho,))
                if best is None or ll > best[0]:
                    best = (ll, (float(rho),))
        elif self.form == "power":
            for beta in np.linspace(0.05, 4.0, grid):
                ll = self._pool_ll(pools, (beta,))
                if best is None or ll > best[0]:
                    best = (ll, (float(beta),))
        else:  # mix
            for w in np.linspace(0.05, 0.95, 10):
                for rho in np.geomspace(0.3, 100, 15):
                    for beta in np.linspace(0.05, 3.0, 12):
                        params = (float(w), float(rho), float(beta))
                        ll = self._pool_ll(pools, params)
                        if best is None or ll > best[0]:
                            best = (ll, params)
        self.loglik_, self.params_ = best
        self.n_params_ = len(best[1])
        return self

    def _pool_ll(self, pools, params):
        ll = 0.0
        for dists, true_i in pools:
            lf = self._logf(np.asarray(dists, float), params)
            ll += lf[true_i] - logsumexp(lf)
        return ll

    def heldout_logloss(self, pools):
        return -self._pool_ll(pools, self.params_) / len(pools)

    def aic(self):
        return 2 * self.n_params_ - 2 * self.loglik_

    def log_score(self, d):
        return self._logf(np.atleast_1d(np.asarray(d, float)), self.params_)


# ==== 4. 异质距离混合（lognormal + log-gamma） ====
class HopMixture:
    """就近分量 lognormal + 跨区分量 log-gamma（重尾），EM。

    双峰检验结论：两分量 lognormal 未获 LRT 支持（真实尾部更重）→ 跨区
    分量用 log-gamma（形状 a<1 时 log 尺度重尾）。
    """

    def __init__(self, max_iter=500, tol=1e-8, seed=0, n_init=8):
        self.max_iter, self.tol, self.seed = max_iter, tol, seed
        self.n_init = n_init

    def fit(self, hops):
        from scipy.stats import gamma, norm
        lx = np.log(np.asarray(hops, float))
        rng = np.random.RandomState(self.seed)
        best = None
        for _ in range(self.n_init):
            m = np.quantile(lx, rng.uniform(0.2, 0.45))
            a = rng.uniform(1.5, 2.5)   # log-gamma 位置参数（跨区分量中位）
            s = lx.std() / 3 + 0.1
            w = 0.5
            for _ in range(self.max_iter):
                # E 步
                lp_near = np.log(w + 1e-12) + norm.logpdf(lx, m, s)
                lp_far = np.log(1 - w + 1e-12) + self._log_loggamma(lx, a)
                r = 1 / (1 + np.exp(lp_far - lp_near))
                # M 步
                w = float(np.clip(r.mean(), 1e-3, 1 - 1e-3))
                m = float((r * lx).sum() / r.sum())
                s = float(np.sqrt((r * (lx - m) ** 2).sum() / r.sum()) + 1e-6)
                a = self._fit_a(lx, r)
                ll = float(np.sum(np.logaddexp(
                    np.log(w) + norm.logpdf(lx, m, s),
                    np.log(1 - w) + self._log_loggamma(lx, a))))
                if best is None or ll > best[0]:
                    pass
                if abs(ll - getattr(self, "_prev_ll", ll)) < self.tol:
                    break
                self._prev_ll = ll
            if best is None or ll > best[0]:
                best = (ll, w, m, s, a)
        self.loglik_, self.w_, self.mu_, self.sigma_, self.a_ = best
        return self

    @staticmethod
    def _log_loggamma(x, a):
        """log X 的密度，X~Gamma(1,1)（即 log-gamma 形状1）平移 a：x = a + G."""
        from scipy.stats import gamma as _g
        return _g.logpdf(np.exp(x - a), 1.0, scale=1.0) + (x - a)

    def _fit_a(self, lx, r):
        """M 步的 a：跨区责任加权下 golden-section 最大化."""
        from scipy.optimize import minimize_scalar
        obj = lambda a: -float(np.sum((1 - r) * self._log_loggamma(lx, a)))
        res = minimize_scalar(obj, bounds=(np.quantile(lx, 0.3),
                                           np.quantile(lx, 0.9)),
                              method="bounded")
        return float(res.x)

    def posterior_near(self, d):
        """P(就近分量 | 距离 d)——λ(s) 的模式后验特征."""
        from scipy.stats import norm
        d = np.atleast_1d(np.asarray(d, float))
        lx = np.log(np.maximum(d, 1e-3))
        lp_near = np.log(self.w_) + norm.logpdf(lx, self.mu_, self.sigma_)
        lp_far = np.log(1 - self.w_) + self._log_loggamma(lx, self.a_)
        return 1 / (1 + np.exp(lp_far - lp_near))


def region_sequences(routes, indices, region_of_poi):
    out = []
    for i in indices:
        s = [int(region_of_poi[int(x)]) for x in routes[i]]
        if len(s) >= 2:
            out.append(s)
    return out


def build_prior(routes, fit_indices, meta=None, clusters=None, cluster_id=None,
                alpha=DEFAULT_ALPHA):
    """一步构建完整先验模型（区域划分在 POI 全集上、统计量只在拟合份上估计）."""
    if meta is None:
        meta = pd.read_csv("data/processed/poi_metadata.csv")
    if clusters is None:
        clusters = np.load("data/processed/clusters.npy", allow_pickle=True)
    if cluster_id is None:
        cluster_id = np.load("data/processed/cluster_id.npy")
    rmap = RegionMap().fit(meta, clusters)
    region_of_poi = rmap.transform(cluster_id)
    seqs = region_sequences(routes, fit_indices, region_of_poi)
    mk = SecondOrderMarkov(alpha=alpha).fit(seqs)
    return rmap, region_of_poi, mk
