"""Dense / hybrid 相关性排序器（bge-m3），注册进 relevance 的 ranker 谱系。

高维度统一：这不是"账本外挂向量库"，而是 read_relevant 的一种【相关性度量】——读的仍是同一个富
S(t)（execute+GOVERN 后的可采纳命题的语言体），只是把"词法重叠"换成"dense 语义相似度"。召回恒在
可采纳集内发生（定理2级合规不变）。mem0 的 dense 检索无治理门；SRG 的 dense 检索天然合规——这是同样
dense 召回下 SRG 胜出的点。

重依赖（sentence-transformers + bge-m3 权重）隔离在本模块，import 失败则静默不注册，core 仍可零依赖跑。
embedding 带缓存（CPU 编码慢，同一 body 只编一次）。hybrid = RRF(lexical, dense)。
"""

from __future__ import annotations

import os

from trace.core.relevance import register_ranker, tokenize, lexical_score

_MODEL = None
_EMB_CACHE = {}   # body 文本 -> 向量（避免重复编码）


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        path = os.getenv("TRACE_EMBED_MODEL",
                         os.getenv("SRG_EMBED_MODEL", "BAAI/bge-m3"))
        device = os.getenv("TRACE_EMBED_DEVICE",
                           os.getenv("SRG_EMBED_DEVICE", "cpu"))
        _MODEL = SentenceTransformer(path, device=device)
    return _MODEL


def _encode(texts: list[str]):
    """批量编码 + 缓存。返回与 texts 对齐的向量列表。"""
    import numpy as np
    todo = [t for t in texts if t not in _EMB_CACHE]
    if todo:
        uniq = list(dict.fromkeys(todo))
        vecs = _model().encode(uniq, normalize_embeddings=True, batch_size=32)
        for t, v in zip(uniq, vecs):
            _EMB_CACHE[t] = np.asarray(v, dtype="float32")
    return [_EMB_CACHE[t] for t in texts]


def _dense_rank(state, query: str, k: int) -> list:
    """dense 语义排序：query 与每命题 body 的 cosine（已归一化→点积）。"""
    import numpy as np
    items = [(pp, getattr(pp, "body", getattr(pp, "value", "")) or "") for pp in state]
    items = [(pp, b) for pp, b in items if b]
    if not items:
        return []
    qv = _encode([query])[0]
    bvs = _encode([b for _, b in items])
    scored = [(float(np.dot(qv, bv)), pp) for (pp, _), bv in zip(items, bvs)]
    scored.sort(key=lambda x: (-x[0], str(getattr(x[1], "t_event", ""))))
    return [pp for _, pp in scored[:k]]


def _hybrid_rank(state, query: str, k: int) -> list:
    """hybrid = RRF(lexical, dense)：两路各自排序，倒数排名融合。mem0/Hindsight 同款多路融合，
    但两路读的都是可采纳 S(t)。"""
    q = tokenize(query)
    lex = [pp for pp in state if getattr(pp, "value", "") and lexical_score(pp, q) > 0]
    lex.sort(key=lambda pp: -lexical_score(pp, q))
    den = _dense_rank(state, query, k=len(state) or 1)
    rrf = {}
    for rank, pp in enumerate(lex):
        rrf_add(rrf, pp, rank)
    for rank, pp in enumerate(den):
        rrf_add(rrf, pp, rank)
    ranked = sorted(rrf.items(), key=lambda kv: -kv[1][0])
    return [pp for _, (_, pp) in ranked[:k]]


def rrf_add(rrf, pp, rank, c: int = 60):
    key = id(pp)
    prev = rrf.get(key, (0.0, pp))
    rrf[key] = (prev[0] + 1.0 / (c + rank + 1), pp)


def _try_register():
    """尝试注册 dense/hybrid ranker；依赖缺失则静默跳过（core 保持零依赖可用）。"""
    try:
        import sentence_transformers  # noqa: F401
        register_ranker("dense", _dense_rank)
        register_ranker("hybrid", _hybrid_rank)
        return True
    except Exception:
        return False


REGISTERED = _try_register()
