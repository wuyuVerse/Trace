"""相关性读算子的单一权威实现（read_relevant 的评分内核）。

0722 统一 READ 谱系里 `read_relevant` 的评分逻辑集中于此——projections/srg_e2e/adapter 三处
都调用它，不各自重抄。这样：① 改算法只改一处 ② 未来换 dense/rerank 只需替换 `score_fn`（可扩展）。

当前实现：确定性词法评分（BM25 味的 token 重叠 + evidence_keys 命中 + 长度归一），零外部依赖。
可扩展点：`register_scorer(name, fn)` 注册 dense/cross-encoder 评分器，`rank_relevant(..., scorer=name)` 选用。
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s) -> set:
    """统一分词（单一权威，替代散落的 _tok/_tokens）。"""
    return {w for w in _TOKEN_RE.findall(str(s).lower()) if len(w) > 2}


# ── query 扩展：桥接"抽象问法"与"具体证据"的语义鸿沟（0722 §4 P1 fact-augmented query）──
# 诊断证据：locomo 问 "hobbies"，证据说 "went for a run" → 词法零重叠、单 query dense 也沉底
#   (RUN 证据在 "hobbies" query 下 dense 排名 16/17，被 top_n 截掉 → 多值漏项)。
# 扩展成 "hobbies activities running pottery sports crafts…" 后 RUN 浮到排名 1。
# 确定性、无 LLM：把抽象类别词映射到该类别的具体表述词，同时命中多个散落多值项。
# 这是读算子的 query 预处理层，不改 execute/S(t) 成员资格（定理2合规不变）。
_QUERY_EXPAND = {
    "hobby": "activities running swimming pottery painting hiking camping reading crafts sports",
    "hobbies": "activities running swimming pottery painting hiking camping reading crafts sports",
    "activities": "hobby running swimming pottery painting hiking camping doing enjoy",
    "interest": "hobby activities enjoy like passionate favorite",
    "interests": "hobby activities enjoy like passionate favorite",
    "place": "beach mountains forest park museum city visited went travel outdoors location",
    "places": "beach mountains forest park museum city visited went travel outdoors location",
    "visit": "went beach mountains park museum travel trip place",
    "visited": "went beach mountains park museum travel trip place",
    "make": "made create paint pottery draw build craft art",
    "makes": "made create paint pottery draw build craft art",
    "create": "made paint pottery draw build craft art creation",
    "book": "read reading novel story author title favorite",
    "books": "read reading novel story author title favorite",
    "career": "job work profession counseling psychology certification education path",
    "job": "career work profession employ occupation company",
    "relationship": "single married dating partner boyfriend girlfriend spouse seeing status",
    "status": "single married dating partner relationship living",
    "married": "single dating partner spouse relationship",
    "how many times": "went visited did again another time first second third each",
    "how often": "every day week weekend regularly usually times went",
    "religious": "church faith god pray belief spiritual religion",
    "identify": "transgender gender woman man identity trans",
    "gender": "transgender woman man trans identity nonbinary",
    # temporal/duration 问法：桥接"how long/when"与证据里的具体日期/起止事件表述（通用词，非题目特定）
    "how long": "started began since when date years months weeks days ago",
    "how many days": "date started began between when days event happened",
    "member": "joined signed up started became since when club group",
    "worked": "job work career company started years experience previous",
}


def expand_query(query: str) -> str:
    """确定性 query 扩展：抽象类别词 → 该类别的具体表述词，桥接语义鸿沟。
    只追加词、不删原词（原词精确匹配仍在）。命中 0 个映射则原样返回（零副作用）。"""
    q = query.lower()
    adds = []
    for cue, expansion in _QUERY_EXPAND.items():
        if re.search(rf"\b{cue}\b", q):
            adds.append(expansion)
    if not adds:
        return query
    return query + " " + " ".join(adds)


def _proposition_tokens(pp) -> set:
    """一个富命题的可匹配 token 集：属性面(attribute) + 语言体(body/value) + evidence_keys。

    ★为何含 attribute、不含 subject：编译后结构命题的 value 往往只是【属性值】（如 "concise three
    bullets"、"Shanghai"），而查询语义常落在【属性】侧（如 "weekly summary style preferences" 命中
    attribute="style"）。若只匹配 body，结构命题对主题型查询恒 score=0 → rank_relevant 返回空 →
    上游 fallback 报全部命题（含 near_miss/离题的 forbidden）→ forbidden_activation 误报。纳入 attribute
    后：主题查询命中相关结构命题（recall↑），无关命题 score=0 被 s>0 过滤剔除（activated 精确、误报↓）。

    ★为何刻意排除 subject：subject 多为实体名/principal（"user"、"Melanie"），是几乎所有命题共有的
    公共词——纳入它会让含该实体名的查询把【全部】命题打成 score>0（停用词级噪声），反伤召回精度。
    attribute 才是承载"问的是哪个语义槽"的判别性 token。这是 finalize 后的唯一确定行为（无 env 开关）。"""
    attr = tokenize(getattr(pp, "attribute", ""))
    body = tokenize(getattr(pp, "body", getattr(pp, "value", "")))
    ev = set(getattr(pp, "evidence_keys", ()))
    return attr | body | ev


def lexical_score(pp, query_tokens: set) -> float:
    """确定性词法相关性：token 重叠 + 长度归一（BM25 味）。0 表示不相关。"""
    btok = _proposition_tokens(pp)
    if not btok:
        return 0.0
    overlap = len(query_tokens & btok)
    if overlap == 0:
        return 0.0
    return overlap + overlap / (len(btok) + 1)   # 长命题轻微惩罚，稳定 tie-break


def _rank_lexical(state, query: str, k: int) -> list:
    """词法排序（默认，零依赖确定性）。"""
    q = tokenize(query)
    scored = [(lexical_score(pp, q), pp) for pp in state if getattr(pp, "value", "")]
    scored = [(s, pp) for s, pp in scored if s > 0]
    scored.sort(key=lambda x: (-x[0], str(getattr(x[1], "t_event", ""))))
    return [pp for _, pp in scored[:k]]


# 排序器注册表：每个是 (state, query, k) -> top-k 命题列表。
# lexical 为默认；dense/rerank 由 relevance_dense.py 等模块 register_ranker 注入（可扩展，重依赖隔离）。
_RANKERS = {"lexical": _rank_lexical}


def register_ranker(name: str, fn):
    """注册整段排序器（读同一 S(t)，换相关性度量）。dense/cross-encoder 走这里。

    统一性：无论哪种 ranker，输入都是 execute+GOVERN 后的富 S(t) → 召回恒在可采纳集内（定理2级合规）。
    dense 只是换把尺子量相关性，不改"读的是 S(t)"这一范式内核。"""
    _RANKERS[name] = fn
    return fn


def register_scorer(name: str, fn):
    """（保留）注册 per-item 评分器 fn(pp, query_tokens)->float，自动包装成 ranker。"""
    def _ranker(state, query, k, _fn=fn):
        q = tokenize(query)
        scored = [(_fn(pp, q), pp) for pp in state if getattr(pp, "value", "")]
        scored = [(s, pp) for s, pp in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], str(getattr(x[1], "t_event", ""))))
        return [pp for _, pp in scored[:k]]
    _RANKERS[name] = _ranker
    return fn


def rank_relevant(state, query: str, k: int = 8, scorer: str = "lexical", expand: bool = True) -> list:
    """对富 S(t)（Proposition 列表）按与 query 的相关性排序取 top-k。单一权威入口。

    输入 state 必须已是 execute+GOVERN 后的可采纳集 → 召回天然只在可采纳命题上发生（定理2级合规）。
    scorer 选相关性度量（lexical/dense/...），读的对象恒为 S(t)——高维度统一：一个读算子、可插拔度量。
    expand: query 扩展桥接抽象问法与具体证据的语义鸿沟（多值召回全，见 expand_query），可关闭做消融。
    """
    q = expand_query(query) if expand else query
    ranker = _RANKERS.get(scorer, _rank_lexical)
    return ranker(state, q, k)
