"""read_relevant 读算子的确定性单测（不依赖 LLM/embedding）。

保护 0722 架构第 4 个核心读算子：分词、词法评分、query 扩展、可采纳性约束。
dense/hybrid 需重依赖，不在此测（隔离在 relevance_dense，端到端 job 覆盖）。
"""
from __future__ import annotations

from trace.core.machine import Proposition
from trace.core.relevance import (
    tokenize, lexical_score, rank_relevant, expand_query, _QUERY_EXPAND,
)


def _p(subj, val, t="2023-01-01"):
    return Proposition(subj, "narrative", val, t, f"m_{subj}")


def run():
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, f"FAIL: {msg}"
        n += 1
        print(f"  ✅ {msg}")

    # 分词：长度过滤 + 小写
    check(tokenize("The Beach Trip") == {"the", "beach", "trip"}, "tokenize 小写+去短词(>2)")
    check("go" not in tokenize("go to beach"), "tokenize 过滤 ≤2 字符词")

    # 词法评分：重叠越多分越高，无重叠为 0
    q = tokenize("beach vacation")
    check(lexical_score(_p("a", "beach vacation was great"), q) > 0, "词法命中>0")
    check(lexical_score(_p("b", "unrelated office meeting"), q) == 0, "词法无重叠=0")

    # rank_relevant：相关项排前，取 top-k
    S = [_p("hit", "beach vacation sandcastle"), _p("mid", "beach only")] + \
        [_p(f"n{i}", f"unrelated topic {i}") for i in range(10)]
    ranked = rank_relevant(S, "beach vacation", k=3, expand=False)
    check(ranked[0].subject == "hit", "最相关项排第一")
    check(len(ranked) <= 3, "top-k 截断生效")

    # query 扩展：抽象词映射到具体词；无副作用
    check("running" in expand_query("what are her hobbies"), "hobbies 扩展含具体活动词")
    check(expand_query("when did she move") == "when did she move", "无映射词原样返回(零副作用)")
    check(expand_query("her hobbies").startswith("her hobbies"), "扩展保留原 query 前缀")
    check("single" in expand_query("relationship status"), "relationship 扩展含 single")

    # 扩展让语义鸿沟项召回：query 'hobbies' 命中证据 'running'（词法）
    S2 = [_p("run", "I went running this morning")] + [_p(f"n{i}", f"chat {i}") for i in range(8)]
    # 词法+扩展：expand 后 query 含 'running' → 直接词法命中
    r = rank_relevant(S2, "what hobbies does she have", k=3, expand=True)
    check(any(p.subject == "run" for p in r), "扩展后词法召回到语义鸿沟证据(running)")
    r_no = rank_relevant(S2, "what hobbies does she have", k=3, expand=False)
    check(not any(p.subject == "run" for p in r_no), "无扩展时词法漏掉(证实扩展的作用)")

    # 词表完整性：每个映射的 expansion 非空
    check(all(v.strip() for v in _QUERY_EXPAND.values()), "扩展词表无空值")

    # ★架构内保证（定理2级）：召回只在 execute 输出的 S(t) 上发生 → 被删命题任何 scorer 都召回不到。
    #   这是"在SRG架构上做召回"的结构铁证：dense/expand 换的是尺子，读的对象恒为可采纳集 S(t)。
    from trace.core.transition import Transition
    from trace.core.machine import execute
    from trace.core.governance import Principal
    prog = [
        Transition(memory_id="s1", subject="user", attribute="secret_pw", op="ASSERT",
                   t_event="2023-01-01", t_ingest=1, content="secret password hunter2 swordfish",
                   scope="same_user", sensitivity=None, valid_until=None),
        Transition(memory_id="s2", subject="user", attribute="secret_pw", op="DELETE",
                   t_event="2023-01-02", t_ingest=2, content="",
                   scope="same_user", sensitivity=None, valid_until=None),
    ]
    St = execute(prog, t_query="2023-06-01", principal=Principal(allow_sensitive=True))
    leaked = rank_relevant(St, "what is the secret password hunter2 swordfish", k=5, expand=False)
    check(not any("secret" in p.attribute for p in leaked),
          "被DELETE命题不在S(t)→召回捞不到(定理2级合规,架构内保证)")

    print(f"\n通过 {n} / {n}\n✅ read_relevant 读算子单测全绿")


if __name__ == "__main__":
    run()
