"""safety 治理修复单测 —— 编译层 sensitivity 抽取 + execute GOVERN 全链路（不依赖 LLM）。

证明：① _detect_sensitivity 确定性正确（敏感命中/普通不误伤）；② 带 sensitivity 标签的命题
经 execute 后，对无授权 principal 结构上不进 S(t)（治理定理生效），对授权 principal 可见；
③ 普通命题不受影响（无 recall 回归）。这是 safety 四腿之第3腿（敏感命题治理）的可信度保证。
"""
from __future__ import annotations

from trace.compile.llm_compiler import _detect_sensitivity
from trace.core.transition import Transition
from trace.core.machine import execute
from trace.core.governance import Principal


def _t(mid, subj, attr, val, sens):
    return Transition(memory_id=mid, subject=subj, attribute=attr, op="ASSERT",
                      t_event="2023-01-01", t_ingest=int(mid[1:]) if mid[1:].isdigit() else 1,
                      content=val, scope="same_user", sensitivity=sens, valid_until=None)


def run():
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, f"FAIL: {msg}"
        n += 1
        print(f"  ✅ {msg}")

    # ── ① _detect_sensitivity 确定性正确 ──
    check(_detect_sensitivity(None, "api_key", "sk-abc123def456") == "sensitive", "api_key 值→sensitive")
    check(_detect_sensitivity(None, "password", "hunter2") == "sensitive", "password 属性→sensitive")
    check(_detect_sensitivity(None, "ssn", "123-45-6789") == "sensitive", "SSN 模式→sensitive")
    check(_detect_sensitivity("sensitive", "note", "x") == "sensitive", "LLM 标 sensitive→sensitive")
    check(_detect_sensitivity(None, "home_city", "Shanghai") is None, "普通居住地→None（不误伤）")
    check(_detect_sensitivity(None, "hobby", "running") is None, "普通爱好→None（不误伤）")
    check(_detect_sensitivity(None, "job", "engineer at NovaTech") is None, "普通职业→None（不误伤）")

    # ── ② + ③ 全链路：sensitivity 标签经 execute GOVERN ──
    prog = [
        _t("m1", "user", "api_key", "sk-secret-xyz", _detect_sensitivity(None, "api_key", "sk-secret-xyz")),
        _t("m2", "user", "hobby", "running", _detect_sensitivity(None, "hobby", "running")),
        _t("m3", "user", "home_city", "Shanghai", _detect_sensitivity(None, "home_city", "Shanghai")),
    ]
    # 无授权 principal：敏感命题被 GOVERN 排除
    S_no = execute(prog, "2023-06-01", principal=Principal(allow_sensitive=False))
    attrs_no = {p.attribute for p in S_no}
    check("api_key" not in attrs_no, "无授权：api_key 结构上不进 S(t)（治理定理生效）")
    check("hobby" in attrs_no and "home_city" in attrs_no, "无授权：普通命题不受影响（无 recall 回归）")
    # 授权 principal：全部可见
    S_yes = execute(prog, "2023-06-01", principal=Principal(allow_sensitive=True))
    check("api_key" in {p.attribute for p in S_yes}, "有授权：api_key 可见（治理是投影非删除）")
    check(len(S_yes) == 3, "有授权：全部 3 命题可见")

    # ── 边界：无敏感命题时两种 principal 结果一致（不无端误伤）──
    prog2 = [_t("m1", "user", "hobby", "running", None), _t("m2", "user", "city", "Paris", None)]
    a = {p.attribute for p in execute(prog2, "2023-06-01", principal=Principal(allow_sensitive=False))}
    b = {p.attribute for p in execute(prog2, "2023-06-01", principal=Principal(allow_sensitive=True))}
    check(a == b == {"hobby", "city"}, "无敏感命题：授权/无授权 S(t) 一致（普通题零影响）")

    # ── ④ safety leg2/leg4：activated 精确性（forbidden_activation 根因）──
    # 结构命题 value 常只是【属性值】，查询语义落在【主题/属性】侧。_proposition_tokens 纳入
    # subject/attribute 后：主题查询命中相关命题(recall)，near_miss/离题命题 score=0 被剔除(不误报激活)。
    from trace.core.machine import execute as _exe  # noqa
    from trace.core import rank_relevant  # noqa
    prog3 = [
        _t("m_stable", "summary", "style", "concise three bullets", None),
        _t("m_old", "travel", "departure_city", "Beijing", None),      # 旧值(t_ingest 早)
        _t("m_upd", "travel", "departure_city", "Shanghai", None),     # 新值 → supersede m_old
        _t("m_near", "tool", "flight_pref", "avoid red-eye flights", None),  # near_miss forbidden
        _t("m_off", "hobby", "music", "jazz", None),                   # 离题
    ]
    # 修正 t_ingest 使 supersede 生效（同槽后写覆盖前写）
    prog3[2] = Transition(memory_id="m_upd", subject="travel", attribute="departure_city", op="ASSERT",
                          t_event="2023-01-01", t_ingest=99, content="Shanghai", scope="same_user",
                          sensitivity=None, valid_until=None)
    S3 = execute(prog3, "2023-06-01", principal=Principal(allow_sensitive=True))
    ids3 = {p.memory_id for p in S3}
    check("m_old" not in ids3, "leg: 被 supersede 的旧值(forbidden)经 execute 排除,不入 S(t)")
    act_q1 = {p.memory_id for p in (rank_relevant(S3, "weekly summary style preferences", k=8) or [])}
    check(act_q1 == {"m_stable"}, "leg2/leg4: 主题查询只激活相关命题,near_miss/离题不误报为 activated")
    act_q2 = {p.memory_id for p in (rank_relevant(S3, "travel departure city", k=8) or [])}
    check("m_upd" in act_q2 and "m_off" not in act_q2, "leg2: 结构命题主题匹配命中(recall),离题不激活")

    print(f"\n通过 {n} / {n}\n✅ safety 治理修复单测全绿（编译抽 sensitivity + execute GOVERN，敏感排除、普通不误伤、activated 精确）")


if __name__ == "__main__":
    run()
