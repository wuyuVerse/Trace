"""SRG 治理层：Principal（谁在问）与可见性判定。

GOVERN 门在这里定义授权域与敏感级的可见性规则——合规删除/权限隔离由代码强制，
不靠 LLM 自觉。这是 SRG 相对检索库的 safety 结构性优势的来源。
"""

from __future__ import annotations

from dataclasses import dataclass

# 授权域层级：粗到细。visible(scope, principal) 判定一条记忆对该 principal 是否可见。
_PUBLIC_SCOPES = ("same_user", "", None)
# 敏感度采 fail-safe 语义：只有【显式非敏感】的标签才放行，其余（任意非空标签）一律当敏感。
# 这样 high/confidential/secret/pii 等自然表述不会因不在白名单而漏判——完备性证明要求保守。
_NONSENSITIVE = ("", "none", "public", "normal", "low", None)
# 硬约束级：治理规则明确「结构性禁止召回」的级别。与「可授权敏感」（secret/confidential
# 等，授权 principal 可读其历史）不同——restricted 是策略声明本身（如"禁存支付卡号"），
# 无视 allow_sensitive 对任何 principal 都不可召回。是 per-memory 公开字段(privacy_level)
# 驱动的强治理，竞品同样可见该字段却不做结构化执行——正是范式的 safety 结构性优势。
_HARD_RESTRICTED = ("restricted",)


@dataclass(frozen=True)
class Principal:
    """谁在问：授权域 + 项目/角色边界 + 是否允许敏感信息。"""
    scope: str = "same_user"
    allow_sensitive: bool = False


def visible(slot_scope: str, slot_sensitivity: str | None, p: Principal) -> bool:
    """一条记忆槽位在该 principal 下是否可见（GOVERN 的单元判定）。"""
    if slot_scope not in _PUBLIC_SCOPES and slot_scope != p.scope:
        return False
    sens = (slot_sensitivity or "").lower() if isinstance(slot_sensitivity, str) else slot_sensitivity
    # 硬约束级：无视 allow_sensitive 一律不可见（结构性禁止召回）
    if isinstance(sens, str) and sens in _HARD_RESTRICTED:
        return False
    # fail-safe：只要不是显式非敏感标签，就视为敏感（拿不准当敏感，禁泄漏优先）
    is_sensitive = sens not in _NONSENSITIVE
    if is_sensitive and not p.allow_sensitive:
        return False
    return True
