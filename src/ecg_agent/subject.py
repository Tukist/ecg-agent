"""受检者人群参数（年龄 / 性别 / BMI）→ 筛查阈值先验。

设计原则
--------
1. 人群表只给**先验**，不替代 L5 个体定标：顺序是「人群先验 → 个人基线微调」。
2. 证据分级写在字段旁：年龄段静息心率区间来自临床常规参考（PALS/AHA 各年龄段
   静息心率范围），可直接核查；性别与 BMI 的修正属于经验性微调（弱证据），
   幅度刻意压小，且只作用于**心动过速上限**，避免把人群假设放大成结论。
3. 不装作已验证：本项目数据（MIT-BIH）全为成人。儿科档位没有在本项目数据上
   验证过；房颤筛查在 18 岁以下不启用（模型训练集为成人）。
4. 儿科不让 L5 放宽心率阈值（rate_fixed=True）：儿童 5 分钟基线受哭闹/活动
   影响大，用年龄段区间比用个人基线更稳。

口径提醒：这些仍是"提示级"阈值，不是诊断标准。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AgeBand", "SubjectProfile", "band_for", "normalize_sex", "prior_for",
    "summary_cn", "threshold_text",
]


@dataclass(frozen=True)
class AgeBand:
    """一个年龄段的人群先验。

    hr_low / hr_high : 该年龄段静息心率的正常参考区间（临床常规值）
    tachy / brady    : 触发提示的阈值。儿科取正常区间上下界；
                       成人沿用项目原有口径（过速 100、过缓 50，偏保守）
    pause_s          : 长间歇阈值。3.0 s 对各年龄都适用，L5 只会往上调
    af_validated     : 房颤筛查是否在该人群验证过（本项目仅成人）
    """

    key: str
    label: str
    age_min: float
    age_max: float
    hr_low: int
    hr_high: int
    tachy: float
    brady: float
    pause_s: float = 3.0
    af_validated: bool = True


# 年龄单位：岁（0.083 ≈ 1 个月）。区间为左闭右开。
BANDS: tuple[AgeBand, ...] = (
    AgeBand("neonate", "新生儿", 0.0, 1 / 12, 100, 180, 180, 100, af_validated=False),
    AgeBand("infant", "婴儿", 1 / 12, 1, 100, 160, 160, 90, af_validated=False),
    AgeBand("toddler", "幼儿", 1, 3, 90, 150, 150, 80, af_validated=False),
    AgeBand("preschool", "学龄前", 3, 6, 80, 140, 140, 70, af_validated=False),
    AgeBand("school", "学龄期", 6, 12, 70, 120, 120, 60, af_validated=False),
    AgeBand("adolescent", "青少年", 12, 18, 60, 100, 100, 50, af_validated=False),
    AgeBand("adult", "成人", 18, 65, 60, 100, 100, 50, af_validated=True),
    # 老年静息心率偏低，过缓阈值放宽到 45 以减少误报；如追求敏感度可调回 50
    AgeBand("elderly", "老年", 65, 200, 55, 90, 100, 45, af_validated=True),
)

_BAND_BY_KEY = {b.key: b for b in BANDS}

_SEX_ALIASES = {
    "male": "male", "m": "male", "男": "male", "男性": "male",
    "female": "female", "f": "female", "女": "female", "女性": "female",
}


def normalize_sex(value) -> str | None:
    """把各种写法统一成 male / female；无法识别返回 None。"""
    if value is None:
        return None
    v = str(value).strip().lower()
    return _SEX_ALIASES.get(v)


def band_for(age: float | None) -> AgeBand:
    """按年龄取年龄段；None / 非正数按成人处理。"""
    if age is None or age <= 0:
        return _BAND_BY_KEY["adult"]
    for b in BANDS:
        if b.age_min <= age < b.age_max:
            return b
    return _BAND_BY_KEY["elderly"]


@dataclass(frozen=True)
class SubjectProfile:
    """受检者信息。所有字段可选：不填就按成人默认阈值。"""

    age: float | None = None      # 岁；0 或 None 视为未填写
    sex: str | None = None        # male / female / 男 / 女
    bmi: float | None = None      # kg/m^2；0 或 None 视为未填写

    def __post_init__(self):
        object.__setattr__(self, "sex", normalize_sex(self.sex))
        if self.age is not None and self.age <= 0:
            object.__setattr__(self, "age", None)
        if self.bmi is not None and self.bmi <= 0:
            object.__setattr__(self, "bmi", None)


def summary_cn(profile: SubjectProfile | None) -> str:
    """给报告/界面用的一句话受检者描述。"""
    p = profile or SubjectProfile()
    if p.age is None:
        base = "未提供年龄（按成人默认）"
    else:
        base = f"{band_for(p.age).label}，{p.age:g} 岁"
    extras = []
    if p.sex:
        extras.append("男" if p.sex == "male" else "女")
    if p.bmi is not None:
        extras.append(f"BMI {p.bmi:g}")
    return base + (f"（{'，'.join(extras)}）" if extras else "")


def threshold_text(prior: dict) -> str:
    """把有效阈值渲染成一句话（报告、CLI、网页共用）。"""
    s = (f"过速 >{prior['tachy']:.0f} bpm、过缓 <{prior['brady']:.0f} bpm、"
         f"停搏 >{prior['pause_s']:.1f} s")
    if not prior.get("af_enabled", True):
        s += "；房颤筛查未启用"
    return s


def prior_for(profile: SubjectProfile | None = None) -> dict:
    """人群档案 → 筛查阈值先验。

    返回的 dict 直接喂给 strategy.run / rhythm.screen_all：
      pause_s / tachy / brady / af_cv / af_enabled
    其余字段只用于展示与审计（summary、notes、调整量）。
    """
    p = profile or SubjectProfile()
    age_known = p.age is not None
    band = band_for(p.age) if age_known else _BAND_BY_KEY["adult"]
    notes: list[str] = []

    # —— 性别：女性静息心率略高（约 2–4 bpm），只上调过速上限。
    #    过缓下限不动，避免为了"人群正确"反而增加误报。证据弱，幅度小。
    sex_shift = 2.0 if p.sex == "female" else 0.0
    if sex_shift:
        notes.append("女性静息心率略高，过速上限 +2 bpm（经验性，弱证据）")

    # —— BMI：肥胖与静息心率轻度升高相关；每高 5 个单位 +1 bpm，上限 +4。
    #    同样只动过速上限，不动过缓/停搏。证据弱，别当临床标准。
    bmi_shift = 0.0
    if p.bmi is not None and p.bmi > 25.0:
        bmi_shift = min(4.0, (p.bmi - 25.0) / 5.0)
        notes.append("BMI 偏高，过速上限 +"
                     f"{bmi_shift:.1f} bpm（经验性，弱证据，上限 +4）")
    if p.bmi is not None and p.bmi >= 30.0:
        notes.append("BMI ≥30：运动伪差/基线漂移风险偏高，建议结合信号质量分解读")
    if p.bmi is not None and p.age is not None and p.age < 18:
        notes.append("儿童 BMI 应按年龄性别百分位判读，此处按成人界值粗略处理")

    if not age_known:
        notes.append("未提供年龄，按成人阈值处理；若为儿童请填写年龄")
    # 儿科说明放在报告的专门段落里（"儿科说明"/"人群限制"），这里不重复
    if band.key == "elderly":
        notes.append("老年静息心率偏低，过缓阈值放宽到 45 bpm 以减少误报")

    tachy = float(band.tachy) + sex_shift + bmi_shift
    prior = {
        "band": band.key,
        "band_label": band.label,
        "age_known": age_known,
        "age": p.age,
        "sex": p.sex,
        "bmi": p.bmi,
        "hr_low": band.hr_low,
        "hr_high": band.hr_high,
        "tachy": round(tachy, 1),
        "brady": float(band.brady),
        "pause_s": float(band.pause_s),
        "af_cv": 0.10,
        "af_enabled": bool(band.af_validated),
        "af_validated": bool(band.af_validated),
        # 儿科：L5 不放宽心率阈值（年龄段区间比 5 分钟个人基线稳）
        "rate_fixed": bool(age_known and p.age is not None and p.age < 18),
        "sex_shift_bpm": sex_shift,
        "bmi_shift_bpm": round(bmi_shift, 1),
        "notes": notes,
    }
    prior["summary"] = summary_cn(p)
    prior["threshold_text"] = threshold_text(prior)
    return prior
