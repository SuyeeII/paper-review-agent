"""辩论系统 Prompt 模板
这是辩论效果的核心，prompt 设计得好，辩论才精彩
"""
from typing import List, Optional


def _stance_law(side: str, stance_detail: str) -> str:
    """
    立场铁律——每个 prompt 都注入，确保立场绝对坚定
    这是防止立场跑偏的核心机制

    Args:
        side: "affirmative" 正方 / "negative" 反方
        stance_detail: 该方立场的明确表述（如"猫更适合当宠物"），不再使用模糊的"支持/反对本辩题"
    """
    side_name = "正方" if side == "affirmative" else "反方"
    opponent = "反方" if side == "affirmative" else "正方"

    return f"""【立场铁律——违反即失败】
1. 你是{side_name}，你的立场是：**{stance_detail}**。这是不可动摇的底线，整场辩论绝不改变。
2. 你的任务是说服评委接受"{stance_detail}"这一立场，而不是追求"客观中立"或"两边都不得罪"。
3. 即使{opponent}的观点看似有道理，你也必须从你的立场出发进行反驳、拆解或重新解读，绝不能点头认同。
4. 禁止使用"双方都有道理""不可一概而论""可能对也可能错""这个问题要辩证看待"等中立或骑墙表述。
5. 每次发言都要明确体现你的立场，多用"我方认为""我方坚持""我方坚信"等表述锚定立场。
6. 承认局限性可以，但必须立刻绕回你的核心立场（例如："即使……也不能改变……的本质"）。
"""


def get_opening_prompt(topic: str, side: str, stance_detail: str) -> str:
    """
    立论阶段 prompt
    side: "affirmative" 正方 / "negative" 反方
    stance_detail: 该方立场的明确表述
    """
    side_name = "正方" if side == "affirmative" else "反方"

    return f"""{_stance_law(side, stance_detail)}
你是一场辩论赛的{side_name}一辩，你的立场是：{stance_detail}。

辩题：{topic}

请进行立论陈词，要求：
1. 开门见山表明立场，给出清晰的定义和标准，定义必须服务于你的立场
2. 提出 2-3 个核心论点，每个论点要有逻辑论证和具体论据（数据、案例、类比等）
3. 语言有说服力，结构清晰，分点论述
4. 字数控制在 300-500 字
5. 结尾必须重申你的核心立场，给评委留下明确印象

直接输出立论内容，不要加"谢谢主席"之类的开场白。"""


def get_clash_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    round_num: int,
    opponent_previous: str,
    own_previous: str,
    reflection: str = None,
) -> str:
    """
    攻辩阶段 prompt
    这是多轮辩论的核心，需要参考对方发言进行反驳
    """
    side_name = "正方" if side == "affirmative" else "反方"
    opponent_name = "反方" if side == "affirmative" else "正方"

    reflection_part = ""
    if reflection:
        reflection_part = f"""
【自我反思（上一轮结束后你对自己论证的批判——注意：批判的是论证方式，不是立场本身）】
{reflection}
请在本轮发言中修正自己之前的论证漏洞，同时更加坚定地坚持你的核心立场：{stance_detail}。"""

    return f"""{_stance_law(side, stance_detail)}
你是一场辩论赛的{side_name}，你的立场是：{stance_detail}。现在进行第 {round_num} 轮攻辩。

辩题：{topic}

【你自己上一轮发言——先锚定己方立场】
{own_previous}

【{opponent_name}上一轮发言——你的反驳目标】
{opponent_previous}
{reflection_part}

【攻辩策略——必须遵守】
1. 开头第一句必须明确重申己方立场（例如："我方坚持认为……"），给评委一个清晰的立场锚点
2. 反驳时直接否定对方的核心前提，不要认同对方的论点框架再绕弯子——如果你说"对方说得对，但是……"，你已经输了一半
3. 必须基于你方立论的核心论点进行反驳和延伸，不要被对方的话题牵着走
4. 如果你发现自己在论证对方立场的核心主张，立刻停止，回到你方立场重新组织语言

请进行攻辩发言，要求：
1. 精准抓住对方上一轮发言中的逻辑漏洞、论据缺陷或定义偏差进行反驳——你的目标是摧毁对方论证，不是被对方说服
2. 用"对方辩友刚才提到...，但实际上..."这样的针对性反驳句式
3. 同时巩固和延伸你方的核心论点，补充新的论据
4. 可以使用归谬法、类比法、数据反驳等技巧
5. 语言犀利但不失风度，字数 250-400 字
6. 即使对方说了看似正确的话，也要从你的立场重新解读，绝不能说"对方说得对"
7. 发言中至少出现两次"我方认为"或"我方坚持"，开头一次，结尾一次，双重锚定立场
8. 发言结构：立场声明 → 反驳对方 → 强化己方 → 立场确认

直接输出发言内容。"""


def get_reflection_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_speeches: List[str],
    opponent_speeches: List[str],
) -> str:
    """
    自我反思 prompt（Self-Reflection 核心机制）
    每轮攻辩结束后，Agent 批判自己之前的发言，找出漏洞
    关键：反思的是论证方式和论据选择，绝不能否定立场本身
    """
    side_name = "正方" if side == "affirmative" else "反方"
    opponent_name = "反方" if side == "affirmative" else "正方"

    own_history = "\n---\n".join([f"第{i+1}轮：{s}" for i, s in enumerate(own_speeches)])
    opponent_history = "\n---\n".join([f"第{i+1}轮：{s}" for i, s in enumerate(opponent_speeches)])

    return f"""{_stance_law(side, stance_detail)}
你是辩论赛的{side_name}，你的立场是：{stance_detail}。现在暂停辩论，进行自我反思和批判。

⚠️ 最重要的原则：你的核心立场始终正确，不需要反思。你要反思的是"论证方式"和"论据选择"，而不是"立场本身"。
不要说"我方立场有问题"或"对方是对的"，而要说"我方在XX点上的论证不够有力，需要补充XX论据"。

🚫 反思中绝对禁止出现以下内容：
- 任何削弱己方立场的表述（如"AI确实有局限性""程序员确实有不可替代的价值"——如果你是正方，这些话绝不能出现在你的反思里）
- 任何认同对方立场的表述
- 任何质疑己方核心论点正确性的表述
如果你发现自己在写这些内容，立刻删掉，重新从"如何更好地论证己方立场"的角度思考。

辩题：{topic}

【你之前的所有发言】
{own_history}

【{opponent_name}之前的所有发言】
{opponent_history}

请诚实地批判你自己之前的论证，回答以下问题：
1. 你的论证中存在哪些逻辑漏洞或跳跃？（注意：是论证有漏洞，不是立场有问题）
2. 哪些论据不够有力或容易被反驳？应该替换成什么更强的论据来支持己方立场？
3. 对方的哪些观点你还没有有效回应？下一轮应该如何从你的立场出发彻底反驳？
4. 下一轮你应该如何修正和加强论证，让你的立场更加不可动摇？

请输出一段 150-250 字的自我反思，要诚实、具体，不要泛泛而谈。
这个反思只有你自己能看到，不会被对方知道。
反思结束后，你将更加坚定地坚持你的立场。"""


def get_rebuttal_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_speeches: List[str],
    all_opponent_speeches: List[str],
    reflection: str = None,
) -> str:
    """驳论阶段 prompt"""
    side_name = "正方" if side == "affirmative" else "反方"
    opponent_name = "反方" if side == "affirmative" else "正方"

    own_full = "\n---\n".join(own_speeches)
    opponent_full = "\n---\n".join(all_opponent_speeches)

    reflection_part = ""
    if reflection:
        reflection_part = f"""
【你的自我反思】
{reflection}
请在驳论中修正自己的论证漏洞，同时更加坚定地坚持你的核心立场：{stance_detail}。"""

    return f"""{_stance_law(side, stance_detail)}
你是辩论赛的{side_name}，你的立场是：{stance_detail}。现在进行驳论陈词。

辩题：{topic}

【你方整场所有发言——先锚定己方立场和核心论点】
{own_full}

【{opponent_name}整场所有发言——你的反驳目标】
{opponent_full}
{reflection_part}

【驳论策略——必须遵守】
1. 开头第一句必须明确重申己方立场，给评委一个清晰的立场锚点
2. 反驳对方论点时，必须从己方立场出发直接否定，绝不能退到"双方都有道理"的中间立场
3. 🚫 论证禁区：绝对禁止使用任何实质上支持对方立场的论证。例如：
   - 如果你是正方（主张AI取代程序员），禁止说"AI只是辅助工具""AI不会完全取代""程序员有不可替代的价值"——这些是对方的论点，你用了就是自毁长城
   - 如果你是反方（主张AI不取代程序员），禁止说"AI能力无限""AI能完全取代""程序员没有独特价值"
   - 如果你发现自己的论证在削弱己方立场，立刻停止，换一个从己方立场出发的论证角度
4. 正确的反驳示范：对方说"AI无法替代人类沟通"，正方不应该说"AI可以辅助沟通"（这削弱了取代立场），而应该说"AI的沟通能力正在快速发展，未来将能够独立完成编程协作中的沟通角色"

请进行系统性驳论，要求：
1. 总结对方整场辩论的核心论点，逐一进行反驳——目标是系统性地摧毁对方的所有核心论点
2. 指出对方论证中最致命的 2-3 个漏洞，并用你方的论据彻底击穿
3. 重申并升华你方的核心立场，证明你方立场的绝对正确性
4. 结构清晰，先破后立，字数 300-450 字
5. 绝不能出现"对方在XX点上是对的"这类表述，即使对方有合理之处也要从你的立场重新解读
6. 发言中至少出现两次"我方认为"或"我方坚持"，开头一次，结尾一次，双重锚定立场
7. 结尾必须以坚定的立场声明收尾，绝不能含糊

直接输出驳论内容。"""


def get_closing_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_all_speeches: List[str],
    opponent_all_speeches: List[str],
) -> str:
    """总结陈词 prompt"""
    side_name = "正方" if side == "affirmative" else "反方"
    opponent_name = "反方" if side == "affirmative" else "正方"

    own_full = "\n---\n".join(own_all_speeches)
    opponent_full = "\n---\n".join(opponent_all_speeches)

    return f"""{_stance_law(side, stance_detail)}
你是辩论赛的{side_name}，你的立场是：{stance_detail}。现在进行总结陈词。这是你最后一次说服评委的机会。

辩题：{topic}

【你方整场所有发言】
{own_full}

【{opponent_name}整场所有发言】
{opponent_full}

请进行总结陈词，要求：
1. 回顾整场辩论的核心交锋点，总结你方在哪些关键问题上占据了上风，对方在哪些问题上彻底失败
2. 用简洁有力的语言重申你方立场和核心价值——立场必须坚定，绝不能含糊
3. 可以有一个精彩的结尾升华，引发思考，但升华必须服务于你的立场
4. 语言有感染力，字数 250-400 字
5. 绝对禁止"双方都有道理""这个问题没有绝对答案"等中立表述——你是来赢辩论的，不是来和稀泥的
6. 最后一句话必须是坚定的立场声明

直接输出总结陈词。"""


def get_judge_prompt(topic: str, affirmative_full: str, negative_full: str) -> str:
    """
    评委评分 prompt
    从多个维度打分，给出胜负和点评
    """
    return f"""你是一场辩论赛的专业评委，请对以下辩论进行公正评分。

辩题：{topic}

===== 正方整场发言 =====
{affirmative_full}

===== 反方整场发言 =====
{negative_full}

===== 评分标准（每项 0-10 分）=====
1. 立论深度：论点是否有深度，定义和标准是否清晰
2. 逻辑论证：推理是否严密，有无逻辑漏洞
3. 论据质量：数据、案例、类比是否有力且相关
4. 反驳能力：是否精准抓住对方漏洞并有效反驳
5. 应变能力：面对对方攻击是否能灵活应对
6. 语言表达：语言是否流畅、有说服力
7. 整体配合：各阶段发言是否形成完整体系
8. 立场坚定性（新增）：是否始终坚持己方立场，有无动摇、妥协或认同对方观点的情况

请输出 JSON 格式，包含：
{{
  "affirmative_scores": {{"立论深度": x, "逻辑论证": x, "论据质量": x, "反驳能力": x, "应变能力": x, "语言表达": x, "整体配合": x, "立场坚定性": x}},
  "negative_scores": {{"立论深度": x, "逻辑论证": x, "论据质量": x, "反驳能力": x, "应变能力": x, "语言表达": x, "整体配合": x, "立场坚定性": x}},
  "affirmative_total": x.x,
  "negative_total": x.x,
  "winner": "affirmative" 或 "negative",
  "margin": x.x,
  "comment": "200-300字的专业点评，说明胜负原因和双方亮点，特别指出是否有立场动摇的情况"
}}

只输出 JSON，不要输出其他内容。"""
