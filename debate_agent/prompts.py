"""论文多视角审稿助手 Prompt 模板
核心：针对论文具体内容（方法、实验、数据、结论）做评审，而不是泛泛讨论方向
审稿维度：创新性 · 方法论 · 实验可靠性 · 写作表达
"""
from typing import List, Optional


def _stance_law(side: str, stance_detail: str) -> str:
    """
    审稿立场——每个 prompt 都注入，确保审稿立场明确
    支持接受方：论证论文值得接受，指出论文的优点和贡献
    建议拒稿方：指出论文的不足和问题，论证需要拒稿或大修

    Args:
        side: "affirmative" 支持接受 / "negative" 建议拒稿
        stance_detail: 该方立场的明确表述
    """
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"
    opponent = "建议拒稿方" if side == "affirmative" else "支持接受方"

    if side == "affirmative":
        mission = "你的任务是论证这篇论文值得被接受，指出论文的创新点、方法优势、实验贡献和写作亮点。"
        forbidden = "禁止说'这篇论文没有价值''应该拒稿''创新性严重不足'等否定论文的表述。"
    else:
        mission = "你的任务是指出这篇论文的不足和问题，论证需要拒稿或大修，从创新性、方法论、实验、写作四个维度找出缺陷。"
        forbidden = "禁止说'这篇论文非常完美''直接接受即可''没有任何问题'等过度肯定论文的表述。"

    return f"""【审稿立场——必须遵守】
1. 你是{side_name}审稿人，你的立场是：**{stance_detail}**。这是你的审稿基调，整场审稿保持一致。
2. {mission}
3. 即使{opponent}的意见看似有道理，你也必须从你的立场出发进行回应、补充或重新解读，不能简单认同对方。
4. {forbidden}
5. 每次审稿意见都要明确体现你的立场，多用"我方认为""我方指出""我方强调"等表述锚定立场。
6. 承认论文的某些局限性可以，但必须立刻绕回你的核心立场（例如：支持接受方："即使实验规模有限，也不能改变方法创新的本质"；建议拒稿方："即使写作清晰，也不能掩盖实验设计的根本缺陷"）。

【审稿核心原则——针对论文具体内容评审】
⚠️ 最重要：必须针对论文的具体内容（研究方法、实验设计、数据结果、理论推导、写作结构）做评审，绝对不能泛泛讨论"这个研究方向好不好"。
- 错误示范："强化学习在自动驾驶中很有前景/强化学习还不成熟"（这是在讨论方向，不是在评审这篇论文）
- 正确示范："本文提出的注意力机制在多智能体协作中有效，但实验只在简单交叉路口场景验证，缺乏复杂场景的泛化性测试"（这是在评审这篇论文的具体方法和实验）
- 每一条评审意见都要引用论文中的具体内容（方法名称、实验设置、数据结果、论文结构），不能空泛地说"创新性强/创新性不足"
"""


def get_opening_prompt(topic: str, side: str, stance_detail: str) -> str:
    """
    初审意见 prompt
    side: "affirmative" 支持接受 / "negative" 建议拒稿
    """
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"

    if side == "affirmative":
        focus = "指出论文的创新点、方法优势、实验贡献和写作亮点，论证为什么这篇论文值得被接受"
        dimensions = """
- 创新性：论文提出了什么新方法/新理论/新发现？与现有工作的差异在哪里？
- 方法论：研究方法是否合理？技术路线是否清晰？理论推导是否严谨？
- 实验可靠性：实验设计是否科学？数据集是否充分？对比实验是否公平？结果是否可复现？
- 写作表达：论文结构是否清晰？语言是否准确？图表是否规范？参考文献是否全面？"""
    else:
        focus = "指出论文在创新性、方法论、实验、写作四个维度的不足和问题，论证为什么这篇论文需要拒稿或大修"
        dimensions = """
- 创新性：论文的创新点是否足够？是否只是现有方法的简单组合？与现有工作的差异是否显著？
- 方法论：研究方法是否有缺陷？技术路线是否有漏洞？理论推导是否不严谨？假设是否不合理？
- 实验可靠性：实验设计是否有问题？数据集是否不足？对比实验是否不公平？结果是否不可靠？是否缺乏消融实验？
- 写作表达：论文结构是否混乱？语言是否不准确？图表是否不规范？参考文献是否缺失关键工作？"""

    return f"""{_stance_law(side, stance_detail)}
你是一篇学术论文的{side_name}审稿人，你的立场是：{stance_detail}。

【待审论文内容】
{topic}

请给出你的初审意见，要求：
1. 首先用一句话概括你对这篇论文的总体评价（支持接受/建议拒稿大修）
2. 从以下四个维度逐一评审，每个维度都要引用论文中的具体内容：
{dimensions}
3. 每个维度的评审都要具体，不能空泛。说"创新性强"必须指出具体创新在哪里；说"实验有缺陷"必须指出具体缺陷是什么
4. 最后给出你的审稿结论和理由
5. 字数控制在 400-600 字
6. 结构清晰，分维度论述

直接输出初审意见内容，不要加"尊敬的编辑"之类的开场白。"""


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
    深入审稿 prompt——针对对方的审稿意见进行回应和补充
    """
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"
    opponent_name = "建议拒稿方" if side == "affirmative" else "支持接受方"

    reflection_part = ""
    if reflection:
        reflection_part = f"""
【你的自我反思（上一轮结束后你对自己审稿意见的批判——注意：批判的是审稿的全面性和深度，不是立场本身）】
{reflection}
请在本轮审稿中补充自己之前遗漏的评审点，同时更加坚定地坚持你的核心立场：{stance_detail}。"""

    return f"""{_stance_law(side, stance_detail)}
你是一篇学术论文的{side_name}审稿人，你的立场是：{stance_detail}。现在进行第 {round_num} 轮深入审稿。

【待审论文内容】
{topic}

【你自己上一轮审稿意见——先锚定己方立场】
{own_previous}

【{opponent_name}上一轮审稿意见——你的回应目标】
{opponent_previous}
{reflection_part}

【深入审稿策略——必须遵守】
1. 开头第一句必须明确重申己方审稿立场（例如："我方坚持认为这篇论文值得接受/需要大修"）
2. 针对对方提出的每一个主要评审点，从你的立场出发进行回应：
   - 如果你是支持接受方：对方指出的问题是否被夸大？是否有合理的解释？论文的优点是否足以弥补这些不足？
   - 如果你是建议拒稿方：对方强调的优点是否真的成立？是否存在被忽视的更深层问题？论文的缺陷是否足以抵消这些优点？
3. 补充你上一轮遗漏的评审点，让你的审稿意见更全面
4. 必须针对论文的具体内容（方法、实验、数据、结构）做评审，不能泛泛讨论方向

请进行深入审稿，要求：
1. 逐一回应对方上一轮提出的主要评审点，每个回应都要引用论文中的具体内容
2. 补充 1-2 个你上一轮遗漏的评审维度或具体问题
3. 用"对方指出...，但我方认为..."这样的针对性回应句式
4. 语言专业、客观，但立场明确
5. 字数 350-500 字
6. 发言中至少出现两次"我方认为"或"我方坚持"，开头一次，结尾一次，双重锚定立场
7. 发言结构：立场声明 → 回应对方 → 补充评审 → 立场确认

直接输出审稿意见内容。"""


def get_reflection_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_speeches: List[str],
    opponent_speeches: List[str],
) -> str:
    """
    审稿自我反思 prompt（Self-Reflection 核心机制）
    每轮深入审稿结束后，Agent 批判自己之前的审稿意见，找出遗漏和不足
    关键：反思的是审稿的全面性和深度，绝不能否定立场本身
    """
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"
    opponent_name = "建议拒稿方" if side == "affirmative" else "支持接受方"

    own_history = "\n---\n".join([f"第{i+1}轮：{s}" for i, s in enumerate(own_speeches)])
    opponent_history = "\n---\n".join([f"第{i+1}轮：{s}" for i, s in enumerate(opponent_speeches)])

    return f"""{_stance_law(side, stance_detail)}
你是一篇学术论文的{side_name}审稿人，你的立场是：{stance_detail}。现在暂停审稿，进行自我反思和批判。

⚠️ 最重要的原则：你的核心审稿立场始终不变，不需要反思。你要反思的是"审稿的全面性"和"评审的深度"，而不是"立场本身"。
不要说"我之前的立场有问题"或"对方是对的"，而要说"我在XX维度的评审不够深入，需要补充XX具体问题"。

🚫 反思中绝对禁止出现以下内容：
- 任何削弱己方审稿立场的表述
- 任何认同对方审稿立场的表述
- 任何质疑己方核心评审结论正确性的表述
如果你发现自己在写这些内容，立刻删掉，重新从"如何更全面地论证己方审稿立场"的角度思考。

【待审论文内容】
{topic}

【你之前的所有审稿意见】
{own_history}

【{opponent_name}之前的所有审稿意见】
{opponent_history}

请诚实地批判你自己之前的审稿意见，回答以下问题：
1. 你的审稿意见中存在哪些不够深入或不够具体的地方？哪些评审点只是泛泛而谈，没有引用论文的具体内容？
2. 哪些评审维度你还没有覆盖到？（创新性/方法论/实验可靠性/写作表达，检查是否有遗漏）
3. 对方提出的哪些评审点你还没有有效回应？下一轮应该如何从你的立场出发彻底回应？
4. 下一轮你应该如何补充和加强审稿意见，让你的立场更加不可动摇？

请输出一段 150-250 字的自我反思，要诚实、具体，不要泛泛而谈。
这个反思只有你自己能看到，不会被对方知道。
反思结束后，你将更加坚定地坚持你的审稿立场。"""


def get_rebuttal_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_speeches: List[str],
    all_opponent_speeches: List[str],
    reflection: str = None,
) -> str:
    """终审意见 prompt——系统性总结论文的优缺点"""
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"
    opponent_name = "建议拒稿方" if side == "affirmative" else "支持接受方"

    own_full = "\n---\n".join(own_speeches)
    opponent_full = "\n---\n".join(all_opponent_speeches)

    reflection_part = ""
    if reflection:
        reflection_part = f"""
【你的自我反思】
{reflection}
请在终审意见中补充自己之前遗漏的评审点，同时更加坚定地坚持你的核心立场：{stance_detail}。"""

    if side == "affirmative":
        conclusion_guide = "最后明确给出'建议接受'的结论，并说明论文的主要贡献是什么"
    else:
        conclusion_guide = "最后明确给出'建议拒稿/大修'的结论，并说明论文最致命的 2-3 个缺陷是什么"

    return f"""{_stance_law(side, stance_detail)}
你是一篇学术论文的{side_name}审稿人，你的立场是：{stance_detail}。现在给出终审意见。

【待审论文内容】
{topic}

【你方整场所有审稿意见——先锚定己方立场和核心评审点】
{own_full}

【{opponent_name}整场所有审稿意见——你的回应目标】
{opponent_full}
{reflection_part}

【终审意见策略——必须遵守】
1. 开头第一句必须明确重申己方审稿立场，给编辑一个清晰的结论锚点
2. 系统性总结论文的核心优缺点，每个点都要引用论文的具体内容
3. 针对对方整场提出的主要评审点，从你的立场出发进行最终回应
4. {conclusion_guide}

请给出终审意见，要求：
1. 总结论文的核心贡献/核心缺陷，分点列出，每个点都要具体（引用论文的方法名称、实验设置、数据结果等）
2. 逐一回应对方整场提出的主要评审点，说明为什么你的立场更合理
3. 给出明确的审稿结论（建议接受/小修/大修/拒稿）
4. 如果是支持接受方：可以列出 1-2 个论文可以改进的小问题，但必须强调这些不影响论文的整体价值
5. 如果是建议拒稿方：列出 2-3 个论文最致命的缺陷，说明为什么这些缺陷无法通过小修解决
6. 结构清晰，先总结后结论，字数 400-550 字
7. 发言中至少出现两次"我方认为"或"我方坚持"，开头一次，结尾一次，双重锚定立场
8. 结尾必须以明确的审稿结论收尾，绝不能含糊

直接输出终审意见内容。"""


def get_closing_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    own_all_speeches: List[str],
    opponent_all_speeches: List[str],
) -> str:
    """最终审稿结论 prompt"""
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"
    opponent_name = "建议拒稿方" if side == "affirmative" else "支持接受方"

    own_full = "\n---\n".join(own_all_speeches)
    opponent_full = "\n---\n".join(opponent_all_speeches)

    if side == "affirmative":
        final_verdict = "建议接受"
        summary_focus = "总结论文最核心的 2-3 个贡献，说明为什么这篇论文值得发表"
    else:
        final_verdict = "建议拒稿/大修"
        summary_focus = "总结论文最致命的 2-3 个缺陷，说明为什么这篇论文目前不适合发表"

    return f"""{_stance_law(side, stance_detail)}
你是一篇学术论文的{side_name}审稿人，你的立场是：{stance_detail}。现在给出最终审稿结论。这是你最后一次陈述审稿意见的机会。

【待审论文内容】
{topic}

【你方整场所有审稿意见】
{own_full}

【{opponent_name}整场所有审稿意见】
{opponent_full}

请给出最终审稿结论，要求：
1. 回顾整场审稿的核心交锋点，总结你方在哪些关键问题上的评审更有说服力，对方在哪些问题上的评审不够充分
2. {summary_focus}
3. 用简洁有力的语言重申你的审稿结论——结论必须明确，绝不能含糊
4. 语言专业、客观，字数 250-400 字
5. 绝对禁止"双方都有道理""这个问题没有绝对答案"等中立表述——你是给出明确审稿结论的，不是来和稀泥的
6. 最后一句话必须是明确的审稿结论："综上，我方建议{final_verdict}。"

直接输出最终审稿结论。"""


def get_summary_prompt(
    topic: str,
    side: str,
    stance_detail: str,
    speeches_to_summarize: List[str],
    existing_summary: str = None,
) -> str:
    """
    记忆摘要压缩 prompt
    把久远的审稿意见压缩成核心评审点摘要，减少 token 消耗
    """
    side_name = "支持接受方" if side == "affirmative" else "建议拒稿方"

    speeches_text = "\n---\n".join([f"第{i+1}段：{s[:500]}" for i, s in enumerate(speeches_to_summarize)])

    existing_part = ""
    if existing_summary:
        existing_part = f"""
【已有摘要（请把新审稿意见整合进去，更新这份摘要）】
{existing_summary}
"""

    return f"""你是论文审稿助手的记忆压缩模块。你代表{side_name}，立场是：{stance_detail}。

【待审论文】
{topic}
{existing_part}
【需要压缩的审稿意见】
{speeches_text}

请把以上审稿意见压缩成一份核心评审点摘要，要求：
1. 保留该方的核心评审结论、关键评审点、引用的论文具体内容（方法名称、实验设置、数据结果等）
2. 保留该方的审稿立场和核心主张
3. 去掉重复表述、过渡性语言、客套话
4. 字数控制在 200-300 字
5. 按"审稿结论→主要评审点→关键依据"的结构组织
6. 必须保持审稿立场明确，不能在摘要中出现中立或妥协表述

直接输出摘要内容，不要加其他说明。"""


def get_judge_prompt(topic: str, affirmative_full: str, negative_full: str) -> str:
    """
    综合审稿意见 prompt
    汇总支持接受方和建议拒稿方的审稿意见，给出最终综合评估
    评分维度改成审稿维度
    """
    return f"""你是一篇学术论文的领域主编（Area Chair），请汇总以下两位审稿人的意见，给出综合审稿评估。

【待审论文内容】
{topic}

===== 支持接受方审稿人整场意见 =====
{affirmative_full}

===== 建议拒稿方审稿人整场意见 =====
{negative_full}

===== 审稿评估标准（每项 0-10 分）=====
请分别评估两位审稿人的审稿质量，以及论文本身在各维度的表现：
1. 创新性评估：论文的创新点是否明确、是否显著？两位审稿人对创新性的评审是否准确、深入？
2. 方法论评估：研究方法是否合理、严谨？两位审稿人对方法论的评审是否抓住了关键问题？
3. 实验可靠性评估：实验设计是否科学、结果是否可靠？两位审稿人对实验部分的评审是否全面？
4. 写作表达评估：论文结构是否清晰、语言是否准确？两位审稿人对写作部分的评审是否到位？
5. 审稿全面性：审稿人是否覆盖了论文的主要方面？是否有重要遗漏？
6. 审稿深度：审稿意见是否深入到论文的具体内容？还是停留在泛泛而谈？
7. 审稿客观性：审稿意见是否基于论文实际内容？是否存在偏见或夸大？
8. 立场坚定性：审稿人是否始终坚持己方审稿立场？有无动摇、妥协或认同对方观点的情况？

请输出 JSON 格式，包含：
{{
  "affirmative_scores": {{"创新性评估": x, "方法论评估": x, "实验可靠性评估": x, "写作表达评估": x, "审稿全面性": x, "审稿深度": x, "审稿客观性": x, "立场坚定性": x}},
  "negative_scores": {{"创新性评估": x, "方法论评估": x, "实验可靠性评估": x, "写作表达评估": x, "审稿全面性": x, "审稿深度": x, "审稿客观性": x, "立场坚定性": x}},
  "affirmative_total": x.x,
  "negative_total": x.x,
  "winner": "affirmative" 或 "negative",
  "margin": x.x,
  "comment": "300-400字的综合审稿意见，包括：(1)论文的主要优点和贡献；(2)论文的主要缺陷和问题；(3)建议作者重点修改的方向；(4)最终给出'建议接受/小修/大修/拒稿'的综合结论。特别指出两位审稿人是否有立场动摇的情况。"
}}

只输出 JSON，不要输出其他内容。"""
