#!/usr/bin/env python3
"""
自适应记忆系统 — CLI 演示与记忆评估
======================================

提供命令行入口来体验自适应记忆系统：

    python demo.py                        # 默认 6 轮模拟演示
    python demo.py -n 20                  # 指定 20 轮长对话 + 记忆评估
    python demo.py -n 50 --eval           # 50 轮 + 独立评估报告
    python demo.py --compression          # 演示压缩机制
    python demo.py -i                     # 模拟引擎交互
    python demo.py --real -i              # 加载真实 Qwen 模型 + 交互
    python demo.py --model <path> -i      # 指定模型路径 + 交互

Web 界面（推荐）::

    python web_server.py --mock           # http://localhost:7860

语料生成策略:
    - n ≤ 30:  从 30 种预设事实模式中随机抽取，每次取值随机化
    - n > 30:  30 条常规对话 + (n-30) 条 "编号N对应X" 格式的备忘对话
    - 每条 Turn 包含 message, fact_key, fact_value, recall_query

Author: Jiangsheng Yu
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import Config
from inference_engine import create_engine, QwenInferenceEngine


# ─── 横幅 ──────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           自适应记忆系统  ·  Adaptive Memory System           ║
║                                                              ║
║   KV-cache → 压缩 → 长期记忆     by Jiangsheng Yu           ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─── 长对话语料库 ──────────────────────────────────────────────
# 每条包含一个 "事实" (fact_key → fact_value)，用于后续记忆评估。
# 对话按主题分组，轮数不够时循环使用。

@dataclass
class Turn:
    """一轮对话，包含用户消息和嵌入的关键事实。"""
    message: str
    fact_key: str          # 事实标签，如 "name"
    fact_value: str        # 期望记住的值，如 "小明"
    recall_query: str      # 用于回忆测试的提问

def _generate_batch() -> List[Turn]:
    """生成一批 30 条随机对话。每条包含一个可验证事实，所有取值随机化。"""
    import random

    name = random.choice(["小明", "小华", "小刚", "小杰", "小龙"])
    age = random.randint(22, 35)

    company = random.choice(["字节跳动", "腾讯", "阿里巴巴", "百度", "华为", "美团", "京东"])

    specialty_msg, specialty_val = random.choice([
        ("分布式系统和微服务架构", "分布式系统"),
        ("高并发系统设计和性能优化", "高并发"),
        ("容器化技术和云原生架构",  "容器化"),
    ])

    pet_name = random.choice(["豆豆", "旺财", "小白", "团团", "毛毛"])
    pet_type = random.choice(["柴犬", "猫", "狗"])
    pet_age  = random.randint(1, 8)

    email = random.choice([
        "xiaoming@example.com", "xiaohua@test.com", "dev_xiao@mail.com",
    ])

    tech_lang = random.choice(["Go", "Java", "Rust", "Python"])
    tech_pair = {"Go": "Kubernetes", "Java": "Spring Cloud",
                 "Rust": "Tokio", "Python": "FastAPI"}

    learning_opts = [l for l in ["Rust", "Kotlin", "Swift", "Scala"]
                     if l != tech_lang]
    learning_lang = random.choice(learning_opts)

    db    = random.choice(["MySQL", "PostgreSQL", "MongoDB"])
    cache = random.choice(["Redis", "Memcached"])

    ai_tech = random.choice(["RAG", "RLHF", "Agent", "多模态"])
    ai_desc = {"RAG": "检索增强生成", "RLHF": "人类反馈强化学习",
               "Agent": "智能体", "多模态": "多模态理解"}

    mq = random.choice(["Kafka", "RabbitMQ", "Pulsar"])

    hobby_msg, hobby_val = random.choice([
        ("骑公路自行车，周末经常骑行100公里",  "骑公路自行车"),
        ("跑步，每周跑三次半程马拉松",        "跑步"),
        ("游泳，坚持每天游1500米",           "游泳"),
        ("爬山，周末经常去周边山区",           "爬山"),
        ("打篮球，周末和朋友约场",            "打篮球"),
    ])

    book = random.choice(["黑客与画家", "代码大全", "人月神话", "设计模式"])
    book_author = {"黑客与画家": "Paul Graham", "代码大全": "McConnell",
                   "人月神话": "Brooks", "设计模式": "GoF"}

    uni, dept, grad_year = random.choice([
        ("上海交通大学", "计算机系",       "2022"),
        ("清华大学",    "自动化系",        "2021"),
        ("浙江大学",    "信息工程系",       "2020"),
        ("北京大学",    "信息科学技术学院",  "2023"),
        ("复旦大学",    "计算机学院",       "2022"),
    ])

    conf = random.choice(["KubeCon", "GopherCon", "PyCon", "QCon"])

    routine_val = random.choice([
        "早上6点起床跑步", "早上7点起床健身", "早上5点半起床冥想",
    ])
    routine_years = random.choice(["一年", "两年", "三年"])

    research = random.choice(["eBPF", "WASM", "量子计算"])
    research_app = {"eBPF": "Kubernetes 的网络可观测性",
                    "WASM": "边缘计算应用", "量子计算": "密码学安全"}

    team_size = random.randint(8, 25)
    role = random.choice(["Tech Lead", "架构师"])

    framework = random.choice(["LangChain", "LlamaIndex", "Haystack"])

    github_id = random.choice(["xiaoming-dev", "xiaohua-dev", "xiao-dev"])

    paper = random.choice(["FlashAttention-2", "Mamba", "RWKV"])

    cloud = random.choice(["AWS", "Azure", "GCP", "阿里云"])

    district = random.choice(["海淀区", "朝阳区"])

    coffee = random.choice(["耶加雪菲", "曼特宁", "蓝山", "瑰夏"])

    prev_opts = [l for l in ["Java", "C++", "PHP", "Ruby"] if l != tech_lang]
    prev_lang = random.choice(prev_opts) if prev_opts else "Java"
    prev_years = random.choice(["2", "3", "4", "5"])

    mesh = random.choice(["Istio", "Linkerd", "Envoy"])

    gf_name = random.choice(["小芳", "小丽", "小雪", "小燕"])
    gf_job  = random.choice(["产品经理", "设计师", "前端工程师"])

    latency_from = random.choice(["200ms", "150ms", "300ms"])
    latency_to   = random.choice(["50ms", "30ms", "20ms"])

    future_opts = [l for l in ["Zig", "Rust", "C"] if l != tech_lang]
    future_lang = random.choice(future_opts) if future_opts else "Zig"

    dish = random.choice(["红烧牛腩", "糖醋排骨", "宫保鸡丁", "麻婆豆腐"])

    return [
        Turn(f"你好！我叫{name}，今年{age}岁，是一名软件工程师。",
             "name", name, "你还记得我叫什么名字吗？"),
        Turn(f"我在北京{company}工作，主要负责后端开发。",
             "company", company, "我在哪家公司工作？"),
        Turn(f"我的专长是{specialty_msg}。",
             "specialty", specialty_val, "我的技术专长是什么？"),
        Turn(f"我养了一只叫{pet_name}的{pet_type}，已经{pet_age}岁了。",
             "pet", pet_name, "我的宠物叫什么？"),
        Turn(f"我的邮箱是{email}，方便以后联系。",
             "email", email, "我的邮箱是什么？"),
        Turn(f"我最近在学习 {learning_lang} 语言，觉得它的设计理念很有趣。",
             "learning_lang", learning_lang, "我最近在学什么编程语言？"),
        Turn(f"我们团队用 {tech_lang} 和 {tech_pair[tech_lang]} 搭建了一套微服务平台。",
             "tech_stack", tech_lang, "我们团队用什么语言搭建微服务平台？"),
        Turn(f"数据库方面，我们主要用 {db} 做主库，{cache} 做缓存。",
             "database", db, "我们的主数据库是什么？"),
        Turn(f"我对 {ai_tech}（{ai_desc[ai_tech]}）技术特别感兴趣，想深入研究。",
             "interest_tech", ai_tech, "我对什么AI技术感兴趣？"),
        Turn(f"我们团队最近完成了一个基于 {mq} 的实时数据管道项目。",
             "project", mq, "我们最近的项目用了什么消息队列？"),
        Turn(f"我喜欢{hobby_msg}。",
             "hobby", hobby_val, "我的爱好是什么？"),
        Turn(f"我最喜欢的书是《{book}》，{book_author[book]} 的经典之作。",
             "favorite_book", book, "我最喜欢的书是哪本？"),
        Turn(f"我是{uni}{dept}毕业的，{grad_year}届。",
             "university", uni, "我是哪个大学毕业的？"),
        Turn(f"我正在准备明年去美国参加 {conf} 大会。",
             "conference", conf, "我计划参加什么技术大会？"),
        Turn(f"我每天{routine_val}，已经坚持了{routine_years}。",
             "routine", routine_val, "我的晨间习惯是什么？"),
        Turn("我认为 WebAssembly 会成为下一代云计算的核心技术。",
             "opinion_tech", "WebAssembly", "我认为什么技术会成为云计算核心？"),
        Turn(f"我在研究 {research} 技术，用它来做{research_app[research]}。",
             "research", research, "我在研究什么技术？"),
        Turn(f"我们部门有{team_size}个人，我是后端组的 {role}。",
             "role", role, "我在团队里是什么角色？"),
        Turn(f"最近我用 {framework} 搭建了一个内部知识库问答系统。",
             "side_project", framework, "我用什么框架搭建知识库？"),
        Turn(f"我的 GitHub ID 是 {github_id}，上面有我的开源项目。",
             "github", github_id, "我的 GitHub ID 是什么？"),
        Turn(f"我上周读了一篇关于 Transformer 架构优化的论文，讲的是 {paper}。",
             "paper", paper, "我最近读了关于什么技术的论文？"),
        Turn(f"我们公司准备从自建机房迁移到 {cloud} 云上，预计半年完成。",
             "cloud_plan", cloud, "我们公司准备迁移到哪个云平台？"),
        Turn(f"我家住在北京{district}，离公司骑车20分钟。",
             "location", district, "我住在北京哪个区？"),
        Turn(f"我喜欢喝手冲咖啡，最常用的豆子是{coffee}。",
             "coffee", coffee, "我最常喝什么咖啡？"),
        Turn(f"我之前做过{prev_years}年 {prev_lang} 开发，后来转到 Go 生态。",
             "prev_lang", prev_lang, "我之前用什么语言开发？"),
        Turn(f"我参与了公司的 Service Mesh 落地，用的是 {mesh}。",
             "service_mesh", mesh, "我们公司用什么 Service Mesh 方案？"),
        Turn(f"我的女朋友叫{gf_name}，是做{gf_job}的。",
             "girlfriend", gf_name, "我女朋友叫什么？"),
        Turn(f"我们团队下季度的目标是把系统延迟从{latency_from}降到{latency_to}以下。",
             "team_goal", latency_to, "我们团队的延迟优化目标是多少？"),
        Turn(f"我正在考虑用 {future_lang} 语言重写核心模块来提升性能。",
             "future_lang", future_lang, "我考虑用什么语言重写核心模块？"),
        Turn(f"我最近迷上了做饭，拿手菜是{dish}。",
             "cooking", dish, "我的拿手菜是什么？"),
    ]


# ── 备忘类对话词库（用于 n > 30 时生成唯一事实） ──────────────

MEMO_WORDS = [
    "苹果", "香蕉", "橘子", "葡萄", "西瓜", "草莓", "芒果", "桃子", "樱桃", "柠檬",
    "大海", "森林", "沙漠", "雪山", "草原", "湖泊", "河流", "瀑布", "山谷", "冰川",
    "星星", "月亮", "太阳", "彩虹", "云朵", "闪电", "北风", "细雨", "晚霞", "朝露",
    "老鹰", "海豚", "熊猫", "老虎", "白鸽", "蝴蝶", "蜻蜓", "鲸鱼", "孔雀", "松鼠",
    "钻石", "翡翠", "珍珠", "琥珀", "水晶", "玛瑙", "翠竹", "红枫", "银杏", "梅花",
    "荷花", "兰花", "菊花", "牡丹", "杜鹃", "茉莉", "丁香", "桂花", "百合", "玫瑰",
]


def _generate_memo_turns(count: int) -> List[Turn]:
    """生成 count 条备忘类对话，格式为 "编号N对应X"，每条有唯一 fact_key。"""
    import random

    values: List[str] = []
    while len(values) < count:
        pool = list(MEMO_WORDS)
        random.shuffle(pool)
        values.extend(pool)
    values = values[:count]

    turns = []
    for i, value in enumerate(values):
        idx = i + 1
        turns.append(Turn(
            f"请记住，编号{idx}对应的是{value}。",
            f"memo_{idx}", value,
            f"编号{idx}对应的是什么？"
        ))
    return turns


def generate_random_pool(n: int = 30) -> List[Turn]:
    """
    生成恰好 n 条随机对话语料，每条有唯一 fact_key。

    - n ≤ 30: 从一批 30 种预设事实模式中随机抽取 n 条（每次取值随机化）
    - n > 30: 30 条常规对话 + (n-30) 条 "编号N对应X" 格式的备忘对话

    Args:
        n: 需要的对话轮数（1 ~ 20000）

    Returns:
        恰好 n 条 Turn 对象的列表
    """
    import random

    batch = _generate_batch()
    if n <= len(batch):
        random.shuffle(batch)
        return batch[:n]

    # n > 30: 30 条常规对话 + 备忘类对话补充
    random.shuffle(batch)
    extra = _generate_memo_turns(n - len(batch))
    return batch + extra


# 保留一个固定语料池用于向后兼容（如 eval/pool API 预览）
CONVERSATION_POOL: List[Turn] = generate_random_pool()


# ─── 记忆评估器 ────────────────────────────────────────────────

@dataclass
class RecallResult:
    """单条回忆测试结果。"""
    fact_key: str
    fact_value: str
    query: str
    response: str
    recalled: bool         # 模型是否成功回忆
    turn_index: int        # 事实首次出现的对话轮次

@dataclass
class EvalReport:
    """记忆评估报告。"""
    total_facts: int = 0
    recalled: int = 0
    forgotten: int = 0
    results: List[RecallResult] = field(default_factory=list)
    total_compressions: int = 0
    total_memories_stored: int = 0

    @property
    def recall_rate(self) -> float:
        return self.recalled / self.total_facts * 100 if self.total_facts else 0.0

    def print_report(self):
        """打印格式化评估报告。"""
        print("\n" + "═" * 60)
        print("📋 记忆效果评估报告")
        print("═" * 60)
        print(f"  对话总轮数     : {self.total_facts} 轮（含事实）")
        print(f"  压缩触发次数   : {self.total_compressions}")
        print(f"  存储记忆条数   : {self.total_memories_stored}")
        print(f"  回忆成功       : {self.recalled} / {self.total_facts}")
        print(f"  回忆失败       : {self.forgotten} / {self.total_facts}")
        print(f"  记忆召回率     : {self.recall_rate:.1f}%")
        print("─" * 60)

        # 按是否召回分组展示
        if self.results:
            print("\n  ✅ 成功回忆:")
            success = [r for r in self.results if r.recalled]
            if success:
                for r in success:
                    print(f"     轮{r.turn_index:>2d} [{r.fact_key}] "
                          f"期望「{r.fact_value}」→ ✓")
            else:
                print("     （无）")

            print("\n  ❌ 未能回忆:")
            failed = [r for r in self.results if not r.recalled]
            if failed:
                for r in failed:
                    print(f"     轮{r.turn_index:>2d} [{r.fact_key}] "
                          f"期望「{r.fact_value}」→ ✗")
                    print(f"          回复: {r.response[:80]}...")
            else:
                print("     （无）")

        # 按对话距离分析衰减
        if len(self.results) >= 5:
            print("\n  📉 记忆衰减分析（按事实出现的轮次远近）:")
            early = [r for r in self.results if r.turn_index <= len(self.results) // 3]
            mid   = [r for r in self.results
                     if len(self.results) // 3 < r.turn_index <= 2 * len(self.results) // 3]
            late  = [r for r in self.results if r.turn_index > 2 * len(self.results) // 3]
            for label, group in [("前期", early), ("中期", mid), ("近期", late)]:
                if group:
                    rate = sum(1 for r in group if r.recalled) / len(group) * 100
                    print(f"     {label}（{len(group)}条）: 召回率 {rate:.0f}%")

        print("═" * 60)


def evaluate_memory(engine: QwenInferenceEngine,
                    facts: List[Dict]) -> EvalReport:
    """
    对引擎执行记忆回忆测试（只读，不修改对话上下文）。

    对每个已植入的事实进行检索 + 提问，判断模型回复中是否包含期望值。
    跳过 engine.chat() 以避免评估查询触发压缩，污染后续检索。

    Args:
        engine: 已完成对话的推理引擎
        facts: 列表，每项 {"key", "value", "query", "turn_index"}

    Returns:
        EvalReport 评估报告（含召回率和衰减分析）
    """
    report = EvalReport()

    # 获取系统统计
    ctx = engine.get_context_info()
    report.total_compressions = ctx.get("total_compressions", 0)
    report.total_memories_stored = ctx.get("total_memories_stored", 0)

    print("\n" + "=" * 60)
    print("🧪 开始记忆评估（逐条回忆测试）")
    print("=" * 60)

    for fact in facts:
        query = fact["query"]
        expected = fact["value"]

        # 只读评估：检索记忆 → 构建 prompt → 直接生成，不修改上下文
        memories = engine.memory_manager.memory_db.retrieve_relevant_memories(
            query,
            limit=engine.config.memory.max_retrieved_memories,
        )

        # 构建 system prompt（含记忆注入）
        system_content = engine.DEFAULT_SYSTEM_PROMPT
        if memories:
            memory_lines = []
            for mem in memories:
                ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                memory_lines.append(f"  [{ts}] {mem.summary}")
            system_content += (
                "\n\n## 相关历史记忆\n" + "\n".join(memory_lines)
            )

        # 把当前上下文里的对话消息也带上，保持一致性
        context_msgs = []
        for msg in engine.memory_manager.context.messages:
            context_msgs.append({"role": msg.role.value, "content": msg.content})

        messages = (
            [{"role": "system", "content": system_content}]
            + context_msgs
            + [{"role": "user", "content": query}]
        )

        response = engine._generate_response(messages)

        # 判断回忆是否成功
        recalled = expected.lower() in response.lower()

        result = RecallResult(
            fact_key=fact["key"],
            fact_value=expected,
            query=query,
            response=response,
            recalled=recalled,
            turn_index=fact["turn_index"],
        )
        report.results.append(result)
        report.total_facts += 1
        if recalled:
            report.recalled += 1
        else:
            report.forgotten += 1

        status = "✅" if recalled else "❌"
        print(f"  {status} [{fact['key']}] "
              f"期望「{expected}」— {'命中' if recalled else '未命中'}")

    return report


# ─── 演示：多轮对话 + 评估 ─────────────────────────────────────

def demo_rounds(rounds: int, run_eval: bool = True) -> QwenInferenceEngine:
    """
    执行指定轮数的多轮对话，随后进行记忆效果评估。

    Args:
        rounds: 对话轮数
        run_eval: 对话结束后是否自动运行评估
    """
    import os, tempfile
    from pathlib import Path

    print("\n" + "=" * 60)
    print(f"🔬 多轮对话演示（{rounds} 轮）")
    print("=" * 60)

    # 使用临时数据库，避免与之前运行的数据混淆
    tmp_db = Path(tempfile.gettempdir()) / f"adaptive_mem_demo_{os.getpid()}.db"

    config = Config()
    config.memory.db_path = tmp_db
    # 降低检索阈值，让更多记忆被召回（默认 0.6 对短查询过于严格）
    config.memory.relevance_threshold = 0.3
    # 长对话增加检索数量，避免有用记忆被排挤
    if rounds > 20:
        config.memory.max_retrieved_memories = 20

    if rounds > 10:
        config.model.max_context_length = 800
        config.model.compression_threshold = 0.5
        print(f"   上下文窗口: {config.model.max_context_length} tokens "
              f"(压缩阈值 {config.model.compression_threshold})")
    print(f"   检索阈值: {config.memory.relevance_threshold}")

    engine = create_engine(mock=True, config=config)

    # 按指定轮数生成随机语料（每次内容不同）
    turns = generate_random_pool(rounds)

    # 记录每轮的事实，用于评估
    planted_facts: List[Dict] = []

    for idx, turn in enumerate(turns, 1):
        info_pre = engine.get_context_info()
        response = engine.chat(turn.message, verbose=False)
        info_post = engine.get_context_info()

        compressed = info_post["total_compressions"] > info_pre.get("total_compressions", 0)
        marker = " ⚡压缩" if compressed else ""

        print(f"\n  [{idx:>3d}/{rounds}] 用户: {turn.message[:45]}...")
        print(f"          tokens: {info_pre['current_tokens']} → "
              f"{info_post['current_tokens']}{marker}")

        planted_facts.append({
            "key": turn.fact_key,
            "value": turn.fact_value,
            "query": turn.recall_query,
            "turn_index": idx,
        })

    # 对话结束统计
    stats = engine.get_context_info()
    print("\n" + "─" * 60)
    print("📊 对话结束统计")
    print("─" * 60)
    print(f"  总轮数       : {rounds}")
    print(f"  当前 tokens  : {stats['current_tokens']}")
    print(f"  压缩次数     : {stats['total_compressions']}")
    print(f"  存储记忆     : {stats['total_memories_stored']}")
    print(f"  活跃记忆     : {stats['active_memories']}")

    # 执行记忆评估
    if run_eval and planted_facts:
        report = evaluate_memory(engine, planted_facts)
        report.print_report()

    # 清理临时数据库
    try:
        os.unlink(tmp_db)
    except OSError:
        pass

    return engine


# ─── 演示：压缩机制 ─────────────────────────────────────────────

def demo_compression() -> QwenInferenceEngine:
    """降低上下文阈值以快速触发压缩，演示 C → M 流程。"""
    print("\n" + "=" * 60)
    print("🗜️  压缩机制演示")
    print("=" * 60)

    config = Config()
    config.model.max_context_length = 1000
    config.model.compression_threshold = 0.5

    engine = create_engine(mock=True, config=config)

    messages = [
        "我想了解人工智能的历史。AI 是计算机科学的一个重要分支，致力于创建能够执行需要人类智能的任务的系统。",
        "AI 的概念最早可以追溯到古希腊神话中的自动机，但作为学科始于 1956 年达特茅斯会议。",
        "早期的研究集中在符号推理和问题求解。1960 年代出现了第一批专家系统。",
        "1980 年代专家系统繁荣，随后 'AI 寒冬' 让研究陷入低谷。2010 年代深度学习才让 AI 重新崛起。",
        "现在请告诉我，我们之前讨论了什么内容？",
    ]

    for msg in messages:
        info_pre = engine.get_context_info()
        response = engine.chat(msg, verbose=False)
        info_post = engine.get_context_info()

        print(f"\n用户: {msg[:50]}...")
        print(f"   tokens: {info_pre['current_tokens']} → {info_post['current_tokens']}")
        print(f"   压缩次数: {info_post['total_compressions']}")
        if info_post["total_compressions"] > info_pre.get("total_compressions", 0):
            print("   ⚡ 触发了压缩！旧对话已转化为长期记忆")
        print(f"助手: {response[:80]}...")

    return engine


# ─── 交互模式 ──────────────────────────────────────────────────

def interactive(engine: QwenInferenceEngine):
    """命令行交互循环。

    特殊命令:
        quit / exit  — 退出
        stats        — 打印记忆统计
        info         — 打印上下文状态
        clear        — 清空当前上下文
        memories     — 列出所有长期记忆
        eval         — 运行记忆评估（仅在 demo_rounds 后有效）
    """
    print("\n" + "=" * 60)
    print("💬 交互模式")
    print("=" * 60)
    print("  quit/exit  退出 | stats  统计 | info  上下文")
    print("  clear  清空上下文 | memories  查看记忆")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit"):
            print("再见！")
            break
        if cmd == "stats":
            print(json.dumps(engine.get_memory_stats(), indent=2, ensure_ascii=False, default=str))
            continue
        if cmd == "info":
            print(json.dumps(engine.get_context_info(), indent=2, ensure_ascii=False))
            continue
        if cmd == "clear":
            engine.clear_context()
            continue
        if cmd == "memories":
            if engine.memory_manager:
                for mem in engine.memory_manager.memory_db.get_all_memories():
                    print(f"  [{mem.timestamp.strftime('%Y-%m-%d %H:%M')}] {mem.summary[:60]}...")
            continue

        engine.chat(user_input)


# ─── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="自适应记忆系统 — CLI 演示")
    parser.add_argument("--real", action="store_true", help="加载真实 Qwen 模型")
    parser.add_argument("--model", type=str, help="模型路径（覆盖默认配置）")
    parser.add_argument("--compression", action="store_true", help="演示压缩机制")
    parser.add_argument("-n", "--rounds", type=int, default=0,
                        help="指定对话轮数（>0 时执行多轮测试 + 评估）")
    parser.add_argument("--eval", action="store_true",
                        help="对话结束后运行记忆评估（与 -n 配合）")
    parser.add_argument("-i", "--interactive", action="store_true", help="进入交互模式")
    args = parser.parse_args()

    print(BANNER)

    if args.real or args.model:
        # ── 真实模型 ──────────────────────────────
        config = Config()
        if args.model:
            config.model.model_name = args.model

        print(f"加载模型: {config.model.model_name}")
        try:
            engine = create_engine(config=config, mock=False)
        except Exception as exc:
            print(f"\n❌ 模型加载失败: {exc}")
            print("请确保: 1) 已安装依赖  2) 有足够显存  3) 模型路径正确")
            return

        interactive(engine)
    else:
        # ── 模拟模式 ──────────────────────────────
        if args.rounds > 0:
            engine = demo_rounds(args.rounds, run_eval=args.eval or args.rounds >= 6)
        elif args.compression:
            engine = demo_compression()
        else:
            # 默认 6 轮 + 自动评估
            engine = demo_rounds(6, run_eval=True)

        if args.interactive:
            interactive(engine)

    print("\n演示完成！")


if __name__ == "__main__":
    main()
