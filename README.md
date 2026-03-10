# Adaptive Memory System — 自适应记忆系统

> **Author:** Jiangsheng Yu &ensp;|&ensp; **License:** MIT &ensp;|&ensp; **Version:** 1.0.0

为 **Qwen2.5-7B-Instruct** 等大语言模型提供的自适应记忆管理框架。  
在超长对话中，系统自动压缩旧上下文为 **长期记忆** 并持久化到 SQLite，后续对话通过 **三路混合检索**（语义 + 关键词 + 文本直匹配）调取相关记忆，从而：

- ✅ **突破上下文窗口限制** — 对话永不中断
- ✅ **减少幻觉** — 记忆带时间戳标注，模型区分当前 vs 历史
- ✅ **即开即用** — 提供 Web UI (FastAPI + SSE) 和 CLI 两种使用方式
- ✅ **自动化评估** — 内置记忆召回率测试，支持 Web 可视化评估

---

## 核心原理

```
用户输入
   ↓
┌─────────────────── AdaptiveMemoryManager ───────────────────┐
│ 1. 计算上下文 token 数                                       │
│ 2. tokens ≥ max_context_length × 0.75 → 触发压缩            │
│ 3. 旧对话 C → 记忆 M = (T, S, Keywords) → 存入 SQLite       │
│ 4. 三路混合检索（语义 + 关键词 + 文本）→ 注入系统提示词       │
│ 5. 构建增强上下文 → LLM 推理                                 │
└─────────────────────────────────────────────────────────────┘
```

| 符号 | 含义 | 示例 |
|------|------|------|
| **T** | 时间戳 | `2026-03-09 14:30` |
| **S** | 摘要 | `"用户自我介绍为软件工程师，讨论了微服务架构。"` |
| **Keywords** | 主题词 | `["微服务", "Spring Cloud", "Kubernetes"]` |

---

## 项目结构

```
Adaptive_Memory/
├── config.py             # 全局配置 (模型·记忆·压缩)
├── models.py             # 数据模型 — Memory / Message / ConversationContext
├── compressor.py         # 上下文压缩器 (C → M)，jieba TF-IDF 关键词提取
├── memory_store.py       # 长期记忆持久化 — SQLite + 三路混合检索 (语义/关键词/文本)
├── memory_manager.py     # 自适应记忆管理器 — 监控·压缩·检索·注入
├── inference_engine.py   # LLM 推理引擎 — Qwen2.5 集成 + Mock 引擎（含 30 种事实模式）
├── demo.py               # CLI 演示·多轮对话·记忆评估·随机语料生成
├── web_server.py         # Web 后端 — FastAPI + SSE 流式对话 + 评估 API
├── static/index.html     # Web 前端 — 聊天 / 上下文仪表盘 / 记忆面板 / 评估面板
├── __init__.py           # 包元信息 & 公共导出
├── requirements.txt      # Python 依赖
└── memory_store/         # SQLite 数据库文件 (自动创建)
```

---

## 快速开始

### 1. 安装依赖

```bash
cd Adaptive_Memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Web 界面（推荐）

```bash
# 模拟模式 — 无需 GPU，立即体验
python web_server.py --mock

# 真实模型
python web_server.py

# 自定义
python web_server.py --model /path/to/Qwen2.5-7B-Instruct --port 8080
```

打开浏览器访问 **http://localhost:7860**，即可看到：

| 区域 | 功能 |
|------|------|
| 聊天区 | 流式对话、思考过程折叠、压缩通知 |
| 上下文仪表盘 | 实时 token 用量，颜色随阈值变化 |
| 记忆库面板 | 查看 / 删除所有长期记忆 |
| 记忆测试面板 | 预览语料、重新生成、运行评估（含进度条和衰减分析） |
| 统计面板 | 压缩次数、session 轮次、token 统计 |

### 3. CLI 演示

```bash
python demo.py                       # 默认 6 轮模拟对话 + 评估
python demo.py -n 20                 # 20 轮长对话 + 记忆评估
python demo.py -n 50 --eval          # 50 轮 + 独立评估报告
python demo.py --compression         # 演示压缩机制
python demo.py -i                    # 模拟引擎交互模式
python demo.py --real -i             # 加载真实 Qwen 模型 + 交互
python demo.py --model <path> -i     # 指定模型路径 + 交互
```

### 4. Python API

```python
from inference_engine import create_engine

# 模拟模式（测试）
engine = create_engine(mock=True)
engine.chat("你好！我叫小明。")
engine.chat("我是一名软件工程师。")
engine.chat("你还记得我叫什么吗？")  # 系统自动检索记忆

# 查看统计
print(engine.get_memory_stats())
print(engine.get_context_info())
```

```python
# 单独使用记忆管理器
from memory_manager import create_memory_manager

manager = create_memory_manager()
manager.set_system_prompt("你是一个有记忆能力的助手。")
manager.add_user_message("我叫小明")
manager.add_assistant_message("你好小明！")

if manager.should_compress():
    result = manager.perform_compression()
    print(f"摘要: {result.memory.summary}")
    print(f"主题词: {result.memory.keywords}")

memories = manager.retrieve_relevant_memories("小明是谁？")
```

```python
# 自定义配置
from config import Config

config = Config()
config.model.model_name = "Qwen/Qwen2.5-7B-Instruct"
config.model.max_context_length = 32768
config.model.compression_threshold = 0.75  # 75% 触发压缩
config.memory.use_semantic_search = True
config.compression.preserve_recent_turns = 2

engine = create_engine(config=config)
```

```python
# 使用随机语料池生成评估数据
from demo import generate_random_pool, evaluate_memory

pool = generate_random_pool(50)  # 生成 50 条随机语料
# pool 中每条 Turn 包含 message, fact_key, fact_value, recall_query
```

---

## Web API 参考

### 对话 & 状态

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/chat` | POST | SSE 流式对话 — `{"message": "..."}` |
| `/api/context` | GET | 上下文状态 (token 用量、使用率) |
| `/api/memories` | GET | 所有长期记忆列表 |
| `/api/memories/{id}` | DELETE | 删除指定记忆 |
| `/api/stats` | GET | 记忆 & 会话统计 |
| `/api/clear` | POST | 清空对话上下文 |
| `/api/session` | GET | 导出当前会话 |
| `/api/system-prompt` | POST | 设置系统提示词 — `{"prompt": "..."}` |

### 记忆评估

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/eval/pool?n=30` | GET | 随机生成 n 条对话语料预览（每次不同） |
| `/api/eval/run` | POST | 执行多轮对话 + 记忆评估 — SSE 流式进度 — `{"rounds": 20}` |

### SSE 事件类型 — `/api/chat`

| 事件 | 数据 | 说明 |
|------|------|------|
| `thinking` | `{log, compressed, active_memories}` | 思考日志、是否触发压缩、活跃记忆 |
| `token` | `{text}` | 流式 token |
| `context` | `{current_tokens, max_tokens, ...}` | 上下文状态快照 |
| `done` | `{response}` | 完成信号 |
| `error` | `{message}` | 错误信息 |

### SSE 事件类型 — `/api/eval/run`

| 事件 | 数据 | 说明 |
|------|------|------|
| `turn` | `{index, total, message, compressed, ...}` | 一轮对话完成 |
| `eval` | `{fact_key, fact_value, recalled, progress, ...}` | 一条回忆测试完成 |
| `report` | `{recall_rate, results, decay, ...}` | 最终评估报告（含衰减分析） |
| `error` | `{message}` | 错误信息 |

---

## 记忆评估系统

系统内置了完整的记忆召回率评估框架，用于量化衡量记忆系统的效果。

### 评估流程

1. **生成语料** — 随机生成 n 条对话，每条嵌入一个可验证的事实（如姓名、公司、爱好等）
2. **执行对话** — 依次发送所有对话给引擎，触发自然的压缩和记忆存储
3. **回忆测试** — 对每个事实进行只读检索+提问，检查模型是否能正确回忆
4. **衰减分析** — 按事实出现的时间远近（前期/中期/近期）分析召回率衰减

### 语料类型

| 范围 | 类型 | 说明 |
|------|------|------|
| n ≤ 30 | 常规对话 | 30 种预设事实模式（姓名、公司、宠物、邮箱、技术栈等），每次随机取值 |
| n > 30 | 常规 + 备忘 | 前 30 条为常规对话，超出部分为 "编号N对应X" 格式的备忘对话 |

### Web 评估面板

侧边栏「记忆测试」标签页提供三个操作按钮：

- **预览语料** — 查看已缓存的语料（不重新生成）
- **重新生成** — 调用 API 随机生成新语料
- **开始测试** — 运行完整的对话 + 评估流程，实时显示进度和结果

---

## 配置参考

### ModelConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | 模型路径 |
| `max_context_length` | `32768` | 最大上下文 token 数 |
| `compression_threshold` | `0.75` | 触发压缩的阈值 |
| `max_new_tokens` | `2048` | 单次最大生成长度 |
| `device` | `auto` | 设备优先级: cuda > mps > cpu |
| `load_in_4bit` | `True` | 4-bit 量化（仅 CUDA） |

### MemoryConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `db_path` | `./memory_store/long_term_memory.db` | 数据库路径 |
| `max_keywords` | `5` | 每条记忆最大主题词数 |
| `max_retrieved_memories` | `5` | 每次检索返回的记忆数量 |
| `use_semantic_search` | `True` | 启用语义搜索 (bge 嵌入) |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `relevance_threshold` | `0.6` | 语义相关性阈值 |

### CompressionConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `preserve_recent_turns` | `2` | 压缩时保留最近几轮不压缩 |
| `compress_system_prompt` | `False` | 是否压缩系统提示词 |

---

## 避免幻觉策略

| 策略 | 实现方式 |
|------|----------|
| **时间戳标注** | 每条记忆含 `[记忆 03-09 14:30]` 前缀 |
| **保留近期上下文** | 最近 2 轮对话不压缩，保证连贯性 |
| **语义相关性过滤** | 只注入与当前话题相关的记忆（余弦 ≥ 0.6） |
| **三路混合检索** | 语义 + 关键词 + 文本直匹配并行执行，合并去重 |
| **系统提示引导** | 提示模型区分"当前对话"与"历史记忆" |
| **不确定性表达** | 引导模型在不确定时主动声明 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Qwen2.5-7B-Instruct (transformers) |
| 分词/关键词 | jieba TF-IDF |
| 语义搜索 | sentence-transformers (bge-small-zh-v1.5) |
| 持久化 | SQLAlchemy + SQLite |
| Web 后端 | FastAPI + Uvicorn + SSE |
| Web 前端 | 原生 HTML/CSS/JS（零依赖） |

### 压缩流程

1. 获取可压缩消息（排除最近 N 轮）
2. 使用 LLM 生成摘要和主题词（Mock 模式使用正则提取）
3. 创建 Memory 对象 M = (T, S, Keywords)
4. 计算嵌入向量（bge-small-zh-v1.5）
5. 存储到 SQLite
6. 从上下文移除已压缩消息

### 记忆检索（三路并行）

1. **语义搜索** — 计算查询嵌入向量，余弦相似度 ≥ 阈值的记忆入选
2. **关键词搜索** — jieba 分词后与记忆主题词做交集匹配
3. **文本直匹配** — 关键词在记忆摘要文本中直接搜索
4. 三路结果合并去重，语义优先，截取前 N 条注入系统提示

---

## 限制与未来

### 当前限制

- 压缩有损，可能丢失细节
- 语义搜索依赖嵌入模型质量
- 长期记忆需定期清理

### 计划改进

- [ ] 分层记忆（短期 / 工作 / 长期）
- [ ] 记忆遗忘机制
- [ ] 多模态记忆支持
- [ ] 分布式存储

---

## License

MIT License
