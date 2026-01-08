# Adaptive Memory System - 自适应记忆系统

> **Author:** Jiangsheng Yu  
> **License:** MIT  
> **Version:** 0.1.0

一个用于大语言模型的自适应记忆管理框架，通过智能压缩和检索机制突破上下文长度限制，同时避免幻觉。

## 核心思想

在 LLM 中，KV-cache 是短期记忆。当上下文 C 接近模型能力极限的 **3/4** 时，系统自动将旧上下文压缩为记忆 M：

```
M = (T, S, Keywords)
```

| 符号 | 含义 | 说明 |
|------|------|------|
| T | 时间戳 | 记忆创建时间 |
| S | 摘要 | 对话内容的简洁描述 |
| Keywords | 主题词 | 3-5个关键词 |

压缩后的记忆存储到 SQLite 数据库，作为长期记忆随时可检索。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       用户输入                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 AdaptiveMemoryManager                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. 监控上下文长度 (token count)                       │  │
│  │  2. tokens >= max_length × 0.75 → 触发压缩            │  │
│  │  3. 语义检索相关的长期记忆                             │  │
│  │  4. 构建增强上下文 (当前对话 + 相关记忆)               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ Compressor │  │ MemoryDB   │  │ LLM        │
   │            │  │ (SQLite)   │  │ (Qwen2.5)  │
   │  C → M     │  │ 长期记忆    │  │ 推理生成   │
   └────────────┘  └────────────┘  └────────────┘
```

## 文件结构

```
Adaptive_Memory/
├── __init__.py           # 包入口，导出所有公共接口
├── config.py             # 配置模块
├── models.py             # 数据模型 (Memory, Message)
├── memory_store.py       # SQLite 长期记忆存储
├── compressor.py         # 上下文压缩器 (C → M)
├── memory_manager.py     # 自适应记忆管理器
├── inference_engine.py   # LLM 推理引擎
├── demo.py               # 演示脚本
├── requirements.txt      # Python 依赖
└── README.md             # 说明文档
```

## 快速开始

### 安装

```bash
cd Adaptive_Memory
pip install -r requirements.txt
```

### 模拟模式 (无需 GPU)

```bash
python demo.py
```

### 真实模型

```bash
# 使用 Qwen2.5-7B-Instruct (需要 GPU)
python demo.py --real

# 指定模型路径
python demo.py --model /path/to/Qwen2.5-7B-Instruct

# 交互模式
python demo.py --real -i
```

## 编程接口

### 基本用法

```python
from adaptive_memory import create_engine

# 创建引擎 (测试模式)
engine = create_engine(mock=True)

# 对话
response = engine.chat("你好！我叫小明。")
response = engine.chat("我在学习机器学习。")
response = engine.chat("你还记得我叫什么吗？")  # 会检索记忆

# 查看统计
print(engine.get_memory_stats())
```

### 自定义配置

```python
from adaptive_memory import create_engine, Config

config = Config()
config.model.model_name = "Qwen/Qwen2.5-7B-Instruct"
config.model.max_context_length = 32768
config.model.compression_threshold = 0.75  # 3/4 触发压缩

engine = create_engine(config=config, mock=False)
```

### 单独使用记忆管理器

```python
from adaptive_memory import create_memory_manager

manager = create_memory_manager()
manager.set_system_prompt("你是一个有记忆能力的助手。")

# 添加对话
manager.add_user_message("我叫小明")
manager.add_assistant_message("你好小明！")

# 检查压缩
if manager.should_compress():
    result = manager.perform_compression()
    print(f"压缩完成: {result.memory.summary}")

# 检索记忆
memories = manager.retrieve_relevant_memories("小明是谁？")
```

## 避免幻觉的策略

| 策略 | 说明 |
|------|------|
| 明确标注来源 | 历史记忆带时间戳标注 `[记忆 01-08 10:30]` |
| 保留近期对话 | 最近 2 轮对话不压缩，确保连贯性 |
| 语义相关性 | 只检索与当前话题相关的记忆 |
| 系统提示引导 | 指导模型区分当前对话和历史信息 |
| 不确定性表达 | 模型在不确定时主动表明 |

## 配置参数

### ModelConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | 模型路径 |
| `max_context_length` | `32768` | 最大上下文 tokens |
| `compression_threshold` | `0.75` | 压缩触发阈值 |
| `max_new_tokens` | `2048` | 最大生成长度 |
| `device` | `auto` | 设备 (cuda > mps > cpu) |
| `load_in_4bit` | `True` | 4-bit 量化 |

### MemoryConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `db_path` | `./memory_store/long_term_memory.db` | 数据库路径 |
| `max_keywords` | `5` | 最大主题词数 |
| `max_retrieved_memories` | `5` | 检索记忆数量 |
| `use_semantic_search` | `True` | 启用语义搜索 |
| `relevance_threshold` | `0.6` | 相关性阈值 |

### CompressionConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `preserve_recent_turns` | `2` | 保留最近对话轮数 |
| `compress_system_prompt` | `False` | 是否压缩系统提示 |

## 技术细节

### 压缩触发条件

```
current_tokens >= max_context_length × compression_threshold
```

Qwen2.5-7B-Instruct 示例:
```
32768 × 0.75 = 24576 tokens → 触发压缩
```

### 压缩流程

1. 获取可压缩消息 (排除最近 N 轮)
2. 使用模型生成摘要和主题词
3. 创建 Memory 对象
4. 计算嵌入向量 (语义搜索)
5. 存储到 SQLite
6. 从上下文移除已压缩消息

### 记忆检索

1. 计算用户输入的嵌入向量
2. 余弦相似度搜索
3. 过滤低相关性结果
4. 注入系统提示

## 限制与未来

### 当前限制

- 压缩有损，可能丢失细节
- 语义搜索依赖嵌入模型质量
- 长期记忆需定期清理

### 计划改进

- [ ] 分层记忆 (短期/工作/长期)
- [ ] 记忆遗忘机制
- [ ] 多模态记忆支持
- [ ] 分布式存储

## License

MIT License
