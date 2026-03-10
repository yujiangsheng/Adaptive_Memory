"""
Adaptive Memory System — 自适应记忆系统
==========================================

为大语言模型提供的自适应记忆管理框架。
通过 *智能压缩* 与 *三路混合检索* 突破上下文长度限制，同时减少幻觉。

核心原理::

    KV-cache (短期记忆) → 压缩 → 长期记忆 (SQLite)

    当上下文 token 数 >= max_context_length × 0.75 时，自动将旧对话
    压缩为记忆  M = (T, S, Keywords)，存入数据库供后续检索。

模块一览:
    models           数据模型 — Memory / Message / ConversationContext
    memory_store     SQLite 长期记忆持久化 — 语义 + 关键词 + 文本三路混合检索
    compressor       上下文压缩器 (C → M)  — jieba TF-IDF + LLM 摘要
    memory_manager   自适应记忆管理器 — 监控 · 压缩 · 检索 · 注入
    inference_engine LLM 推理引擎 — Qwen2.5-7B-Instruct + Mock 引擎 (30 种事实模式)
    demo             CLI 演示 · 多轮对话 · 记忆评估 · 随机语料生成
    web_server       Web UI — FastAPI + SSE 流式前端 + 评估 API

Quick Start::

    # Python API
    from inference_engine import create_engine
    engine = create_engine(mock=True)        # 模拟模式，无需 GPU
    engine.chat("你好，我是小明")
    engine.chat("你还记得我叫什么吗？")

    # Web 界面
    python web_server.py --mock              # http://localhost:7860

    # CLI 评估
    python demo.py -n 20                     # 20 轮对话 + 记忆评估

Author:  Jiangsheng Yu
License: MIT
"""

__title__ = "Adaptive Memory System"
__version__ = "1.0.0"
__author__ = "Jiangsheng Yu"
__license__ = "MIT"
from .config import Config, ModelConfig, MemoryConfig, CompressionConfig, default_config

__all__ = [
    # 数据模型
    "Memory",
    "Message",
    "MessageRole",
    "ConversationContext",
    
    # 存储
    "MemoryDatabase",
    "get_memory_db",
    
    # 压缩
    "ContextCompressor",
    "CompressionResult",
    "create_compressor",
    
    # 记忆管理
    "AdaptiveMemoryManager",
    "ThinkingAwareMemoryManager",
    "create_memory_manager",
    
    # 推理引擎
    "QwenInferenceEngine",
    "MockInferenceEngine",
    "create_engine",
    
    # 配置
    "Config",
    "ModelConfig",
    "MemoryConfig",
    "CompressionConfig",
    "default_config",
]
