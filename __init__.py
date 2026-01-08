"""
Adaptive Memory System - 自适应记忆系统
===========================================

一个用于大语言模型的自适应记忆管理框架，通过智能压缩和检索机制
突破上下文长度限制，同时避免幻觉。

核心原理:
    KV-cache (短期记忆) → 压缩 → 长期记忆 (数据库)
    
    当上下文 C 接近模型容量的 3/4 时，自动压缩为:
    M = (T, S, Keywords)
    - T: 时间戳
    - S: 简短摘要  
    - Keywords: 主题词列表

主要组件:
    - models: 数据模型 (Memory, Message, ConversationContext)
    - memory_store: SQLite 长期记忆数据库
    - compressor: 上下文压缩器 (C → M)
    - memory_manager: 自适应记忆管理器
    - inference_engine: LLM 推理引擎 (Qwen2.5)

Quick Start:
    >>> from adaptive_memory import create_engine
    >>> engine = create_engine(mock=True)  # 测试模式
    >>> response = engine.chat("你好，我是小明")
    >>> response = engine.chat("你还记得我叫什么吗？")

Author:
    Jiangsheng Yu

License:
    MIT License
"""

__title__ = "Adaptive Memory System"
__version__ = "0.1.0"
__author__ = "Jiangsheng Yu"
__license__ = "MIT"

from .models import Memory, Message, MessageRole, ConversationContext
from .memory_store import MemoryDatabase, get_memory_db
from .compressor import ContextCompressor, CompressionResult, create_compressor
from .memory_manager import (
    AdaptiveMemoryManager,
    ThinkingAwareMemoryManager,
    create_memory_manager
)
from .inference_engine import QwenInferenceEngine, create_engine
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
    "create_engine",
    
    # 配置
    "Config",
    "ModelConfig",
    "MemoryConfig",
    "CompressionConfig",
    "default_config",
]
