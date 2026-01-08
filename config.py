"""
自适应记忆系统 - 配置模块
==========================

定义系统的所有可配置参数，包括模型、记忆和压缩相关配置。

核心参数:
    - compression_threshold: 触发压缩的阈值 (默认 0.75，即 3/4)
    - max_context_length: 模型最大上下文长度
    - preserve_recent_turns: 压缩时保留的最近对话轮数

Usage:
    >>> from config import Config, default_config
    >>> 
    >>> # 使用默认配置
    >>> config = default_config
    >>> print(config.model.compression_trigger_tokens)  # 24576
    >>> 
    >>> # 自定义配置
    >>> config = Config()
    >>> config.model.max_context_length = 16384
    >>> config.model.compression_threshold = 0.8

Author: Jiangsheng Yu
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型路径或名称
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Qwen2.5-7B-Instruct 的最大上下文长度
    # 官方支持 32768 tokens，但实际推荐使用更短的长度以保证质量
    max_context_length: int = 32768
    
    # 触发压缩的阈值 (3/4 = 0.75)
    compression_threshold: float = 0.75
    
    # 计算触发压缩的 token 数
    @property
    def compression_trigger_tokens(self) -> int:
        return int(self.max_context_length * self.compression_threshold)
    
    # 最大生成长度
    max_new_tokens: int = 2048
    
    # 设备配置
    # auto 模式下的优先级: GPU (cuda) > MPS (Apple Silicon) > CPU
    device: str = "auto"  # "cuda", "mps", "cpu", "auto"
    
    # 量化配置 (节省显存)
    load_in_4bit: bool = True
    load_in_8bit: bool = False


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    # 数据库路径
    db_path: Path = Path("./memory_store/long_term_memory.db")
    
    # 摘要的最大长度 (tokens)
    summary_max_tokens: int = 256
    
    # 最大主题词数量
    max_keywords: int = 5
    
    # 检索相关记忆时的最大数量
    max_retrieved_memories: int = 5
    
    # 是否使用语义搜索进行记忆检索
    use_semantic_search: bool = True
    
    # 语义搜索的嵌入模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # 记忆相关性阈值
    relevance_threshold: float = 0.6


@dataclass
class CompressionConfig:
    """压缩配置"""
    # 压缩时保留的最近对话轮数 (不压缩)
    preserve_recent_turns: int = 2
    
    # 系统提示词是否参与压缩
    compress_system_prompt: bool = False
    
    # 压缩提示词模板
    compression_prompt_template: str = """请将以下对话内容压缩为简洁的摘要，保留关键信息：

{content}

请按以下格式输出：
摘要：[简洁描述对话的核心内容，100字以内]
主题词：[用逗号分隔的3-5个关键词]"""


@dataclass
class Config:
    """全局配置"""
    model: ModelConfig = None
    memory: MemoryConfig = None
    compression: CompressionConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.compression is None:
            self.compression = CompressionConfig()
        
        # 确保数据库目录存在
        self.memory.db_path.parent.mkdir(parents=True, exist_ok=True)


# 默认配置实例
default_config = Config()
