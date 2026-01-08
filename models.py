"""
自适应记忆系统 - 数据模型
==========================

定义系统核心数据结构:

    Memory: 压缩后的记忆单元
        M = (T, S, Keywords)
        - T: 时间戳
        - S: 简短摘要
        - Keywords: 主题词列表
    
    Message: 单条对话消息
        包含角色 (user/assistant/system) 和内容
    
    ConversationContext: 对话上下文
        管理当前会话的所有消息和活跃记忆

Usage:
    >>> from models import Memory, Message, MessageRole
    >>> 
    >>> # 创建消息
    >>> msg = Message(role=MessageRole.USER, content="你好")
    >>> 
    >>> # 创建记忆
    >>> from datetime import datetime
    >>> mem = Memory(
    ...     timestamp=datetime.now(),
    ...     summary="用户自我介绍，名叫小明。",
    ...     keywords=["自我介绍", "小明"]
    ... )

Author: Jiangsheng Yu
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """单条对话消息"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )
    
    def __len__(self) -> int:
        """返回内容长度（用于简单估算）"""
        return len(self.content)


@dataclass
class Memory:
    """
    压缩后的记忆单元 M
    
    M = (T, S, Keywords) 其中:
    - T: 时间戳
    - S: 简短摘要
    - Keywords: 主题词列表
    """
    # 时间戳 T
    timestamp: datetime
    
    # 摘要 S
    summary: str
    
    # 主题词列表
    keywords: List[str]
    
    # 原始对话的起止时间
    original_start_time: datetime = None
    original_end_time: datetime = None
    
    # 原始对话轮数
    original_turn_count: int = 0
    
    # 唯一标识符
    id: str = None
    
    # 嵌入向量 (用于语义搜索)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.id is None:
            # 基于内容生成唯一ID
            content_hash = hashlib.md5(
                f"{self.timestamp.isoformat()}{self.summary}".encode()
            ).hexdigest()[:16]
            self.id = f"mem_{content_hash}"
        
        if self.original_start_time is None:
            self.original_start_time = self.timestamp
        if self.original_end_time is None:
            self.original_end_time = self.timestamp
    
    def to_dict(self) -> dict:
        """转换为字典，用于存储"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "keywords": self.keywords,
            "original_start_time": self.original_start_time.isoformat(),
            "original_end_time": self.original_end_time.isoformat(),
            "original_turn_count": self.original_turn_count,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """从字典创建实例"""
        return cls(
            id=data.get("id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            summary=data["summary"],
            keywords=data["keywords"],
            original_start_time=datetime.fromisoformat(data.get("original_start_time", data["timestamp"])),
            original_end_time=datetime.fromisoformat(data.get("original_end_time", data["timestamp"])),
            original_turn_count=data.get("original_turn_count", 0),
            embedding=data.get("embedding")
        )
    
    def to_context_string(self) -> str:
        """转换为可插入上下文的字符串格式"""
        keywords_str = ", ".join(self.keywords)
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"[记忆 {time_str}] {self.summary} (主题: {keywords_str})"
    
    def __repr__(self) -> str:
        return f"Memory(id={self.id}, summary={self.summary[:50]}..., keywords={self.keywords})"


@dataclass
class ConversationContext:
    """
    对话上下文管理
    
    管理当前的 KV-cache (短期记忆) 和相关的长期记忆
    """
    # 系统提示词
    system_prompt: Optional[str] = None
    
    # 当前对话消息列表 (短期记忆 / KV-cache 内容)
    messages: List[Message] = field(default_factory=list)
    
    # 已激活的长期记忆 (从数据库检索出的相关记忆)
    active_memories: List[Memory] = field(default_factory=list)
    
    # 当前 token 计数 (需要由 tokenizer 更新)
    current_token_count: int = 0
    
    # 会话ID
    session_id: str = None
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: MessageRole, content: str) -> Message:
        """添加新消息"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def get_messages_for_model(self) -> List[dict]:
        """获取发送给模型的消息格式"""
        result = []
        
        # 添加系统提示词
        if self.system_prompt:
            # 如果有活跃的记忆，将其注入系统提示词
            if self.active_memories:
                memories_context = "\n\n## 相关历史记忆\n"
                for mem in self.active_memories:
                    memories_context += f"- {mem.to_context_string()}\n"
                enhanced_system = self.system_prompt + memories_context
            else:
                enhanced_system = self.system_prompt
            
            result.append({"role": "system", "content": enhanced_system})
        
        # 添加对话消息
        for msg in self.messages:
            result.append({"role": msg.role.value, "content": msg.content})
        
        return result
    
    def get_compressible_messages(self, preserve_recent: int = 2) -> List[Message]:
        """
        获取可以被压缩的消息
        
        保留最近的 preserve_recent 轮对话不压缩
        """
        if len(self.messages) <= preserve_recent * 2:
            return []
        
        # 保留最近的消息
        cutoff = len(self.messages) - preserve_recent * 2
        return self.messages[:cutoff]
    
    def remove_compressed_messages(self, count: int):
        """移除已被压缩的消息"""
        self.messages = self.messages[count:]
    
    def clear_messages(self):
        """清空消息"""
        self.messages = []
        self.current_token_count = 0
    
    def activate_memories(self, memories: List[Memory]):
        """激活检索到的记忆"""
        self.active_memories = memories
    
    def to_dict(self) -> dict:
        """序列化"""
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages],
            "active_memories": [mem.to_dict() for mem in self.active_memories],
            "current_token_count": self.current_token_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """反序列化"""
        ctx = cls(
            session_id=data.get("session_id"),
            system_prompt=data.get("system_prompt"),
            current_token_count=data.get("current_token_count", 0)
        )
        ctx.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx.active_memories = [Memory.from_dict(m) for m in data.get("active_memories", [])]
        return ctx
