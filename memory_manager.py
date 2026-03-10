"""
自适应记忆系统 - 记忆管理器
==============================

核心组件，整合 KV-cache 监控、压缩触发、长期记忆存储和检索。

提供两种管理器:
    - AdaptiveMemoryManager: 基础管理器
    - ThinkingAwareMemoryManager: 扩展版，在 "思考" 阶段执行压缩和检索

工作流程:
    用户输入 → 检查上下文长度 →
        如果接近 3/4 容量 → 触发压缩 → 存储记忆
        → 三路混合检索相关记忆 → 构建增强上下文 → 模型推理

避免幻觉策略:
    1. 保留最近的对话轮次不压缩，确保连贯性
    2. 在系统提示中注入相关的历史记忆作为参考
    3. 使用三路混合检索（语义 + 关键词 + 文本）确保检索覆盖率
    4. 明确标注记忆来源，让模型知道这是历史信息

Usage:
    >>> from memory_manager import create_memory_manager
    >>> 
    >>> manager = create_memory_manager()
    >>> manager.set_system_prompt("你是一个有记忆的助手。")
    >>> 
    >>> # 添加对话
    >>> manager.add_user_message("我叫小明")
    >>> manager.add_assistant_message("你好小明！")
    >>> 
    >>> # 检查是否需要压缩
    >>> if manager.should_compress():
    ...     manager.perform_compression()
    >>> 
    >>> # 获取上下文信息
    >>> info = manager.get_context_info()

Author: Jiangsheng Yu
"""

from datetime import datetime
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from models import Message, Memory, MessageRole, ConversationContext
from memory_store import MemoryDatabase, get_memory_db
from compressor import ContextCompressor, CompressionResult, create_compressor
from config import Config, default_config


@dataclass
class MemoryManagerState:
    """记忆管理器状态"""
    total_compressions: int = 0
    total_tokens_compressed: int = 0
    total_memories_stored: int = 0
    last_compression_time: Optional[datetime] = None
    current_session_turns: int = 0


class AdaptiveMemoryManager:
    """
    自适应记忆管理器
    
    核心功能：
    1. 监控上下文长度，在接近容量限制时自动压缩
    2. 将压缩后的记忆存储到长期记忆数据库
    3. 根据用户输入检索相关的历史记忆
    4. 构建增强的上下文，包含工作记忆和相关长期记忆
    """
    
    def __init__(
        self,
        config: Config = None,
        tokenizer: Any = None,
        generate_fn: Callable[[str], str] = None
    ):
        """
        初始化记忆管理器
        
        Args:
            config: 配置对象
            tokenizer: 用于计算 token 数的分词器
            generate_fn: 模型生成函数，用于生成摘要
        """
        self.config = config or default_config
        self._tokenizer = tokenizer
        self._generate_fn = generate_fn
        
        # 初始化组件
        self.memory_db = get_memory_db(self.config.memory)
        self.compressor = create_compressor(
            summarize_fn=self._model_summarize,
            config=self.config.compression
        )
        
        # 当前对话上下文
        self.context = ConversationContext()
        
        # 状态追踪
        self.state = MemoryManagerState()
        
        # 回调函数
        self._on_compression_callback: Optional[Callable[[CompressionResult], None]] = None
        self._on_memory_retrieved_callback: Optional[Callable[[List[Memory]], None]] = None
    
    def set_tokenizer(self, tokenizer):
        """设置分词器"""
        self._tokenizer = tokenizer
    
    def set_generate_function(self, fn: Callable[[str], str]):
        """设置生成函数"""
        self._generate_fn = fn
        self.compressor.set_summarize_function(self._model_summarize)
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.context.system_prompt = prompt
    
    def on_compression(self, callback: Callable[[CompressionResult], None]):
        """注册压缩事件回调"""
        self._on_compression_callback = callback
    
    def on_memory_retrieved(self, callback: Callable[[List[Memory]], None]):
        """注册记忆检索事件回调"""
        self._on_memory_retrieved_callback = callback
    
    def _model_summarize(self, text: str) -> str:
        """使用模型生成摘要"""
        if self._generate_fn:
            return self._generate_fn(text)
        else:
            # 如果没有生成函数，使用简单截取
            return text[:500] + "..."
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except:
                pass
        # 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
        return len(text) // 2
    
    def count_context_tokens(self) -> int:
        """计算当前上下文的总 token 数"""
        messages = self.context.get_messages_for_model()
        total_text = ""
        for msg in messages:
            total_text += msg.get("content", "")
        return self.count_tokens(total_text)
    
    def should_compress(self) -> bool:
        """检查是否应该触发压缩"""
        current_tokens = self.count_context_tokens()
        self.context.current_token_count = current_tokens
        
        return self.compressor.should_compress(
            current_tokens,
            self.config.model.max_context_length,
            self.config.model.compression_threshold
        )
    
    def perform_compression(self) -> Optional[CompressionResult]:
        """
        执行压缩操作
        
        1. 获取可压缩的消息（保留最近几轮）
        2. 压缩为记忆
        3. 存储到数据库
        4. 从上下文中移除已压缩的消息
        """
        # 获取可压缩的消息
        compressible = self.context.get_compressible_messages(
            preserve_recent=self.config.compression.preserve_recent_turns
        )
        
        if not compressible:
            print("没有可压缩的消息")
            return None
        
        print(f"\n{'='*50}")
        print(f"触发压缩: {len(compressible)} 条消息")
        print(f"{'='*50}")
        
        # 执行压缩
        result = self.compressor.compress_messages(
            compressible,
            use_model_summarize=(self._generate_fn is not None)
        )
        
        if result.success and result.memory:
            # 存储到数据库
            self.memory_db.save_memory(result.memory)
            
            # 从上下文中移除已压缩的消息
            self.context.remove_compressed_messages(len(compressible))
            
            # 更新状态
            self.state.total_compressions += 1
            self.state.total_tokens_compressed += result.original_token_count
            self.state.total_memories_stored += 1
            self.state.last_compression_time = datetime.now()
            
            print(f"✓ 压缩完成:")
            print(f"  - 摘要: {result.memory.summary[:100]}...")
            print(f"  - 主题词: {', '.join(result.memory.keywords)}")
            print(f"  - 原始 token 数: {result.original_token_count}")
            print(f"  - 压缩后 token 数: {result.compressed_token_count}")
            
            # 触发回调
            if self._on_compression_callback:
                self._on_compression_callback(result)
        else:
            print(f"✗ 压缩失败: {result.error_message}")
        
        return result
    
    def retrieve_relevant_memories(self, query: str) -> List[Memory]:
        """
        根据查询检索相关的长期记忆
        
        这些记忆会被添加到上下文中，帮助模型回答问题
        """
        memories = self.memory_db.retrieve_relevant_memories(
            query,
            limit=self.config.memory.max_retrieved_memories
        )
        
        if memories:
            print(f"\n检索到 {len(memories)} 条相关记忆")
            for mem in memories:
                print(f"  - [{mem.timestamp.strftime('%m-%d %H:%M')}] {mem.summary[:50]}...")
        
        # 激活这些记忆
        self.context.activate_memories(memories)
        
        # 触发回调
        if self._on_memory_retrieved_callback and memories:
            self._on_memory_retrieved_callback(memories)
        
        return memories
    
    def add_user_message(self, content: str) -> Message:
        """
        添加用户消息
        
        1. 检索相关记忆
        2. 添加消息到上下文
        3. 检查是否需要压缩
        """
        # 检索相关记忆
        self.retrieve_relevant_memories(content)
        
        # 添加消息
        msg = self.context.add_message(MessageRole.USER, content)
        self.state.current_session_turns += 1
        
        # 检查是否需要压缩
        if self.should_compress():
            self.perform_compression()
        
        return msg
    
    def add_assistant_message(self, content: str) -> Message:
        """添加助手消息"""
        msg = self.context.add_message(MessageRole.ASSISTANT, content)
        
        # 检查是否需要压缩
        if self.should_compress():
            self.perform_compression()
        
        return msg
    
    def get_messages_for_model(self) -> List[Dict[str, str]]:
        """获取发送给模型的消息"""
        return self.context.get_messages_for_model()
    
    def get_context_info(self) -> Dict[str, Any]:
        """获取当前上下文信息"""
        current_tokens = self.count_context_tokens()
        max_tokens = self.config.model.max_context_length
        threshold = self.config.model.compression_trigger_tokens
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "compression_threshold": threshold,
            "usage_percentage": round(current_tokens / max_tokens * 100, 2),
            "will_compress_at": threshold,
            "message_count": len(self.context.messages),
            "active_memories": len(self.context.active_memories),
            "total_compressions": self.state.total_compressions,
            "total_memories_stored": self.state.total_memories_stored
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        db_stats = self.memory_db.get_statistics()
        return {
            **db_stats,
            "session_stats": {
                "total_compressions": self.state.total_compressions,
                "total_tokens_compressed": self.state.total_tokens_compressed,
                "current_session_turns": self.state.current_session_turns,
                "last_compression": self.state.last_compression_time.isoformat() 
                    if self.state.last_compression_time else None
            }
        }
    
    def clear_context(self, keep_system_prompt: bool = True):
        """清空当前上下文"""
        system_prompt = self.context.system_prompt if keep_system_prompt else None
        self.context = ConversationContext()
        self.context.system_prompt = system_prompt
        self.state.current_session_turns = 0
        print("上下文已清空")
    
    def export_session(self) -> Dict[str, Any]:
        """导出当前会话"""
        return {
            "context": self.context.to_dict(),
            "state": {
                "total_compressions": self.state.total_compressions,
                "total_tokens_compressed": self.state.total_tokens_compressed,
                "current_session_turns": self.state.current_session_turns
            },
            "config": {
                "model_name": self.config.model.model_name,
                "max_context_length": self.config.model.max_context_length,
                "compression_threshold": self.config.model.compression_threshold
            }
        }
    
    def import_session(self, session_data: Dict[str, Any]):
        """导入会话"""
        if "context" in session_data:
            self.context = ConversationContext.from_dict(session_data["context"])
        if "state" in session_data:
            state = session_data["state"]
            self.state.total_compressions = state.get("total_compressions", 0)
            self.state.total_tokens_compressed = state.get("total_tokens_compressed", 0)
            self.state.current_session_turns = state.get("current_session_turns", 0)


class ThinkingAwareMemoryManager(AdaptiveMemoryManager):
    """
    思考感知记忆管理器
    
    扩展自适应记忆管理器，支持在"思考"过程中进行压缩
    这模拟了用户要求的"模型在思考过程中对旧上下文进行压缩"
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thinking_mode = False
        self._thinking_buffer: List[str] = []
    
    def enter_thinking_mode(self):
        """进入思考模式"""
        self._thinking_mode = True
        self._thinking_buffer = []
        print("\n[进入思考模式]")
    
    def exit_thinking_mode(self) -> str:
        """退出思考模式，返回思考内容"""
        self._thinking_mode = False
        thinking_content = "\n".join(self._thinking_buffer)
        self._thinking_buffer = []
        print("[退出思考模式]\n")
        return thinking_content
    
    def think(self, thought: str):
        """记录思考内容"""
        if self._thinking_mode:
            self._thinking_buffer.append(thought)
            print(f"  💭 {thought[:100]}...")
    
    def process_with_thinking(self, user_input: str) -> str:
        """
        带思考的处理流程
        
        1. 检查是否需要压缩
        2. 如果需要，在"思考"阶段执行压缩
        3. 检索相关记忆
        4. 返回增强的上下文
        """
        self.enter_thinking_mode()
        
        # 检查上下文长度
        current_tokens = self.count_context_tokens()
        threshold = self.config.model.compression_trigger_tokens
        
        self.think(f"当前上下文: {current_tokens} tokens, 阈值: {threshold} tokens")
        
        # 如果需要压缩
        if current_tokens >= threshold:
            self.think("上下文接近容量限制，需要进行压缩...")
            result = self.perform_compression()
            if result and result.success:
                self.think(f"压缩完成，生成记忆: {result.memory.summary[:50]}...")
        
        # 检索相关记忆
        self.think(f"检索与 '{user_input[:30]}...' 相关的历史记忆...")
        memories = self.retrieve_relevant_memories(user_input)
        if memories:
            self.think(f"找到 {len(memories)} 条相关记忆")
        else:
            self.think("没有找到相关记忆")
        
        thinking_log = self.exit_thinking_mode()
        
        # 添加用户消息（不再重复检索记忆）
        self.context.add_message(MessageRole.USER, user_input)
        self.state.current_session_turns += 1
        
        return thinking_log


# 便捷函数
def create_memory_manager(
    config: Config = None,
    tokenizer = None,
    generate_fn: Callable[[str], str] = None,
    thinking_aware: bool = True
) -> AdaptiveMemoryManager:
    """
    创建记忆管理器实例
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        generate_fn: 生成函数
        thinking_aware: 是否使用思考感知管理器
    
    Returns:
        AdaptiveMemoryManager 实例
    """
    if thinking_aware:
        manager = ThinkingAwareMemoryManager(config, tokenizer, generate_fn)
    else:
        manager = AdaptiveMemoryManager(config, tokenizer, generate_fn)
    
    return manager
