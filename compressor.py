"""
自适应记忆系统 - 上下文压缩器
==============================

将上下文 C 压缩为记忆 M = (T, S, Keywords):
    - T: 时间戳
    - S: 简短摘要
    - Keywords: 主题词列表

压缩策略:
    1. 检测上下文长度是否接近阈值 (3/4)
    2. 使用模型生成摘要
    3. 提取关键词/主题词 (jieba TF-IDF)
    4. 生成 Memory 对象

Usage:
    >>> from compressor import create_compressor
    >>> from models import Message, MessageRole
    >>> 
    >>> compressor = create_compressor()
    >>> 
    >>> messages = [
    ...     Message(MessageRole.USER, "你好，我叫小明"),
    ...     Message(MessageRole.ASSISTANT, "你好小明！")
    ... ]
    >>> 
    >>> result = compressor.compress_messages(messages)
    >>> if result.success:
    ...     print(f"摘要: {result.memory.summary}")
    ...     print(f"主题词: {result.memory.keywords}")

Author: Jiangsheng Yu
"""

import re
import jieba
import jieba.analyse
from datetime import datetime
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from models import Message, Memory, MessageRole, ConversationContext
from config import CompressionConfig, default_config


@dataclass
class CompressionResult:
    """压缩结果"""
    memory: Memory
    compressed_message_count: int
    original_token_count: int
    compressed_token_count: int
    success: bool
    error_message: Optional[str] = None


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(
        self, 
        config: CompressionConfig = None,
        summarize_fn: Callable[[str], str] = None
    ):
        """
        初始化压缩器
        
        Args:
            config: 压缩配置
            summarize_fn: 自定义摘要函数，接收文本返回摘要
                         如果不提供，将使用默认的基于模型的摘要
        """
        self.config = config or default_config.compression
        self._summarize_fn = summarize_fn
        
        # 初始化 jieba 用于中文关键词提取
        jieba.initialize()
    
    def set_summarize_function(self, fn: Callable[[str], str]):
        """设置摘要生成函数"""
        self._summarize_fn = fn
    
    def _messages_to_text(self, messages: List[Message]) -> str:
        """将消息列表转换为文本"""
        lines = []
        for msg in messages:
            role_name = {
                MessageRole.SYSTEM: "系统",
                MessageRole.USER: "用户",
                MessageRole.ASSISTANT: "助手"
            }.get(msg.role, str(msg.role))
            lines.append(f"{role_name}: {msg.content}")
        return "\n\n".join(lines)
    
    def _detect_language(self, text: str) -> str:
        """检测文本主要语言"""
        # 简单的中英文检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "zh"
        else:
            return "en"
    
    def _detect_sentence_end(self, text: str) -> bool:
        """检测文本是否以正确的句号结尾"""
        text = text.strip()
        if not text:
            return False
        
        lang = self._detect_language(text)
        if lang == "zh":
            return text.endswith("。") or text.endswith("！") or text.endswith("？")
        else:
            return text.endswith(".") or text.endswith("!") or text.endswith("?")
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取文本的关键词/主题词
        
        使用 jieba 的 TF-IDF 算法提取关键词
        """
        # 使用 jieba 提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
        
        if not keywords:
            # 备用：使用 TextRank 算法
            keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False)
        
        return list(keywords)
    
    def _default_summarize(self, text: str) -> str:
        """
        默认的摘要方法（简单截取）
        
        实际使用时应该通过 set_summarize_function 设置使用模型的摘要方法
        """
        # 这是一个简化版本，实际应使用LLM生成摘要
        sentences = re.split(r'[。.!！?？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            summary = text[:500]
        else:
            # 取首尾句子作为简单摘要
            summary = f"{sentences[0]}...{sentences[-1]}"
        
        # 确保以正确的标点结尾
        lang = self._detect_language(text)
        if lang == "zh" and not summary.endswith("。"):
            summary = summary.rstrip(".,。") + "。"
        elif lang == "en" and not summary.endswith("."):
            summary = summary.rstrip(".,。") + "."
        
        return summary[:500]  # 限制长度
    
    def summarize(self, text: str) -> str:
        """
        生成摘要
        
        如果设置了自定义摘要函数，使用它；否则使用默认方法
        """
        if self._summarize_fn:
            return self._summarize_fn(text)
        else:
            return self._default_summarize(text)
    
    def parse_model_compression_output(self, output: str) -> Tuple[str, List[str]]:
        """
        解析模型输出的压缩结果
        
        期望格式：
        摘要：[内容]
        主题词：[词1, 词2, 词3]
        """
        summary = ""
        keywords = []
        
        # 提取摘要
        summary_match = re.search(r'摘要[：:]\s*(.+?)(?=主题词|关键词|$)', output, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
            summary = summary.strip('[]【】')
        
        # 提取主题词
        keywords_match = re.search(r'(?:主题词|关键词)[：:]\s*(.+?)$', output, re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            keywords_text = keywords_text.strip('[]【】')
            # 分割关键词（支持逗号、顿号、空格分隔）
            keywords = re.split(r'[,，、\s]+', keywords_text)
            keywords = [k.strip() for k in keywords if k.strip()]
        
        # 如果解析失败，使用整个输出作为摘要
        if not summary:
            summary = output[:300]
        
        return summary, keywords
    
    def compress_messages(
        self, 
        messages: List[Message],
        use_model_summarize: bool = True
    ) -> CompressionResult:
        """
        压缩消息列表为记忆
        
        Args:
            messages: 要压缩的消息列表
            use_model_summarize: 是否使用模型进行摘要（需要先设置summarize_fn）
        
        Returns:
            CompressionResult: 压缩结果
        """
        if not messages:
            return CompressionResult(
                memory=None,
                compressed_message_count=0,
                original_token_count=0,
                compressed_token_count=0,
                success=False,
                error_message="没有可压缩的消息"
            )
        
        try:
            # 转换为文本
            text = self._messages_to_text(messages)
            original_length = len(text)
            
            # 获取时间范围
            start_time = messages[0].timestamp
            end_time = messages[-1].timestamp
            
            # 生成摘要
            if use_model_summarize and self._summarize_fn:
                # 使用模型生成摘要
                prompt = self.config.compression_prompt_template.format(content=text)
                model_output = self._summarize_fn(prompt)
                summary, keywords = self.parse_model_compression_output(model_output)
            else:
                # 使用默认方法
                summary = self._default_summarize(text)
                keywords = self.extract_keywords(text, self.config.max_keywords if hasattr(self.config, 'max_keywords') else 5)
            
            # 确保有关键词
            if not keywords:
                keywords = self.extract_keywords(text, 5)
            
            # 创建记忆对象
            memory = Memory(
                timestamp=datetime.now(),
                summary=summary,
                keywords=keywords[:5],  # 限制最多5个关键词
                original_start_time=start_time,
                original_end_time=end_time,
                original_turn_count=len(messages)
            )
            
            return CompressionResult(
                memory=memory,
                compressed_message_count=len(messages),
                original_token_count=original_length,
                compressed_token_count=len(summary),
                success=True
            )
            
        except Exception as e:
            return CompressionResult(
                memory=None,
                compressed_message_count=0,
                original_token_count=0,
                compressed_token_count=0,
                success=False,
                error_message=str(e)
            )
    
    def should_compress(
        self, 
        current_tokens: int, 
        max_tokens: int, 
        threshold: float = 0.75
    ) -> bool:
        """
        判断是否应该触发压缩
        
        Args:
            current_tokens: 当前上下文的 token 数
            max_tokens: 模型最大上下文长度
            threshold: 触发压缩的阈值（默认 3/4 = 0.75）
        
        Returns:
            bool: 是否应该压缩
        """
        return current_tokens >= max_tokens * threshold
    
    def get_compression_prompt(self, messages: List[Message]) -> str:
        """获取用于让模型压缩的提示词"""
        text = self._messages_to_text(messages)
        return self.config.compression_prompt_template.format(content=text)


# ─── 工厂函数 ────────────────────────────────────────────────

def create_compressor(
    summarize_fn: Callable[[str], str] = None,
    config: CompressionConfig = None,
) -> ContextCompressor:
    """创建压缩器实例。

    Args:
        summarize_fn: 可选的自定义摘要函数（通常由 LLM 提供）。
        config:       压缩配置，缺省使用 ``default_config.compression``。

    Returns:
        ContextCompressor 实例。
    """
    return ContextCompressor(config=config, summarize_fn=summarize_fn)
