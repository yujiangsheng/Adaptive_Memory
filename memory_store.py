"""
自适应记忆系统 - 长期记忆存储
==============================

使用 SQLite 实现记忆的持久化存储，支持:
    - CRUD 操作 (创建、读取、更新、删除)
    - 三路混合检索 (语义 + 关键词 + 文本直匹配，并行执行后合并去重)
    - 嵌入向量持久化 (bge-small-zh-v1.5)

检索策略 (retrieve_relevant_memories):
    1. 语义搜索 — 查询向量与记忆嵌入的余弦相似度 ≥ 阈值
    2. 关键词搜索 — jieba 分词后与记忆主题词做交集
    3. 文本直匹配 — 关键词在摘要中直接出现
    三路结果合并去重，语义优先，截取前 limit 条返回。

数据库结构:
    memories 表:
        - id: 唯一标识符
        - timestamp: 记忆时间戳
        - summary: 摘要文本
        - keywords: 主题词 (JSON)
        - embedding: 嵌入向量 (二进制)

Usage:
    >>> from memory_store import get_memory_db
    >>> from models import Memory
    >>> from datetime import datetime
    >>> 
    >>> # 获取数据库实例
    >>> db = get_memory_db()
    >>> 
    >>> # 保存记忆
    >>> mem = Memory(
    ...     timestamp=datetime.now(),
    ...     summary="用户学习Python编程。",
    ...     keywords=["Python", "编程"]
    ... )
    >>> db.save_memory(mem)
    >>> 
    >>> # 检索相关记忆 (三路混合)
    >>> results = db.retrieve_relevant_memories("编程语言")

Author: Jiangsheng Yu
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from models import Memory
from config import MemoryConfig, default_config

Base = declarative_base()


class MemoryRecord(Base):
    """数据库记忆记录"""
    __tablename__ = "memories"
    
    id = Column(String(32), primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    keywords = Column(Text, nullable=False)  # JSON 存储
    original_start_time = Column(DateTime)
    original_end_time = Column(DateTime)
    original_turn_count = Column(Integer, default=0)
    embedding = Column(LargeBinary)  # 二进制存储嵌入向量
    created_at = Column(DateTime, default=datetime.now)
    
    def to_memory(self) -> Memory:
        """转换为 Memory 对象"""
        embedding = None
        if self.embedding:
            embedding = np.frombuffer(self.embedding, dtype=np.float32).tolist()
        
        return Memory(
            id=self.id,
            timestamp=self.timestamp,
            summary=self.summary,
            keywords=json.loads(self.keywords),
            original_start_time=self.original_start_time,
            original_end_time=self.original_end_time,
            original_turn_count=self.original_turn_count,
            embedding=embedding
        )
    
    @classmethod
    def from_memory(cls, memory: Memory) -> "MemoryRecord":
        """从 Memory 对象创建"""
        embedding_bytes = None
        if memory.embedding:
            embedding_bytes = np.array(memory.embedding, dtype=np.float32).tobytes()
        
        return cls(
            id=memory.id,
            timestamp=memory.timestamp,
            summary=memory.summary,
            keywords=json.dumps(memory.keywords, ensure_ascii=False),
            original_start_time=memory.original_start_time,
            original_end_time=memory.original_end_time,
            original_turn_count=memory.original_turn_count,
            embedding=embedding_bytes
        )


class MemoryDatabase:
    """长期记忆数据库"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or default_config.memory
        
        # 确保目录存在
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建数据库连接
        db_url = f"sqlite:///{self.config.db_path}"
        self.engine = create_engine(db_url, echo=False)
        
        # 创建表
        Base.metadata.create_all(self.engine)
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # 嵌入模型（延迟加载）
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """延迟加载嵌入模型"""
        if self._embedding_model is None and self.config.use_semantic_search:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"加载嵌入模型: {self.config.embedding_model}")
                self._embedding_model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                print(f"警告: 无法加载嵌入模型，将使用关键词匹配: {e}")
                self.config.use_semantic_search = False
        return self._embedding_model
    
    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """计算文本的嵌入向量"""
        if not self.config.use_semantic_search or self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"计算嵌入向量失败: {e}")
            return None
    
    def save_memory(self, memory: Memory) -> bool:
        """保存记忆到数据库"""
        # 如果没有嵌入向量，计算一个
        if memory.embedding is None and self.config.use_semantic_search:
            # 使用摘要和关键词生成嵌入
            text_for_embedding = f"{memory.summary} {' '.join(memory.keywords)}"
            memory.embedding = self.compute_embedding(text_for_embedding)
        
        session = self._get_session()
        try:
            record = MemoryRecord.from_memory(memory)
            session.merge(record)  # merge 支持更新或插入
            session.commit()
            print(f"✓ 记忆已保存: {memory.id}")
            return True
        except Exception as e:
            session.rollback()
            print(f"✗ 保存记忆失败: {e}")
            return False
        finally:
            session.close()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """根据ID获取记忆"""
        session = self._get_session()
        try:
            record = session.query(MemoryRecord).filter(
                MemoryRecord.id == memory_id
            ).first()
            return record.to_memory() if record else None
        finally:
            session.close()
    
    def get_recent_memories(self, limit: int = 10) -> List[Memory]:
        """获取最近的记忆"""
        session = self._get_session()
        try:
            records = session.query(MemoryRecord).order_by(
                MemoryRecord.timestamp.desc()
            ).limit(limit).all()
            return [r.to_memory() for r in records]
        finally:
            session.close()
    
    def search_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Memory]:
        """根据关键词搜索记忆"""
        session = self._get_session()
        try:
            # 简单的关键词匹配
            all_records = session.query(MemoryRecord).all()
            
            scored_records = []
            for record in all_records:
                record_keywords = set(json.loads(record.keywords))
                query_keywords = set(keywords)
                
                # 计算交集得分
                intersection = record_keywords & query_keywords
                if intersection:
                    score = len(intersection) / len(query_keywords)
                    scored_records.append((record, score))
            
            # 按得分排序
            scored_records.sort(key=lambda x: x[1], reverse=True)
            
            return [r.to_memory() for r, _ in scored_records[:limit]]
        finally:
            session.close()
    
    def search_by_semantic(self, query: str, limit: int = 5) -> List[Tuple[Memory, float]]:
        """语义搜索记忆"""
        if not self.config.use_semantic_search or self.embedding_model is None:
            # 回退到关键词搜索
            import jieba
            keywords = list(jieba.cut(query))
            memories = self.search_by_keywords(keywords, limit)
            return [(m, 0.5) for m in memories]  # 默认相似度
        
        # 计算查询的嵌入向量
        query_embedding = self.compute_embedding(query)
        if query_embedding is None:
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        session = self._get_session()
        try:
            records = session.query(MemoryRecord).filter(
                MemoryRecord.embedding.isnot(None)
            ).all()
            
            scored_memories = []
            for record in records:
                if record.embedding:
                    mem_vec = np.frombuffer(record.embedding, dtype=np.float32)
                    # 余弦相似度
                    similarity = np.dot(query_vec, mem_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(mem_vec) + 1e-8
                    )
                    
                    if similarity >= self.config.relevance_threshold:
                        memory = record.to_memory()
                        scored_memories.append((memory, float(similarity)))
            
            # 按相似度排序
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            return scored_memories[:limit]
        finally:
            session.close()
    
    def retrieve_relevant_memories(self, query: str, limit: int = None) -> List[Memory]:
        """检索与查询相关的记忆（三路混合检索）。

        执行语义搜索、关键词匹配、文本直匹配三路并行搜索，
        结果合并去重后按优先级排序：语义 > 关键词 > 文本。

        Args:
            query: 用户查询文本
            limit: 返回结果数量上限，默认使用配置值

        Returns:
            相关记忆列表，最多 limit 条
        """
        limit = limit or self.config.max_retrieved_memories

        # 提取查询关键词
        import jieba
        keywords = [w for w in jieba.cut(query) if len(w) > 1]

        # 1. 语义搜索
        sem_results = self.search_by_semantic(query, limit)
        sem_memories = [memory for memory, _ in sem_results]
        seen_ids = {m.id for m in sem_memories}

        # 2. 关键词搜索（始终执行，结果追加到语义结果之后）
        kw_extra = []
        if keywords:
            kw_results = self.search_by_keywords(keywords, limit)
            for m in kw_results:
                if m.id not in seen_ids:
                    kw_extra.append(m)
                    seen_ids.add(m.id)

        # 3. 文本直匹配（始终执行）
        text_extra = []
        if keywords:
            session = self._get_session()
            try:
                all_records = session.query(MemoryRecord).all()
                for record in all_records:
                    if record.id not in seen_ids:
                        if any(kw in record.summary for kw in keywords):
                            text_extra.append(record.to_memory())
                            seen_ids.add(record.id)
            finally:
                session.close()

        # 合并：语义优先，关键词和文本匹配补充，截取前 limit 条
        memories = sem_memories + kw_extra + text_extra
        return memories[:limit]
    
    def get_all_memories(self) -> List[Memory]:
        """获取所有记忆"""
        session = self._get_session()
        try:
            records = session.query(MemoryRecord).order_by(
                MemoryRecord.timestamp.desc()
            ).all()
            return [r.to_memory() for r in records]
        finally:
            session.close()
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        session = self._get_session()
        try:
            deleted = session.query(MemoryRecord).filter(
                MemoryRecord.id == memory_id
            ).delete()
            session.commit()
            return deleted > 0
        except Exception as e:
            session.rollback()
            print(f"删除记忆失败: {e}")
            return False
        finally:
            session.close()
    
    def clear_all_memories(self) -> int:
        """清空所有记忆"""
        session = self._get_session()
        try:
            deleted = session.query(MemoryRecord).delete()
            session.commit()
            print(f"已清空 {deleted} 条记忆")
            return deleted
        except Exception as e:
            session.rollback()
            print(f"清空记忆失败: {e}")
            return 0
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """获取数据库统计信息"""
        session = self._get_session()
        try:
            total_count = session.query(MemoryRecord).count()
            
            if total_count > 0:
                oldest = session.query(MemoryRecord).order_by(
                    MemoryRecord.timestamp.asc()
                ).first()
                newest = session.query(MemoryRecord).order_by(
                    MemoryRecord.timestamp.desc()
                ).first()
                
                return {
                    "total_memories": total_count,
                    "oldest_memory": oldest.timestamp.isoformat() if oldest else None,
                    "newest_memory": newest.timestamp.isoformat() if newest else None,
                    "database_path": str(self.config.db_path)
                }
            else:
                return {
                    "total_memories": 0,
                    "database_path": str(self.config.db_path)
                }
        finally:
            session.close()


# 便捷函数
def get_memory_db(config: MemoryConfig = None) -> MemoryDatabase:
    """获取记忆数据库实例"""
    return MemoryDatabase(config)
