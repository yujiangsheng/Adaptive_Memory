#!/usr/bin/env python3
"""
自适应记忆系统 - 演示脚本
========================

演示如何使用自适应记忆系统:
    1. 突破 LLM 的上下文长度限制
    2. 在对话过程中自动压缩旧上下文
    3. 将压缩后的记忆存储到数据库
    4. 检索相关的历史记忆来增强回答
    5. 通过明确的记忆标注来避免幻觉

使用方法:
    python demo.py                    # 模拟模式 (测试)
    python demo.py --real             # 真实模型
    python demo.py --model <path>     # 指定模型路径
    python demo.py -i                 # 交互模式

Author: Jiangsheng Yu
"""

import argparse
import json
from datetime import datetime

from config import Config, ModelConfig, MemoryConfig
from inference_engine import create_engine, QwenInferenceEngine
from memory_manager import create_memory_manager


def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           自适应记忆系统 - Adaptive Memory System            ║
║                        by Jiangsheng Yu                      ║
║                                                              ║
║  KV-cache → 压缩 → 长期记忆                                  ║
║  突破上下文限制 | 避免幻觉                                    ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demo_mock():
    """使用模拟引擎进行演示"""
    print("\n" + "="*60)
    print("🔬 模拟模式演示")
    print("="*60)
    
    # 创建模拟引擎
    engine = create_engine(mock=True)
    
    # 模拟对话
    conversations = [
        "你好！我叫小明，今年25岁，是一名软件工程师。",
        "我最近在学习机器学习，特别是深度学习方面的内容。",
        "你能给我推荐一些学习资源吗？",
        "谢谢！我之前学过 Python 和数据分析。",
        "我对自然语言处理特别感兴趣。",
        "你还记得我叫什么名字吗？我是做什么工作的？"
    ]
    
    for user_input in conversations:
        response = engine.chat(user_input)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 会话统计")
    print("="*60)
    stats = engine.get_memory_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    
    return engine


def demo_compression():
    """演示压缩机制"""
    print("\n" + "="*60)
    print("🗜️ 压缩机制演示")
    print("="*60)
    
    # 创建一个低阈值的配置来触发压缩
    config = Config()
    config.model.max_context_length = 1000  # 设置很低的阈值
    config.model.compression_threshold = 0.5  # 50% 就触发压缩
    
    # 使用模拟引擎
    engine = create_engine(mock=True, config=config)
    
    # 生成长对话来触发压缩
    long_conversations = [
        "我想了解人工智能的历史。人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支，致力于创建能够执行需要人类智能的任务的系统。",
        "人工智能的概念最早可以追溯到古希腊神话中的自动机和中世纪的机械人偶。但作为一个学科，AI始于1956年的达特茅斯会议。",
        "早期的AI研究主要集中在符号推理和问题求解上。1960年代出现了第一批专家系统，这些系统能够在特定领域提供专业级的建议。",
        "1980年代见证了专家系统的繁荣，但随后的'AI寒冬'让研究陷入低谷。直到2010年代，深度学习的突破才让AI重新崛起。",
        "现在请告诉我，我们之前讨论了什么内容？"
    ]
    
    for user_input in long_conversations:
        print(f"\n用户: {user_input[:50]}...")
        
        # 检查压缩前的状态
        info = engine.get_context_info()
        print(f"   压缩前 - tokens: {info.get('current_tokens', 0)}, 消息数: {info.get('message_count', 0)}")
        
        response = engine.chat(user_input, verbose=False)
        
        # 检查压缩后的状态
        info = engine.get_context_info()
        print(f"   压缩后 - tokens: {info.get('current_tokens', 0)}, 压缩次数: {info.get('total_compressions', 0)}")
        print(f"助手: {response[:100]}...")
    
    return engine


def demo_memory_retrieval():
    """演示记忆检索"""
    print("\n" + "="*60)
    print("🔍 记忆检索演示")
    print("="*60)
    
    from models import Memory
    from memory_store import get_memory_db
    
    # 初始化数据库
    db = get_memory_db()
    
    # 添加一些测试记忆
    test_memories = [
        Memory(
            timestamp=datetime.now(),
            summary="用户询问了关于Python编程的问题，讨论了列表和字典的使用方法。",
            keywords=["Python", "编程", "列表", "字典"]
        ),
        Memory(
            timestamp=datetime.now(),
            summary="讨论了机器学习的基础概念，包括监督学习和无监督学习的区别。",
            keywords=["机器学习", "监督学习", "无监督学习"]
        ),
        Memory(
            timestamp=datetime.now(),
            summary="用户分享了他的项目经历，是一个使用React和Node.js开发的Web应用。",
            keywords=["React", "Node.js", "Web应用", "项目"]
        )
    ]
    
    for mem in test_memories:
        db.save_memory(mem)
    
    # 检索测试
    queries = [
        "Python编程",
        "深度学习",
        "前端开发"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = db.retrieve_relevant_memories(query, limit=2)
        for mem in results:
            print(f"  → {mem.summary[:60]}... (关键词: {', '.join(mem.keywords)})")
    
    # 清理测试数据
    print("\n清理测试数据...")
    db.clear_all_memories()


def demo_real_model(model_path: str = None):
    """使用真实模型进行演示"""
    print("\n" + "="*60)
    print("🚀 真实模型演示")
    print("="*60)
    
    config = Config()
    if model_path:
        config.model.model_name = model_path
    
    print(f"加载模型: {config.model.model_name}")
    print("这可能需要几分钟时间...")
    
    try:
        engine = create_engine(config=config, mock=False)
        
        print("\n✓ 模型加载成功！")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'stats' 查看统计信息")
        print("输入 'clear' 清空上下文")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break
                
                if user_input.lower() == 'stats':
                    stats = engine.get_memory_stats()
                    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
                    continue
                
                if user_input.lower() == 'clear':
                    engine.clear_context()
                    continue
                
                response = engine.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n已中断，再见！")
                break
        
        return engine
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("请确保：")
        print("1. 已安装所有依赖: pip install -r requirements.txt")
        print("2. 有足够的显存/内存")
        print("3. 模型路径正确")
        return None


def interactive_mode(engine: QwenInferenceEngine):
    """交互模式"""
    print("\n" + "="*60)
    print("💬 进入交互模式")
    print("="*60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'stats' 查看统计信息")
    print("输入 'info' 查看上下文信息")
    print("输入 'clear' 清空上下文")
    print("输入 'memories' 查看所有记忆")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            
            if user_input.lower() == 'stats':
                stats = engine.get_memory_stats()
                print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
                continue
            
            if user_input.lower() == 'info':
                info = engine.get_context_info()
                print(json.dumps(info, indent=2, ensure_ascii=False))
                continue
            
            if user_input.lower() == 'clear':
                engine.clear_context()
                continue
            
            if user_input.lower() == 'memories':
                if engine.memory_manager:
                    memories = engine.memory_manager.memory_db.get_all_memories()
                    for mem in memories:
                        print(f"  [{mem.timestamp.strftime('%Y-%m-%d %H:%M')}] {mem.summary[:50]}...")
                continue
            
            response = engine.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自适应记忆系统演示")
    parser.add_argument("--real", action="store_true", help="使用真实模型")
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--demo", type=str, choices=["mock", "compression", "retrieval", "all"],
                       default="all", help="演示类型")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.real or args.model:
        engine = demo_real_model(args.model)
        if engine and args.interactive:
            interactive_mode(engine)
    else:
        if args.demo in ["mock", "all"]:
            engine = demo_mock()
        
        if args.demo in ["compression", "all"]:
            demo_compression()
        
        if args.demo in ["retrieval", "all"]:
            demo_memory_retrieval()
        
        if args.interactive and args.demo in ["mock", "all"]:
            interactive_mode(engine)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
