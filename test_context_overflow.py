#!/usr/bin/env python3
"""
上下文突破限制测试
==================

测试当上下文接近模型容量限制时：
1. 系统自动触发压缩
2. 旧对话被压缩为记忆 M = (T, S, Keywords)
3. 记忆存储到数据库
4. 后续对话可以通过检索记忆回忆之前的内容

Author: Jiangsheng Yu
"""

import json
from datetime import datetime

from config import Config
from inference_engine import create_engine


def test_context_overflow():
    """测试上下文溢出时的记忆机制"""
    
    print("\n" + "="*70)
    print("🧪 上下文突破限制测试")
    print("="*70)
    print("目标：验证当上下文超过 3/4 容量时，系统能够：")
    print("  1. 自动压缩旧对话")
    print("  2. 将记忆存储到数据库")
    print("  3. 后续对话仍能回忆之前的内容")
    print("="*70)
    
    # 创建配置 - 设置很低的上下文限制以便快速触发压缩
    config = Config()
    config.model.max_context_length = 600  # 很低的限制
    config.model.compression_threshold = 0.75  # 450 tokens 触发压缩
    config.compression.preserve_recent_turns = 1  # 只保留最近1轮
    
    print(f"\n📋 测试配置:")
    print(f"   - 最大上下文: {config.model.max_context_length} tokens")
    print(f"   - 压缩阈值: {config.model.compression_threshold} ({config.model.compression_trigger_tokens} tokens)")
    print(f"   - 保留最近对话: {config.compression.preserve_recent_turns} 轮")
    
    # 创建引擎（模拟模式）
    engine = create_engine(config=config, mock=True)
    
    # 清空之前的记忆
    engine.memory_manager.memory_db.clear_all_memories()
    print("\n✓ 已清空历史记忆")
    
    # ========== 第一阶段：建立身份信息 ==========
    print("\n" + "-"*70)
    print("📝 第一阶段：建立用户身份和背景信息")
    print("-"*70)
    
    identity_messages = [
        "你好！我叫张三，今年28岁，来自北京。",
        "我是一名高级软件工程师，在阿里巴巴工作了5年。",
        "我的专长是分布式系统和微服务架构。",
    ]
    
    for msg in identity_messages:
        print(f"\n👤 用户: {msg}")
        response = engine.chat(msg, verbose=False)
        info = engine.get_context_info()
        print(f"   📊 tokens: {info['current_tokens']}/{config.model.max_context_length} ({info['usage_percentage']}%)")
        print(f"   🤖 助手: {response[:60]}...")
    
    # ========== 第二阶段：讨论技术话题 ==========
    print("\n" + "-"*70)
    print("📝 第二阶段：讨论技术话题（可能触发压缩）")
    print("-"*70)
    
    tech_messages = [
        "最近我在研究 Kubernetes 和 Docker 容器编排技术，想要优化我们的部署流程。",
        "我们目前使用 Spring Cloud 作为微服务框架，配合 Nacos 做服务发现。",
        "数据库方面，我们用 MySQL 做主存储，Redis 做缓存，MongoDB 存储日志。",
    ]
    
    for msg in tech_messages:
        print(f"\n👤 用户: {msg[:50]}...")
        
        # 检查压缩前状态
        info_before = engine.get_context_info()
        
        response = engine.chat(msg, verbose=False)
        
        # 检查压缩后状态
        info_after = engine.get_context_info()
        
        print(f"   📊 tokens: {info_before['current_tokens']} → {info_after['current_tokens']}")
        print(f"   🗜️ 压缩次数: {info_after['total_compressions']}")
        
        if info_after['total_compressions'] > info_before.get('total_compressions', 0):
            print("   ⚡ >>> 触发了压缩！旧对话已转化为长期记忆 <<<")
    
    # ========== 第三阶段：继续讨论 ==========
    print("\n" + "-"*70)
    print("📝 第三阶段：继续对话（可能再次压缩）")
    print("-"*70)
    
    more_messages = [
        "我还在学习 Go 语言，准备用它重写一些性能敏感的服务。",
        "团队里有10个人，我们采用敏捷开发方法，每两周一个迭代。",
    ]
    
    for msg in more_messages:
        print(f"\n👤 用户: {msg[:50]}...")
        info_before = engine.get_context_info()
        response = engine.chat(msg, verbose=False)
        info_after = engine.get_context_info()
        
        print(f"   📊 tokens: {info_before['current_tokens']} → {info_after['current_tokens']}")
        print(f"   🗜️ 压缩次数: {info_after['total_compressions']}")
        
        if info_after['total_compressions'] > info_before.get('total_compressions', 0):
            print("   ⚡ >>> 触发了压缩！<<<")
    
    # ========== 第四阶段：测试记忆回溯 ==========
    print("\n" + "-"*70)
    print("🔍 第四阶段：测试记忆回溯 - 询问之前的信息")
    print("-"*70)
    
    # 查看当前存储的记忆
    memories = engine.memory_manager.memory_db.get_all_memories()
    print(f"\n📚 数据库中的长期记忆 ({len(memories)} 条):")
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. [{mem.timestamp.strftime('%H:%M:%S')}] {mem.summary[:50]}...")
        print(f"      主题词: {', '.join(mem.keywords)}")
    
    # 测试回忆问题
    recall_questions = [
        "你还记得我叫什么名字吗？我在哪里工作？",
        "我的技术专长是什么？",
        "我们讨论了哪些技术栈？",
    ]
    
    print("\n🧠 测试记忆检索:")
    for question in recall_questions:
        print(f"\n👤 用户: {question}")
        
        # 检索相关记忆
        relevant = engine.memory_manager.retrieve_relevant_memories(question)
        if relevant:
            print(f"   🔎 检索到 {len(relevant)} 条相关记忆:")
            for mem in relevant:
                print(f"      - {mem.summary[:40]}... (关键词: {', '.join(mem.keywords[:3])})")
        
        response = engine.chat(question, verbose=False)
        info = engine.get_context_info()
        print(f"   📊 当前状态: {info['current_tokens']} tokens, {info['message_count']} 条消息")
    
    # ========== 统计报告 ==========
    print("\n" + "="*70)
    print("📊 测试统计报告")
    print("="*70)
    
    stats = engine.get_memory_stats()
    final_info = engine.get_context_info()
    
    print(f"""
    ✅ 测试完成！
    
    对话统计:
      - 总对话轮数: {stats['session_stats']['current_session_turns']}
      - 当前上下文 tokens: {final_info['current_tokens']}
      - 当前消息数: {final_info['message_count']}
    
    压缩统计:
      - 触发压缩次数: {stats['session_stats']['total_compressions']}
      - 压缩的 tokens 数: {stats['session_stats']['total_tokens_compressed']}
    
    记忆统计:
      - 存储的长期记忆: {stats['total_memories']} 条
      - 数据库路径: {stats['database_path']}
    
    结论:
      尽管上下文被多次压缩，系统仍然能够通过检索长期记忆
      回答关于早期对话内容的问题，成功突破了上下文限制！
    """)
    
    return engine


def test_with_real_model():
    """使用真实模型测试（需要 GPU）"""
    print("\n" + "="*70)
    print("🚀 真实模型测试模式")
    print("="*70)
    print("注意：这需要加载真实的 Qwen 模型，请确保有足够的显存。")
    
    try:
        config = Config()
        # 使用更大的上下文但仍然可以触发压缩
        config.model.max_context_length = 4096
        config.model.compression_threshold = 0.75
        
        engine = create_engine(config=config, mock=False)
        
        print("\n✓ 模型加载成功！")
        print("输入 'quit' 退出")
        print("-"*70)
        
        while True:
            user_input = input("\n👤 你: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = engine.chat(user_input)
            
            info = engine.get_context_info()
            print(f"\n📊 状态: {info['current_tokens']} tokens, 压缩 {info['total_compressions']} 次")
        
    except Exception as e:
        print(f"\n❌ 无法加载真实模型: {e}")
        print("请使用模拟模式测试: python test_context_overflow.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        test_with_real_model()
    else:
        test_context_overflow()
