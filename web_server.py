"""
自适应记忆系统 - Web 服务器
============================

基于 FastAPI 的 Web 后端，为前端提供 REST API 和 SSE 流式接口。

对话 & 状态端点:
    POST /api/chat            - 发送消息并流式获取回复 (SSE)
    GET  /api/context         - 获取上下文状态 (token 用量等)
    GET  /api/memories        - 获取所有存储的记忆
    DELETE /api/memories/{id} - 删除指定记忆
    GET  /api/stats           - 获取记忆统计信息
    POST /api/clear           - 清空当前对话上下文
    GET  /api/session         - 导出当前会话
    POST /api/system-prompt   - 设置系统提示词

记忆评估端点:
    GET  /api/eval/pool?n=30  - 随机生成 n 条对话语料预览
    POST /api/eval/run        - 执行多轮对话 + 记忆评估 (SSE 流式进度)

启动方式::

    python web_server.py
    python web_server.py --mock              # 模拟模式 (不加载真实模型)
    python web_server.py --model <path>      # 指定模型路径
    python web_server.py --port 8080         # 指定端口

Author: Jiangsheng Yu
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import Config
from inference_engine import create_engine, QwenInferenceEngine, MockInferenceEngine


# ─── 全局引擎实例 ─────────────────────────────────────────────
engine: Optional[QwenInferenceEngine] = None

app = FastAPI(title="自适应记忆系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 静态文件 & 首页 ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ─── API 端点 ────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    """
    对话接口 — SSE 流式返回 (Server-Sent Events)

    请求体: { "message": "用户消息" }
    
    事件类型:
        thinking  — 思考日志
        token     — 流式 token
        context   — 上下文状态快照
        done      — 完成信号
        error     — 错误
    """
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "message 不能为空"}, status_code=400)

    async def event_stream():
        try:
            mm = engine.memory_manager

            # ── 1. 思考阶段: 压缩 & 记忆检索 ─────────────
            if hasattr(mm, "process_with_thinking"):
                # 收集压缩前的状态
                pre_info = mm.get_context_info()
                thinking_log = mm.process_with_thinking(user_message)

                post_info = mm.get_context_info()

                # 发送思考日志
                yield _sse("thinking", {
                    "log": thinking_log,
                    "compressed": post_info["total_compressions"] > pre_info["total_compressions"],
                    "active_memories": [
                        {
                            "summary": m.summary,
                            "keywords": m.keywords,
                            "timestamp": m.timestamp.isoformat(),
                        }
                        for m in mm.context.active_memories
                    ],
                })
            else:
                mm.add_user_message(user_message)

            # ── 2. 获取模型消息上下文 ─────────────────────
            messages = mm.get_messages_for_model()

            # ── 3. 流式生成回复 ───────────────────────────
            full_response = ""
            try:
                for token in engine.stream_chat_from_messages(messages):
                    full_response += token
                    yield _sse("token", {"text": token})
            except AttributeError:
                # 如果引擎不支持 stream_chat_from_messages，退回非流式
                full_response = engine._generate_response(messages)
                yield _sse("token", {"text": full_response})

            # ── 4. 记录助手消息 ───────────────────────────
            mm.add_assistant_message(full_response)

            # ── 5. 发送上下文状态 ─────────────────────────
            yield _sse("context", mm.get_context_info())

            # ── 6. 完成 ──────────────────────────────────
            yield _sse("done", {"response": full_response})

        except Exception as exc:
            traceback.print_exc()
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/context")
async def get_context():
    """获取当前上下文状态"""
    return engine.get_context_info()


@app.get("/api/memories")
async def get_memories():
    """获取所有存储的长期记忆"""
    memories = engine.memory_manager.memory_db.get_all_memories()
    return [
        {
            "id": m.id,
            "summary": m.summary,
            "keywords": m.keywords,
            "timestamp": m.timestamp.isoformat(),
            "turn_count": m.original_turn_count,
        }
        for m in memories
    ]


@app.get("/api/stats")
async def get_stats():
    """获取记忆统计信息"""
    return engine.get_memory_stats()


@app.post("/api/clear")
async def clear_context():
    """清空当前对话上下文"""
    engine.clear_context()
    return {"status": "ok", "message": "上下文已清空"}


@app.get("/api/session")
async def export_session():
    """导出当前会话"""
    return engine.export_session()


@app.post("/api/system-prompt")
async def set_system_prompt(request: Request):
    """设置系统提示词"""
    body = await request.json()
    prompt = body.get("prompt", "")
    engine.set_system_prompt(prompt)
    return {"status": "ok"}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """删除指定记忆"""
    ok = engine.memory_manager.memory_db.delete_memory(memory_id)
    return {"status": "ok" if ok else "not_found"}


# ─── 记忆评估 API ────────────────────────────────────────────

@app.get("/api/eval/pool")
async def eval_pool(n: int = 30):
    """返回随机生成的 n 条对话语料预览（每次请求随机生成不同内容）。"""
    from demo import generate_random_pool
    n = min(max(n, 1), 20000)
    sample_pool = generate_random_pool(n)
    return {
        "total": len(sample_pool),
        "note": "每次请求随机生成不同内容",
        "turns": [
            {
                "index": i + 1,
                "message": t.message,
                "fact_key": t.fact_key,
                "fact_value": t.fact_value,
                "recall_query": t.recall_query,
            }
            for i, t in enumerate(sample_pool)
        ],
    }


@app.post("/api/eval/run")
async def eval_run(request: Request):
    """
    执行多轮对话 + 记忆评估 — SSE 流式返回进度

    请求体: { "rounds": 20 }

    事件类型:
        turn      — 一轮对话完成
        eval      — 一条回忆测试完成
        report    — 最终评估报告
        error     — 错误
    """
    body = await request.json()
    rounds = min(max(int(body.get("rounds", 20)), 1), 20000)

    async def event_stream():
        import os
        import tempfile
        import random
        from demo import generate_random_pool

        try:
            # 创建隔离的临时引擎，不影响主对话
            tmp_db = Path(tempfile.gettempdir()) / f"eval_{os.getpid()}.db"
            eval_config = Config()
            eval_config.memory.db_path = tmp_db
            eval_config.memory.relevance_threshold = 0.3
            if rounds > 10:
                eval_config.model.max_context_length = 800
                eval_config.model.compression_threshold = 0.5
            if rounds > 20:
                eval_config.memory.max_retrieved_memories = 20

            eval_engine = create_engine(mock=True, config=eval_config)

            # 按指定轮数生成随机语料（每次内容不同）
            turns = generate_random_pool(rounds)

            # 记录事实
            planted_facts = []

            # ── 阶段 1: 多轮对话 ──
            for idx, turn in enumerate(turns, 1):
                pre = eval_engine.get_context_info()
                eval_engine.chat(turn.message, verbose=False)
                post = eval_engine.get_context_info()
                compressed = post["total_compressions"] > pre.get("total_compressions", 0)

                planted_facts.append({
                    "key": turn.fact_key,
                    "value": turn.fact_value,
                    "query": turn.recall_query,
                    "turn_index": idx,
                })

                yield _sse("turn", {
                    "index": idx,
                    "total": rounds,
                    "message": turn.message[:50],
                    "fact_key": turn.fact_key,
                    "tokens": post["current_tokens"],
                    "compressed": compressed,
                    "compressions": post["total_compressions"],
                    "memories_stored": post["total_memories_stored"],
                })

            # ── 阶段 2: 记忆评估（只读） ──
            eval_results = []
            recalled_count = 0

            for fact in planted_facts:
                query = fact["query"]
                expected = fact["value"]

                memories = eval_engine.memory_manager.memory_db.retrieve_relevant_memories(
                    query, limit=eval_config.memory.max_retrieved_memories
                )
                system_content = eval_engine.DEFAULT_SYSTEM_PROMPT
                if memories:
                    mem_lines = []
                    for mem in memories:
                        ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                        mem_lines.append(f"  [{ts}] {mem.summary}")
                    system_content += "\n\n## 相关历史记忆\n" + "\n".join(mem_lines)

                ctx_msgs = [{"role": m.role.value, "content": m.content}
                            for m in eval_engine.memory_manager.context.messages]
                messages = ([{"role": "system", "content": system_content}]
                            + ctx_msgs + [{"role": "user", "content": query}])

                response = eval_engine._generate_response(messages)
                hit = expected.lower() in response.lower()
                if hit:
                    recalled_count += 1

                result = {
                    "fact_key": fact["key"],
                    "fact_value": expected,
                    "query": query,
                    "response": response[:120],
                    "recalled": hit,
                    "turn_index": fact["turn_index"],
                }
                eval_results.append(result)

                yield _sse("eval", {
                    **result,
                    "progress": len(eval_results),
                    "total_facts": len(planted_facts),
                })

            # ── 阶段 3: 最终报告 ──
            total_facts = len(planted_facts)
            # 衰减分析
            third = max(total_facts // 3, 1)
            decay = {}
            for label, group in [
                ("early", eval_results[:third]),
                ("mid", eval_results[third:2*third]),
                ("late", eval_results[2*third:]),
            ]:
                if group:
                    decay[label] = {
                        "count": len(group),
                        "recalled": sum(1 for r in group if r["recalled"]),
                        "rate": round(sum(1 for r in group if r["recalled"]) / len(group) * 100, 1),
                    }

            stats = eval_engine.get_context_info()
            yield _sse("report", {
                "rounds": rounds,
                "total_facts": total_facts,
                "recalled": recalled_count,
                "forgotten": total_facts - recalled_count,
                "recall_rate": round(recalled_count / total_facts * 100, 1) if total_facts else 0,
                "compressions": stats["total_compressions"],
                "memories_stored": stats["total_memories_stored"],
                "results": eval_results,
                "decay": decay,
            })

            # 清理
            try:
                os.unlink(tmp_db)
            except OSError:
                pass

        except Exception as exc:
            traceback.print_exc()
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── 辅助函数 ────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """构造一条 SSE 消息"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ─── 为引擎补充 stream_chat_from_messages 方法 ───────────────
def _patch_engine(eng: QwenInferenceEngine):
    """
    给引擎增加 stream_chat_from_messages 方法，
    该方法接受已构建好的 messages 列表直接进行流式生成，
    避免在 chat 流程中重复处理记忆管理。
    """

    def stream_chat_from_messages(messages):
        """根据已构建的消息列表进行流式生成"""
        if eng.model is None:
            # Mock 引擎 — 直接返回整段
            resp = eng._generate_response(messages)
            yield resp
            return

        from transformers import TextIteratorStreamer
        from threading import Thread

        text = eng.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = eng.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=eng.config.model.max_context_length - eng.gen_config.max_new_tokens,
        ).to(eng.model.device)

        streamer = TextIteratorStreamer(
            eng.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": eng.gen_config.max_new_tokens,
            "temperature": eng.gen_config.temperature,
            "top_p": eng.gen_config.top_p,
            "do_sample": eng.gen_config.do_sample,
            "pad_token_id": eng.tokenizer.pad_token_id,
        }
        thread = Thread(target=eng.model.generate, kwargs=gen_kwargs)
        thread.start()

        for tok in streamer:
            yield tok

        thread.join()

    eng.stream_chat_from_messages = stream_chat_from_messages


# ─── 入口 ────────────────────────────────────────────────────

def main():
    global engine

    parser = argparse.ArgumentParser(description="自适应记忆系统 - Web 服务器")
    parser.add_argument("--mock", action="store_true", help="使用模拟引擎（不加载真实模型）")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    args = parser.parse_args()

    config = Config()
    if args.model:
        config.model.model_name = args.model

    print("╔══════════════════════════════════════════════╗")
    print("║   自适应记忆系统 - Adaptive Memory System    ║")
    print("║              Web Interface                   ║")
    print("╚══════════════════════════════════════════════╝")

    engine = create_engine(config=config, mock=args.mock)
    _patch_engine(engine)

    print(f"\n🌐 前端地址: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
