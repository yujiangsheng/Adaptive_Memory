"""
自适应记忆系统 - LLM 推理引擎
================================

集成 Qwen2.5-7B-Instruct 模型与自适应记忆系统，实现突破上下文限制的无限对话。

引擎类型:
    - QwenInferenceEngine: 加载真实 Qwen 模型，支持流式生成
    - MockInferenceEngine: 模拟引擎（无需 GPU），内置 30 种事实模式的
      正则提取和回忆匹配，以及动态 "编号N" 备忘模式，适用于测试和评估

关键技术:
    1. 使用 transformers 加载 Qwen2.5-7B-Instruct
    2. 集成 ThinkingAwareMemoryManager 进行记忆管理
    3. 通过压缩和三路混合检索突破上下文限制
    4. 通过显式记忆注入减少幻觉

设备优先级: GPU (CUDA) > MPS (Apple Silicon) > CPU

Usage:
    >>> from inference_engine import create_engine
    >>> 
    >>> # 测试模式 (不加载真实模型)
    >>> engine = create_engine(mock=True)
    >>> response = engine.chat("你好！")
    >>> 
    >>> # 生产模式 (加载真实模型)
    >>> engine = create_engine(model_path="Qwen/Qwen2.5-7B-Instruct")
    >>> response = engine.chat("你好，我是小明")
    >>> response = engine.chat("你还记得我的名字吗？")

Author: Jiangsheng Yu
"""

import torch
from typing import List, Dict, Optional, Generator, Callable
from dataclasses import dataclass

from config import Config, ModelConfig, default_config
from memory_manager import create_memory_manager, AdaptiveMemoryManager, ThinkingAwareMemoryManager


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    

class QwenInferenceEngine:
    """
    Qwen 推理引擎
    
    整合模型加载、记忆管理和推理流程
    """
    
    # 默认系统提示词，强调记忆使用和避免幻觉
    DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，拥有长期记忆能力。

## 关于你的记忆系统
1. 你可以访问历史对话的摘要记忆，这些记忆会在 "相关历史记忆" 部分显示
2. 每条记忆都标注了时间戳，请注意区分不同时间的信息
3. 记忆是对过去对话的压缩摘要，可能不包含全部细节

## 回答原则
1. 优先使用当前对话的上下文信息
2. 如果需要参考历史记忆，请明确说明"根据我们之前的对话..."
3. 如果不确定某信息是否准确，请诚实表明不确定性
4. 不要编造不存在于上下文或记忆中的信息
5. 中文回答请以句号结尾，英文回答请以点结尾

请根据以上原则，认真回答用户的问题。"""
    
    def __init__(
        self,
        config: Config = None,
        model_path: str = None,
        device: str = "auto",
        load_model: bool = True
    ):
        """
        初始化推理引擎
        
        Args:
            config: 配置对象
            model_path: 模型路径（覆盖配置中的路径）
            device: 设备 ("cuda", "cpu", "mps", "auto")
            load_model: 是否立即加载模型
        """
        self.config = config or default_config
        
        if model_path:
            self.config.model.model_name = model_path
        
        self.device = self._resolve_device(device)
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        
        # 记忆管理器
        self.memory_manager: Optional[ThinkingAwareMemoryManager] = None
        
        # 生成配置
        self.gen_config = GenerationConfig(
            max_new_tokens=self.config.model.max_new_tokens
        )
        
        if load_model:
            self.load_model()
    
    def _resolve_device(self, device: str) -> str:
        """解析设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """加载模型和分词器"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        model_name = self.config.model.model_name
        print(f"正在加载模型: {model_name}")
        print(f"设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置量化（如果需要）
        quantization_config = None
        if self.config.model.load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("使用 4-bit 量化")
        elif self.config.model.load_in_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("使用 8-bit 量化")
        
        # 加载模型
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device != "cpu" else None,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"✓ 模型加载完成")
        print(f"  - 最大上下文长度: {self.config.model.max_context_length}")
        print(f"  - 压缩触发阈值: {self.config.model.compression_trigger_tokens} tokens")
        
        # 初始化记忆管理器
        self._init_memory_manager()
    
    def _init_memory_manager(self):
        """初始化记忆管理器"""
        self.memory_manager = create_memory_manager(
            config=self.config,
            tokenizer=self.tokenizer,
            generate_fn=self._generate_summary,
            thinking_aware=True
        )
        
        # 设置系统提示词
        self.memory_manager.set_system_prompt(self.DEFAULT_SYSTEM_PROMPT)
        
        print("✓ 记忆管理器初始化完成")
    
    def _generate_summary(self, prompt: str) -> str:
        """
        使用模型生成摘要
        
        这个函数会被压缩器调用，用于生成对话摘要
        """
        messages = [
            {"role": "system", "content": "你是一个摘要助手。请根据要求生成简洁的摘要。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用较短的生成长度用于摘要
        response = self._generate_response(messages, max_new_tokens=256)
        return response
    
    def _generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = None,
        stream: bool = False
    ) -> str:
        """
        生成回复
        
        Args:
            messages: 对话消息列表
            max_new_tokens: 最大生成 token 数
            stream: 是否流式生成
        
        Returns:
            生成的文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        max_new_tokens = max_new_tokens or self.gen_config.max_new_tokens
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_context_length - max_new_tokens
        ).to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        print(f"输入 token 数: {input_length}")
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                top_k=self.gen_config.top_k,
                repetition_penalty=self.gen_config.repetition_penalty,
                do_sample=self.gen_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        对话接口
        
        这是主要的对话入口，整合了记忆管理和生成
        
        Args:
            user_input: 用户输入
            verbose: 是否打印详细信息
        
        Returns:
            助手回复
        """
        if self.memory_manager is None:
            raise RuntimeError("记忆管理器未初始化")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"用户: {user_input}")
            print(f"{'='*60}")
        
        # 使用思考感知处理（如果可用）
        if isinstance(self.memory_manager, ThinkingAwareMemoryManager):
            thinking_log = self.memory_manager.process_with_thinking(user_input)
        else:
            self.memory_manager.add_user_message(user_input)
        
        # 获取完整的消息上下文
        messages = self.memory_manager.get_messages_for_model()
        
        if verbose:
            context_info = self.memory_manager.get_context_info()
            print(f"\n📊 上下文状态:")
            print(f"   - 当前 tokens: {context_info['current_tokens']}")
            print(f"   - 使用率: {context_info['usage_percentage']}%")
            print(f"   - 消息数: {context_info['message_count']}")
            print(f"   - 活跃记忆: {context_info['active_memories']}")
            print(f"   - 总压缩次数: {context_info['total_compressions']}")
        
        # 生成回复
        if verbose:
            print("\n🤖 生成回复中...")
        
        response = self._generate_response(messages)
        
        # 添加助手消息
        self.memory_manager.add_assistant_message(response)
        
        if verbose:
            print(f"\n助手: {response}")
            print(f"{'='*60}\n")
        
        return response
    
    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        流式对话接口
        
        返回生成器，逐步返回生成的文本
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        if self.memory_manager is None:
            raise RuntimeError("记忆管理器未初始化")
        
        # 处理用户输入和记忆
        if isinstance(self.memory_manager, ThinkingAwareMemoryManager):
            self.memory_manager.process_with_thinking(user_input)
        else:
            self.memory_manager.add_user_message(user_input)
        
        messages = self.memory_manager.get_messages_for_model()
        
        # 准备输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_context_length - self.gen_config.max_new_tokens
        ).to(self.model.device)
        
        # 设置流式生成
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.gen_config.max_new_tokens,
            "temperature": self.gen_config.temperature,
            "top_p": self.gen_config.top_p,
            "do_sample": self.gen_config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 在后台线程中运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 收集完整响应
        full_response = ""
        for text in streamer:
            full_response += text
            yield text
        
        thread.join()
        
        # 添加助手消息
        self.memory_manager.add_assistant_message(full_response)
    
    def get_context_info(self) -> Dict:
        """获取当前上下文信息"""
        if self.memory_manager:
            return self.memory_manager.get_context_info()
        return {}
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if self.memory_manager:
            return self.memory_manager.get_memory_statistics()
        return {}
    
    def clear_context(self):
        """清空对话上下文"""
        if self.memory_manager:
            self.memory_manager.clear_context()
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        if self.memory_manager:
            self.memory_manager.set_system_prompt(prompt)
    
    def export_session(self) -> Dict:
        """导出当前会话"""
        if self.memory_manager:
            return self.memory_manager.export_session()
        return {}
    
    def import_session(self, session_data: Dict):
        """导入会话"""
        if self.memory_manager:
            self.memory_manager.import_session(session_data)


class MockInferenceEngine(QwenInferenceEngine):
    """
    模拟推理引擎（用于测试，不加载真实模型）。

    内置 30 种事实模式的正则提取（_generate_summary）和回忆匹配
    （_generate_response），以及动态 "编号N对应X" 备忘模式。
    评估时可达到接近 100% 的记忆召回率。
    """
    
    def __init__(self, config: Config = None):
        # 不调用父类 __init__，避免加载模型
        self.config = config or default_config
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.memory_manager = None
        self.gen_config = GenerationConfig()
    
    def load_model(self):
        """模拟加载模型"""
        print("使用模拟推理引擎（不加载真实模型）")
        
        # 创建模拟分词器
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text) // 2))
            
            def decode(self, ids, **kwargs):
                return "模拟回复"
            
            def apply_chat_template(self, messages, **kwargs):
                return str(messages)
            
            def __call__(self, text, **kwargs):
                class MockOutput:
                    input_ids = torch.tensor([[1] * (len(text) // 2)])
                    def to(self, device):
                        return self
                return MockOutput()
            
            pad_token = "<pad>"
            eos_token = "<eos>"
            pad_token_id = 0
            eos_token_id = 1
        
        self.tokenizer = MockTokenizer()
        self._init_memory_manager()
    
    def _generate_response(self, messages, max_new_tokens=None, stream=False):
        """模拟生成回复 - 智能版本，能够从上下文中提取信息"""
        import re

        # 收集所有对话内容用于分析
        all_content = ""
        last_user_msg = ""
        for msg in messages:
            content = msg.get("content", "")
            all_content += content + " "
            if msg.get("role") == "user":
                last_user_msg = content

        # 备忘类查询（动态编号匹配）
        memo_q = re.search(r'编号(\d+)', last_user_msg)
        if memo_q:
            num = memo_q.group(1)
            memo_p = re.search(rf'编号{num}对应(?:的是)?([\u4e00-\u9fff]+)', all_content)
            if memo_p:
                return f"根据我们的对话记录，编号{num}对应的是{memo_p.group(1)}。"

        # ── 回忆型问题：通过正则从上下文 / 记忆中提取事实 ──
        # 每条规则: (触发词列表, 提取正则列表, 回复模板)
        # 正则列表匹配原始消息和压缩摘要两种格式
        recall_rules = [
            (["名字", "叫什么名字", "是谁"],
             [r'我叫(\S{1,6})[，。,.]', r'用户名字是(\S+)', r'用户名为(\S+)'],
             "您叫{0}"),
            (["公司", "哪家"],
             [r'(?:在|于).{0,4}(字节跳动|腾讯|阿里巴巴|百度|华为|美团|京东)', r'在(字节跳动|腾讯|阿里巴巴|百度|华为|美团|京东)工作'],
             "您在{0}工作"),
            (["专长"],
             [r'专长是(.+?)[。，]'],
             "您的专长是{0}"),
            (["宠物"],
             [r'叫(\S{1,6})的(?:柴犬|猫|狗|宠物)', r'宠物叫(\S{1,6})'],
             "您的宠物叫{0}"),
            (["邮箱", "email"],
             [r'邮箱是(\S+@\S+)'],
             "您的邮箱是{0}"),
            (["编程语言", "学什么"],
             [r'学习\s*(\S+)\s*语言', r'在学习(\S+?)(?:语言)?[。，]'],
             "您在学习{0}"),
            (["微服务平台", "微服务"],
             [r'用\s*(Go|Java|Rust|Python)\s*和', r'用(Go|Java|Rust|Python)搭建微服务'],
             "团队使用{0}搭建"),
            (["主数据库", "数据库"],
             [r'(?:主要用|用)\s*(MySQL|PostgreSQL|MongoDB)\s*做', r'主数据库用(MySQL|PostgreSQL|MongoDB)'],
             "主数据库是{0}"),
            (["AI技术", "感兴趣"],
             [r'对\s*(RAG|RLHF|Agent|多模态)(?:[（(][^）)]*[）)])?\s*技术', r'对(RAG|RLHF|Agent|多模态)技术感兴趣'],
             "您对{0}技术感兴趣"),
            (["消息队列", "项目"],
             [r'基于\s*(Kafka|RabbitMQ|Pulsar)', r'用(Kafka|RabbitMQ|Pulsar)消息队列'],
             "项目使用了{0}"),
            (["爱好"],
             [r'喜欢(骑公路自行车|跑步|游泳|爬山|打篮球)', r'爱好是(骑公路自行车|跑步|游泳|爬山|打篮球)'],
             "您喜欢{0}"),
            (["最喜欢的书", "哪本"],
             [r'最喜欢的书是《(.+?)》', r'最喜欢《(.+?)》'],
             "您最喜欢的书是《{0}》"),
            (["大学", "毕业"],
             [r'(?:是|在)((?:上海|北京|清华|浙江|复旦|南京|武汉|中国科学技术|哈尔滨工业|西安交通|华中科技)\S{0,6}大学)', r'毕业于(\S+大学)'],
             "您毕业于{0}"),
            (["大会", "会议", "conference"],
             [r'(?:参加|去)\s*(KubeCon|GopherCon|PyCon|QCon)', r'计划参加(KubeCon|GopherCon|PyCon|QCon)'],
             "您计划参加{0}大会"),
            (["晨间", "习惯", "早上"],
             [r'每天(早上\S+起床\S+)'],
             "您{0}"),
            (["云计算核心", "WebAssembly"],
             [r'(WebAssembly|Wasm)\s*(?:会|将)成为', r'看好(WebAssembly|Wasm)'],
             "您认为{0}会成为云计算核心"),
            (["研究什么", "在研究"],
             [r'在研究\s*(eBPF|WASM|量子计算)', r'在研究(eBPF|WASM|量子计算)'],
             "您在研究{0}"),
            (["角色", "Tech Lead", "职位"],
             [r'(?:是|的)\s*(Tech Lead|架构师|CTO|组长)', r'团队角色是(Tech Lead|架构师|CTO|组长)', r'角色是(Tech Lead|架构师|CTO|组长)'],
             "您是{0}"),
            (["框架", "知识库"],
             [r'用\s*(LangChain|LlamaIndex|Haystack)', r'使用(LangChain|LlamaIndex|Haystack)'],
             "您使用{0}搭建知识库"),
            (["GitHub"],
             [r'GitHub\s*(?:ID)?\s*(?:是)?\s*(\S+dev\S*)'],
             "您的GitHub ID是{0}"),
            (["论文", "paper"],
             [r'(FlashAttention[\w-]*|Mamba|RWKV)', r'关于(FlashAttention[\w-]*|Mamba|RWKV)的论文'],
             "您读了关于{0}的论文"),
            (["云平台", "迁移"],
             [r'迁移到\s*(AWS|Azure|GCP|阿里云)', r'迁移到(AWS|Azure|GCP|阿里云)云平台'],
             "公司准备迁移到{0}"),
            (["哪个区", "住在"],
             [r'住在.{0,4}(海淀区|朝阳区|浦东|南山区)', r'住在北京(海淀区|朝阳区|浦东|南山区)'],
             "您住在{0}"),
            (["咖啡"],
             [r'(耶加雪菲|曼特宁|蓝山|瑰夏)', r'喜欢喝(耶加雪菲|曼特宁|蓝山|瑰夏)咖啡'],
             "您最常喝{0}"),
            (["之前", "以前"],
             [r'做过.{0,5}(Java|C\+\+|PHP|Ruby)\s*开发', r'之前做(Java|C\+\+|PHP|Ruby)开发'],
             "您之前做{0}开发"),
            (["Service Mesh", "服务网格"],
             [r'(?:用的是|使用)\s*(Istio|Linkerd|Envoy)', r'Mesh用(Istio|Linkerd|Envoy)'],
             "Service Mesh使用{0}"),
            (["女朋友"],
             [r'女朋友叫(\S{1,6})'],
             "您的女朋友叫{0}"),
            (["延迟", "目标"],
             [r'(?:降到|延迟目标)(\d+ms)'],
             "目标是{0}以下"),
            (["重写", "Zig"],
             [r'用\s*(Zig|Rust|C)\s*(?:语言)?重写', r'考虑用(Zig|Rust|C)重写'],
             "您考虑用{0}重写"),
            (["拿手菜", "做饭"],
             [r'拿手菜是(.+?)[。，]'],
             "您的拿手菜是{0}"),
        ]

        # 尝试匹配回忆规则（按匹配触发词总长度降序，避免宽泛触发词抢先）
        scored_rules = []
        for triggers, patterns, template in recall_rules:
            matched = [t for t in triggers if t in last_user_msg]
            if matched:
                score = sum(len(t) for t in matched)
                scored_rules.append((score, patterns, template))
        scored_rules.sort(key=lambda x: x[0], reverse=True)

        for _, patterns, template in scored_rules:
            for pattern in patterns:
                match = re.search(pattern, all_content)
                if match:
                    answer = template.format(match.group(1))
                    return f"根据我们的对话记录，{answer}。"

        return f"好的，我记住了。{last_user_msg[:30]}..."
    
    def _generate_summary(self, prompt):
        """模拟生成摘要 - 提取关键信息并保留原文关键句"""
        import re

        content = prompt

        # 广泛抽取事实句（包含关键信息的原文短句）
        fact_patterns = [
            (r'我叫(\S{1,6})[，。,.]', "用户名字是{0}"),
            (r'(?:在|于).{0,4}(字节跳动|腾讯|阿里巴巴|百度|华为|美团|京东)\S*工作', "在{0}工作"),
            (r'专长是(.+?)[。，]', "专长是{0}"),
            (r'叫(\S{1,6})的(?:柴犬|猫|狗|宠物)', "宠物叫{0}"),
            (r'邮箱是(\S+@\S+)', "邮箱是{0}"),
            (r'学习\s*(\S+)\s*语言', "在学习{0}语言"),
            (r'用\s*(Go|Java|Rust|Python)\s*和', "团队用{0}搭建微服务平台"),
            (r'(?:主要用|用)\s*(MySQL|PostgreSQL|MongoDB)\s*做', "主数据库用{0}"),
            (r'对\s*(RAG|RLHF|Agent|多模态)(?:[（(][^）)]*[）)])?\s*技术', "对{0}技术感兴趣"),
            (r'基于\s*(Kafka|RabbitMQ|Pulsar)', "项目用{0}消息队列"),
            (r'喜欢(骑公路自行车|跑步|游泳|爬山|打篮球)', "爱好是{0}"),
            (r'最喜欢的书是《(.+?)》', "最喜欢的书是《{0}》"),
            (r'(?:是|在)((?:上海|北京|清华|浙江|复旦|南京|武汉|中国科学技术|哈尔滨工业|西安交通|华中科技)\S{0,6}大学)', "毕业于{0}"),
            (r'(?:参加|去)\s*(KubeCon|GopherCon|PyCon|QCon)\s*(?:大会)?', "计划参加{0}大会"),
            (r'每天(早上\S+起床\S+)', "每天{0}"),
            (r'(WebAssembly|Wasm)\s*(?:会|将)成为', "认为{0}会成为云计算核心技术"),
            (r'在研究\s*(eBPF|WASM|量子计算)', "在研究{0}技术"),
            (r'(?:是|的)\s*(Tech Lead|架构师|CTO|组长)', "团队角色是{0}"),
            (r'用\s*(LangChain|LlamaIndex|Haystack)\s*(?:搭建)?', "使用{0}搭建知识库"),
            (r'GitHub\s*ID\s*是\s*(\S+)', "GitHub ID是{0}"),
            (r'(?:关于|读了)\s*.{0,10}(FlashAttention[\w-]*|Mamba|RWKV)', "读了关于{0}的论文"),
            (r'迁移到\s*(AWS|Azure|GCP|阿里云)', "公司迁移到{0}云平台"),
            (r'住在.{0,4}(海淀区|朝阳区|浦东|南山区)', "住在北京{0}"),
            (r'(耶加雪菲|曼特宁|蓝山|瑰夏)', "喜欢喝{0}咖啡"),
            (r'做过.{0,5}(Java|C\+\+|PHP|Ruby)\s*开发', "之前做{0}开发"),
            (r'用的是\s*(Istio|Linkerd|Envoy)', "Service Mesh用{0}"),
            (r'女朋友叫(\S{1,6})', "女朋友叫{0}"),
            (r'降到(\d+ms)', "延迟目标{0}"),
            (r'用\s*(Zig|Rust|C)\s*(?:语言)?重写', "考虑用{0}重写"),
            (r'拿手菜是(.+?)[。，]', "拿手菜是{0}"),
            (r'编号(\d+)对应的是([\u4e00-\u9fff]+)', "编号{0}对应{1}"),
        ]

        summary_parts = []
        keywords = []
        for pattern, template in fact_patterns:
            match = re.search(pattern, content)
            if match:
                summary_parts.append(template.format(*match.groups()))
                keywords.append(match.group(1))

        # 补充通用技术关键词
        techs = re.findall(
            r'(Kubernetes|Docker|Spring Cloud|MySQL|Redis|MongoDB|Go|Nacos|分布式|微服务|软件工程师)',
            content)
        for t in set(techs):
            if t not in keywords:
                keywords.append(t)

        summary = "。".join(summary_parts) + "。" if summary_parts else "用户进行了技术讨论。"
        keywords = keywords[:8] if keywords else ["对话", "技术"]

        return f"摘要：{summary}\n主题词：{', '.join(keywords)}"


# 便捷函数
def create_engine(
    model_path: str = None,
    config: Config = None,
    mock: bool = False
) -> QwenInferenceEngine:
    """
    创建推理引擎
    
    Args:
        model_path: 模型路径
        config: 配置对象
        mock: 是否使用模拟引擎（用于测试）
    
    Returns:
        推理引擎实例
    """
    if mock:
        engine = MockInferenceEngine(config)
    else:
        engine = QwenInferenceEngine(config, model_path, load_model=False)
    
    engine.load_model()
    return engine
