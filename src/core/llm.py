import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# 載入 .env 變數
load_dotenv()

class LLMService:
    def __init__(self):
        # 初始化 OpenAI Client
        # 這裡的寫法同時支援 OpenAI 官方 API 與 Local LLM (如 vLLM/Ollama)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") # 預留給 Local LLM 用
        )
        self.model = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

    def get_response(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        發送 prompt 給 LLM 並取得回應字串。
        支援 GPT-5 / o1 系列 (移除 temperature，使用 max_completion_tokens)
        """
        try:
            # 準備訊息列表
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 如果有 system_prompt，把它加到最前面
            # (注意：某些推理模型偏好把 system 指令直接寫在 user content 裡，
            # 但目前的 API 標準通常允許 Developer Message 或 System Message)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # ----------------------------------------------------
                # ❌ 移除 temperature (GPT-5 / o1 強制為 1)
                # temperature=0.7, 
                # ----------------------------------------------------
                # ✅ 使用新版參數控制長度
                max_completion_tokens=500  
            )
            return response.choices[0].message.content
            
        except OpenAIError as e:
            print(f"❌ LLM API Error: {e}")
            return f"Error: {str(e)}"