from src.core.llm import LLMService

def main():
    print("🤖 初始化 LLM Service...")
    llm = LLMService()
    
    user_input = "請用一句話解釋什麼是 RAG (Retrieval-Augmented Generation)？"
    print(f"👤 User: {user_input}")
    
    print("⏳ AI 思考中...")
    response = llm.get_response(user_input)
    
    print(f"🤖 AI: {response}")

if __name__ == "__main__":
    main()