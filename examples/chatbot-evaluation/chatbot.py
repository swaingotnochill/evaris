"""Simple chatbot using QWEN LLM API."""

import os
import dotenv

dotenv.load_dotenv()


def chatbot(message: str, provider: str = "qwen") -> str:
    """Generate response using QWEN API.

    Args:
        message: User message
        provider: LLM provider (only "qwen" supported)

    Returns:
        Chatbot response
    """
    if provider != "qwen":
        raise ValueError(f"Only 'qwen' provider is supported, got: {provider}")
    return _qwen_chatbot(message)


def _qwen_chatbot(message: str) -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide concise, accurate answers."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content or ""


if __name__ == "__main__":
    test_message = "What is the capital of France?"
    print(f"Question: {test_message}\n")

    try:
        print("QWEN:")
        print(chatbot(test_message))
    except Exception as e:
        print(f"Error: {e}")
