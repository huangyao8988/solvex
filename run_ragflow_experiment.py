"""
使用 Langfuse 低级 SDK 方法运行实验：
遍历指定数据集，调用 RAGFlow API，并使用 @observe 装饰器进行追踪。
"""
import os
import requests
from typing import Dict, Any
from langfuse import get_client, observe

# ==================== 配置区域 ====================
# 请务必在运行前设置以下环境变量
# Langfuse 配置 (用于获取数据集和发送Trace)
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") # 或你的自托管地址

# RAGFlow 配置 (用于API调用)
RAGFLOW_API_BASE = os.getenv("RAGFLOW_API_BASE") # 例如: http://your-ragflow-server:port
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_CHAT_ID = os.getenv("RAGFLOW_CHAT_ID") # 你在RAGFlow创建的聊天助手ID

# 实验配置
DATASET_NAME = "test05"  # 你可以修改为你想测试的任何数据集名称
EXPERIMENT_RUN_NAME = "ragflow_experiment_run_01" # 本次实验运行的名称，用于在Langfuse UI中标识
# ==================== 配置结束 ====================

def validate_environment():
    """检查必要的环境变量是否已设置。"""
    required_env_vars = {
        "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY,
        "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
        "RAGFLOW_API_BASE": RAGFLOW_API_BASE,
        "RAGFLOW_API_KEY": RAGFLOW_API_KEY,
        "RAGFLOW_CHAT_ID": RAGFLOW_CHAT_ID,
    }
    missing_vars = [key for key, value in required_env_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"错误：缺少必需的环境变量: {', '.join(missing_vars)}。请检查配置。")

@observe(name="call_ragflow_api", as_type="generation") # 使用装饰器进行自动插桩[citation:2]
def call_ragflow_api(question: str) -> str:
    """
    调用 RAGFlow 的 OpenAI 兼容对话 API。
    此函数被 @observe 装饰，其输入、输出、耗时和错误将被自动捕获。
    """
    url = f"{RAGFLOW_API_BASE}/api/v1/chats_openai/{RAGFLOW_CHAT_ID}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    payload = {
        "model": "ragflow-model",  # 此字段在RAGFlow中通常可忽略或任意填写[citation:3]
        "messages": [{"role": "user", "content": question}],
        "stream": False  # 为简化示例，使用非流式响应
    }

    # 记录请求信息（可选，会更新到当前trace中）
    # get_client().update_current_observation(input=payload)

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()  # 如果状态码不是200，抛出HTTPError
        result = response.json()

        # 解析响应，根据RAGFlow实际返回结构调整
        # 假设返回格式与OpenAI非流式响应类似[citation:3]
        assistant_message = result["choices"][0]["message"]["content"]
        return assistant_message

    except requests.exceptions.RequestException as e:
        # 网络或HTTP错误
        error_msg = f"RAGFlow API 请求失败: {e}"
        # 错误信息会被 @observe 装饰器自动捕获
        raise RuntimeError(error_msg) from e
    except (KeyError, IndexError) as e:
        # 响应解析错误
        error_msg = f"解析 RAGFlow 响应失败: {e}。原始响应: {result if 'result' in locals() else 'N/A'}"
        raise RuntimeError(error_msg) from e

def run_experiment_on_dataset(dataset_name: str, experiment_run_name: str):
    """
    使用低级SDK方法在数据集上运行实验。
    循环遍历每个数据集项，执行被观察的函数，并将Trace链接到数据集运行。
    """
    print(f"开始实验运行 '{experiment_run_name}'，使用数据集: {dataset_name}")
    langfuse_client = get_client(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST
    )

    # 1. 从Langfuse获取数据集
    try:
        dataset = langfuse_client.get_dataset(dataset_name)
        print(f"数据集 '{dataset_name}' 加载成功。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        # 可能是数据集不存在，或认证失败
        return

    # 2. 循环遍历数据集中的每一项
    for item in dataset.items:
        # 获取当前数据项的输入。假设结构为 {"input": "问题文本", "expected_output": "期望答案"}
        # 根据你的数据集实际结构调整
        item_input = item.input
        if isinstance(item_input, dict) and "input" in item_input:
            question = item_input["input"]
        else:
            # 如果input本身就是字符串，直接使用
            question = str(item_input)

        print(f"处理数据项 ID: {item.id}, 问题: {question[:50]}...")

        # 3. 关键：使用 item.run() 上下文管理器执行任务[citation:2]
        # 这会为每次执行自动创建一个Trace，并将其链接到该数据集项和本次实验运行
        try:
            with item.run(run_name=experiment_run_name):
                # 在这个上下文管理器内，所有被 @observe 装饰的函数调用，
                # 其产生的Trace都会自动成为当前数据项运行（DatasetItemRun）的一部分。
                answer = call_ragflow_api(question)
                print(f"  得到回答: {answer[:50]}...")
                # 你可以在这里添加评估逻辑，将answer与item.expected_output比较
        except Exception as e:
            print(f"  处理数据项 {item.id} 时发生错误: {e}")
            # 错误已被记录在Trace中，实验会继续处理下一项
            continue

    print(f"实验运行 '{experiment_run_name}' 完成。请访问 Langfuse UI 查看Trace和数据集运行详情。")

if __name__ == "__main__":
    # 验证配置
    try:
        validate_environment()
    except ValueError as e:
        print(e)
        exit(1)

    # 运行实验
    run_experiment_on_dataset(DATASET_NAME, EXPERIMENT_RUN_NAME)