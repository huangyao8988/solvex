"""
使用 Langfuse 低级 SDK 方法运行实验：
遍历指定数据集，调用 RAGFlow 自有 API（Converse with chat assistant），并使用 @observe 装饰器进行追踪。
"""
import os
import requests
from typing import Dict, Any, Optional
from langfuse import get_client, observe

# ==================== 配置区域 ====================
# 请务必在运行前设置以下环境变量
# Langfuse 配置 (用于获取数据集和发送Trace)
# LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
# LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
# LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") # 或你的自托管地址

# RAGFlow 配置 (用于API调用)
RAGFLOW_API_BASE = os.getenv("RAGFLOW_API_BASE")  # 例如: http://your-ragflow-server:port
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_CHAT_ID = os.getenv("RAGFLOW_CHAT_ID")  # 你在RAGFlow创建的聊天助手ID

# 实验配置
DATASET_NAME = "test05"  # 你可以修改为你想测试的任何数据集名称
EXPERIMENT_RUN_NAME = "ragflow_converse_experiment_01"  # 本次实验运行的名称，用于在Langfuse UI中标识
# ==================== 配置结束 ====================


def validate_environment():
    """检查必要的环境变量是否已设置。"""
    required_env_vars = {
        # "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY,
        # "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
        "RAGFLOW_API_BASE": RAGFLOW_API_BASE,
        "RAGFLOW_API_KEY": RAGFLOW_API_KEY,
        "RAGFLOW_CHAT_ID": RAGFLOW_CHAT_ID,
    }
    missing_vars = [key for key, value in required_env_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"错误：缺少必需的环境变量: {', '.join(missing_vars)}。请检查配置。")


@observe(name="call_ragflow_converse_api", as_type="generation") # 使用装饰器进行自动插桩[citation:2]
def call_ragflow_converse_api(
    question: str, 
    session_id: Optional[str] = None, 
    stream: bool = False,
    metadata_condition: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    调用 RAGFlow 的 Converse with chat assistant API（自有API）。
    此函数被 @observe 装饰，其输入、输出、耗时和错误将被自动捕获。
    
    参数：
        question: 用户提问的问题
        session_id: 可选的会话ID，用于保持多轮对话上下文
        stream: 是否启用流式响应（默认False，简化处理）
        metadata_condition: 可选的元数据过滤条件
    
    返回：
        完整的响应字典，包含answer、reference等信息
    """
    url = f"{RAGFLOW_API_BASE}/api/v1/chats/{RAGFLOW_CHAT_ID}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    
    # 构建请求体
    payload = {
        "question": question,
        "stream": stream
    }
    
    # 可选参数
    if session_id:
        payload["session_id"] = session_id
    
    if metadata_condition:
        payload["metadata_condition"] = metadata_condition
    
    # 记录请求信息（可选，会更新到当前trace中）
    get_client().update_current_observation(input=payload)

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()  # 如果状态码不是200，抛出HTTPError
        result = response.json()
        
        # 检查响应码
        if result.get("code") != 0:
            error_msg = f"RAGFlow API 返回错误: {result.get('message', '未知错误')}"
            raise RuntimeError(error_msg)
        
        # 非流式响应：直接返回data字段
        if not stream:
            return result.get("data", {})
        
        # 流式响应：需要处理Server-Sent Events (SSE)
        # 注意：这里简化处理，实际流式响应需要按行解析
        else:
            # 对于流式响应，RAGFlow返回的是SSE格式
            # 为简化示例，这里假设调用者知道如何处理流式响应
            return {"stream_response": "流式响应已接收，请使用适当的SSE处理"}
            
    except requests.exceptions.RequestException as e:
        # 网络或HTTP错误
        error_msg = f"RAGFlow API 请求失败: {e}"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        # 其他错误
        error_msg = f"调用 RAGFlow API 时发生错误: {e}"
        raise RuntimeError(error_msg) from e


def extract_answer_from_response(response_data: Dict[str, Any]) -> str:
    """
    从RAGFlow响应中提取答案文本。
    
    参数：
        response_data: call_ragflow_converse_api 返回的数据字典
    
    返回：
        提取的答案文本
    """
    # 根据RAGFlow API文档，非流式响应的答案在data.answer字段
    if "answer" in response_data:
        return response_data["answer"]
    
    # 如果响应结构有变化，尝试其他可能的字段
    elif "data" in response_data and "answer" in response_data["data"]:
        return response_data["data"]["answer"]
    
    # 如果找不到answer字段，返回整个响应供调试
    else:
        return f"无法提取答案，响应结构: {str(response_data)[:200]}..."


def run_experiment_on_dataset(dataset_name: str, experiment_run_name: str):
    """
    使用低级SDK方法在数据集上运行实验。
    循环遍历每个数据集项，执行被观察的函数，并将Trace链接到数据集运行。
    """
    print(f"开始实验运行 '{experiment_run_name}'，使用数据集: {dataset_name}")
    
    # 直接调用 get_client()，它会自动从环境变量中读取配置
    langfuse_client = get_client()
    
    # 存储会话ID，用于多轮对话测试（可选）
    session_ids = {}  # key: dataset_item_id, value: session_id

    # 1. 从Langfuse获取数据集
    try:
        dataset = langfuse_client.get_dataset(dataset_name)
        print(f"数据集 '{dataset_name}' 加载成功，包含 {len(dataset.items)} 个数据项。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        # 可能是数据集不存在，或认证失败
        return

    # 2. 循环遍历数据集中的每一项
    for item in dataset.items:
        # 获取当前数据项的输入
        item_input = item.input
        if isinstance(item_input, dict):
            # 尝试从不同可能的字段中提取问题
            if "input" in item_input:
                question = item_input["input"]
            elif "question" in item_input:
                question = item_input["question"]
            elif "query" in item_input:
                question = item_input["query"]
            else:
                # 如果是字典但没有标准字段，转换为字符串
                question = str(item_input)
        else:
            # 如果input本身就是字符串，直接使用
            question = str(item_input)

        print(f"处理数据项 ID: {item.id}, 问题: {question[:80]}...")

        # 3. 关键：使用 item.run() 上下文管理器执行任务
        try:
            with item.run(run_name=experiment_run_name):
                # 获取之前为该数据项创建的会话ID（如果有）
                previous_session_id = session_ids.get(item.id)
                
                # 调用RAGFlow自有API
                response = call_ragflow_converse_api(
                    question=question,
                    session_id=previous_session_id,
                    stream=False  # 非流式简化处理
                )
                
                # 提取答案
                answer = extract_answer_from_response(response)
                
                # 保存会话ID供后续使用（如果API返回了新的session_id）
                if "session_id" in response:
                    session_ids[item.id] = response["session_id"]
                    print(f"  会话ID: {response['session_id']}")
                
                # 打印答案摘要
                print(f"  得到回答: {answer[:100]}...")
                
                # 检查是否有引用来源
                if "reference" in response and response["reference"]:
                    ref_count = len(response["reference"].get("chunks", {}))
                    print(f"  引用来源: {ref_count} 个文档块")
                
                # 这里可以添加评估逻辑，将answer与item.expected_output比较
                #if hasattr(item, 'expected_output') and item.expected_output:
                    # 简单的字符串包含检查（可根据需要扩展为更复杂的评估）
                #    expected_lower = str(item.expected_output).lower()
                #    answer_lower = answer.lower()
                #    if expected_lower in answer_lower:
                #        print(f"  ✓ 答案包含预期内容")
                #    else:
                #        print(f"  ⚠ 答案可能未包含所有预期内容")
                        
        except Exception as e:
            print(f"  处理数据项 {item.id} 时发生错误: {e}")
            # 错误已被记录在Trace中，实验会继续处理下一项
            continue

    print(f"实验运行 '{experiment_run_name}' 完成。")
    print(f"会话管理统计: 创建了 {len(session_ids)} 个会话")
    print("请访问 Langfuse UI 查看Trace和数据集运行详情。")


def test_single_conversation():
    """测试单个对话，用于调试API连接"""
    print("=== 测试单个对话 ===")
    
    test_question = "你好，请介绍一下你自己"
    
    try:
        response = call_ragflow_converse_api(
            question=test_question,
            stream=False
        )
        
        print(f"问题: {test_question}")
        print(f"回答: {extract_answer_from_response(response)}")
        
        if "session_id" in response:
            print(f"会话ID: {response['session_id']}")
            
        return response
    except Exception as e:
        print(f"测试失败: {e}")
        return None


if __name__ == "__main__":
    # 验证配置
    try:
        validate_environment()
    except ValueError as e:
        print(e)
        exit(1)

    # 可选：先测试单个对话
    # test_response = test_single_conversation()
    
    # 运行实验
    run_experiment_on_dataset(DATASET_NAME, EXPERIMENT_RUN_NAME)
