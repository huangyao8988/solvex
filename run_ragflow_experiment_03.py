"""
使用 Langfuse 低级 SDK 方法运行实验：
遍历指定数据集，调用 RAGFlow 自有 API（Converse with chat assistant），并使用 @observe 装饰器进行追踪。
采用两轮请求方式：第一轮获取session_id，第二轮使用session_id提问。
"""
import os
import requests
import json
from typing import Dict, Any, Optional, List
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
EXPERIMENT_RUN_NAME = "ragflow_converse_experiment_05"  # 本次实验运行的名称，用于在Langfuse UI中标识
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


@observe(name="create_ragflow_session", as_type="generation")
def create_ragflow_session() -> Optional[str]:
    """
    第一轮请求：创建RAGFlow会话，获取session_id。
    发送一个简单的问候消息来初始化会话。
    
    返回：
        会话ID，如果创建失败则返回None
    """
    url = f"{RAGFLOW_API_BASE}/api/v1/chats/{RAGFLOW_CHAT_ID}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    
    # 发送一个简单的问候来初始化会话
    payload = {
        "question": "你好",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 检查响应码
        if result.get("code") != 0:
            error_msg = f"创建会话失败: {result.get('message', '未知错误')}"
            print(f"  警告: {error_msg}")
            return None
        
        # 从响应中提取session_id
        data = result.get("data", {})
        session_id = data.get("session_id")
        
        if session_id:
            print(f"  创建会话成功，session_id: {session_id}")
            return session_id
        else:
            print(f"  警告: 响应中未找到session_id")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  创建会话请求失败: {e}")
        return None
    except Exception as e:
        print(f"  创建会话时发生错误: {e}")
        return None


@observe(name="call_ragflow_with_session", as_type="generation")
def call_ragflow_with_session(question: str, session_id: str) -> Dict[str, Any]:
    """
    第二轮请求：使用已有的session_id向RAGFlow提问。
    
    参数：
        question: 用户提问的问题
        session_id: 之前获取的会话ID
    
    返回：
        完整的响应字典
    """
    url = f"{RAGFLOW_API_BASE}/api/v1/chats/{RAGFLOW_CHAT_ID}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    
    payload = {
        "question": question,
        "session_id": session_id,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 检查响应码
        if result.get("code") != 0:
            error_msg = f"RAGFlow API 返回错误: {result.get('message', '未知错误')}"
            raise RuntimeError(error_msg)
        
        # 返回完整响应数据
        return result.get("data", {})
            
    except requests.exceptions.RequestException as e:
        error_msg = f"RAGFlow API 请求失败: {e}"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"调用 RAGFlow API 时发生错误: {e}"
        raise RuntimeError(error_msg) from e


def extract_answer_from_response(response_data: Dict[str, Any]) -> str:
    """
    从RAGFlow响应中提取答案文本。
    
    参数：
        response_data: call_ragflow_with_session 返回的数据字典
    
    返回：
        提取的答案文本
    """
    # 根据RAGFlow API文档，答案在data.answer字段
    if "answer" in response_data:
        return response_data["answer"]
    
    # 如果找不到answer字段，返回整个响应供调试
    else:
        return f"无法提取答案，响应结构: {str(response_data)[:200]}..."


def parse_reference_info(reference: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析RAGFlow返回的引用信息。
    
    根据RAGFlow API文档，reference字段的结构可能是：
    1. chunks 是字典：{"20": {...}, "21": {...}}
    2. chunks 是列表：[{...}, {...}]
    3. doc_aggs 是字典：{"doc_name": {...}}
    4. doc_aggs 是列表：[{"doc_name": "...", "count": 1}, ...]
    
    参数：
        reference: RAGFlow返回的reference字段
    
    返回：
        解析后的引用信息字典
    """
    result = {
        "chunks_count": 0,
        "documents": []
    }
    
    if not reference:
        return result
    
    # 处理chunks
    chunks = reference.get("chunks", {})
    if isinstance(chunks, dict):
        result["chunks_count"] = len(chunks)
    elif isinstance(chunks, list):
        result["chunks_count"] = len(chunks)
    
    # 处理doc_aggs
    doc_aggs = reference.get("doc_aggs", {})
    documents = []
    
    if isinstance(doc_aggs, dict):
        # doc_aggs是字典
        for doc_name, doc_info in doc_aggs.items():
            if isinstance(doc_info, dict):
                document = {
                    "doc_name": doc_info.get("doc_name", doc_name),
                    "doc_id": doc_info.get("doc_id", ""),
                    "count": doc_info.get("count", 0)
                }
                documents.append(document)
    elif isinstance(doc_aggs, list):
        # doc_aggs是列表
        for doc_info in doc_aggs:
            if isinstance(doc_info, dict):
                document = {
                    "doc_name": doc_info.get("doc_name", "未知文档"),
                    "doc_id": doc_info.get("doc_id", ""),
                    "count": doc_info.get("count", 0)
                }
                documents.append(document)
    
    result["documents"] = documents
    return result


def parse_streaming_response(response):
    """
    解析RAGFlow的流式响应（Server-Sent Events格式）。
    
    根据API文档，流式响应格式为：
    data:{...}
    data:{...}
    data:[DONE]
    """
    content = []
    
    # 按行解析响应
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith('data:'):
            data_line = line[5:].strip()  # 移除'data:'前缀
            
            if data_line == '[DONE]':
                break
                
            try:
                data_json = json.loads(data_line)
                if data_json.get("code") == 0 and "data" in data_json:
                    data_content = data_json["data"]
                    if "answer" in data_content:
                        content.append(data_content["answer"])
                    elif "message" in data_content:
                        content.append(data_content["message"])
            except json.JSONDecodeError:
                continue
    
    # 合并所有内容片段
    full_answer = "".join(content)
    
    return {
        "answer": full_answer,
        "is_streaming": True,
        "chunks_count": len(content)
    }


def run_experiment_on_dataset(dataset_name: str, experiment_run_name: str):
    """
    使用低级SDK方法在数据集上运行实验。
    循环遍历每个数据集项，执行被观察的函数，并将Trace链接到数据集运行。
    采用两轮请求方式：第一轮获取session_id，第二轮使用session_id提问。
    """
    print(f"开始实验运行 '{experiment_run_name}'，使用数据集: {dataset_name}")
    
    # 直接调用 get_client()，它会自动从环境变量中读取配置
    langfuse_client = get_client()
    
    # 会话缓存：key为数据集项ID，value为session_id
    session_cache = {}

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
                # 检查是否已经有该数据项的session_id
                if item.id in session_cache:
                    session_id = session_cache[item.id]
                    print(f"  使用缓存的session_id: {session_id}")
                else:
                    # 第一轮：创建会话，获取session_id
                    print(f"  第一轮：创建会话...")
                    session_id = create_ragflow_session()
                    
                    if not session_id:
                        print(f"  创建会话失败，跳过此项")
                        continue
                    
                    # 缓存session_id
                    session_cache[item.id] = session_id
                
                # 第二轮：使用session_id进行提问
                print(f"  第二轮：使用session_id提问...")
                response = call_ragflow_with_session(
                    question=question,
                    session_id=session_id
                )
                
                # 提取答案
                answer = extract_answer_from_response(response)
                
                # 打印答案摘要
                print(f"  得到回答: {answer[:100]}...")
                
                # 检查是否有引用来源
                if "reference" in response and response["reference"]:
                    # 使用新的解析函数处理reference
                    ref_info = parse_reference_info(response["reference"])
                    
                    print(f"  引用来源: {ref_info['chunks_count']} 个文档块")
                    
                    # 详细显示引用信息
                    if ref_info["documents"]:
                        for doc in ref_info["documents"]:
                            print(f"    - {doc['doc_name']}: {doc['count']} 个片段")
                
                # 这里可以添加评估逻辑，将answer与item.expected_output比较
                if hasattr(item, 'expected_output') and item.expected_output:
                    # 简单的字符串包含检查（可根据需要扩展为更复杂的评估）
                    expected_lower = str(item.expected_output).lower()
                    answer_lower = answer.lower()
                    if expected_lower in answer_lower:
                        print(f"  ✓ 答案包含预期内容")
                    else:
                        print(f"  ⚠ 答案可能未包含所有预期内容")
                        
        except Exception as e:
            print(f"  处理数据项 {item.id} 时发生错误: {e}")
            # 错误已被记录在Trace中，实验会继续处理下一项
            continue

    print(f"实验运行 '{experiment_run_name}' 完成。")
    print(f"会话管理统计: 创建了 {len(session_cache)} 个会话")
    print("请访问 Langfuse UI 查看Trace和数据集运行详情。")


def test_two_round_conversation():
    """测试两轮对话，用于调试API连接"""
    print("=== 测试两轮对话 ===")
    
    test_question = "请介绍一下RAGFlow的主要功能"
    
    try:
        # 第一轮：创建会话
        print("第一轮：创建会话...")
        session_id = create_ragflow_session()
        
        if not session_id:
            print("创建会话失败")
            return None
            
        # 第二轮：使用会话提问
        print(f"第二轮：使用session_id {session_id} 提问...")
        response = call_ragflow_with_session(
            question=test_question,
            session_id=session_id
        )
        
        print(f"问题: {test_question}")
        print(f"回答: {extract_answer_from_response(response)}")
        
        if "reference" in response and response["reference"]:
            ref_info = parse_reference_info(response["reference"])
            print(f"引用来源: {ref_info['chunks_count']} 个文档块")
            
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

    # 可选：先测试两轮对话
    print("=== 先执行测试对话 ===")
    test_response = test_two_round_conversation()
    
    if test_response:
        print("测试成功，开始执行实验...\n")
        # 运行实验
        run_experiment_on_dataset(DATASET_NAME, EXPERIMENT_RUN_NAME)
    else:
        print("测试失败，请检查配置和网络连接")