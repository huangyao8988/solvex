import requests
import json
import sys
import os
from typing import Dict, List, Any, Optional

class RAGFlowRetriever:
    def __init__(self, base_url: str, api_key: str):
        """
        初始化RAGFlow检索器
        
        Args:
            base_url: RAGFlow服务器地址，如 "http://localhost:9380"
            api_key: API密钥
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.endpoint = f"{self.base_url}/api/v1/retrieval"
    
    def load_questions_from_json(self, file_path: str) -> List[str]:
        """
        从JSON文件中读取keywords内容作为查询问题
        
        Args:
            file_path: JSON文件绝对路径
            
        Returns:
            查询问题列表
        """
        try:
            if not os.path.exists(file_path):
                print(f"错误：文件不存在: {file_path}")
                sys.exit(1)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = []
            
            # 检查JSON结构是否符合预期
            if "keywords_transportation_standards" in data:
                for category in data["keywords_transportation_standards"]:
                    # 添加类别信息
                    category_name = category.get("category", "未分类")
                    category_id = category.get("id", "未知ID")
                    
                    # 处理该类别下的所有关键词
                    for keyword in category.get("keywords", []):
                        # 可以按需选择是否添加类别信息到问题中
                        # 选项1: 仅关键词
                        questions.append(keyword)
                        
                        # 选项2: 关键词+类别信息 (根据需求选择)
                        # questions.append(f"{keyword} [{category_name}]")
            
            elif "keywords" in data:
                # 如果JSON是简单格式，直接读取keywords
                questions = data.get("keywords", [])
            else:
                # 尝试查找任何包含关键词的字段
                print("警告：JSON文件格式不符合预期，尝试查找关键词字段...")
                # 递归查找所有字符串列表
                def find_keywords(obj):
                    if isinstance(obj, list):
                        if obj and isinstance(obj[0], str):
                            return obj
                        else:
                            for item in obj:
                                result = find_keywords(item)
                                if result:
                                    return result
                    elif isinstance(obj, dict):
                        for key, value in obj.items():
                            if "keyword" in key.lower() and isinstance(value, list):
                                return value
                            result = find_keywords(value)
                            if result:
                                return result
                    return []
                
                questions = find_keywords(data)
            
            if not questions:
                print("错误：JSON文件中未找到有效的keywords内容")
                sys.exit(1)
            
            print(f"✓ 从JSON文件加载了 {len(questions)} 个查询问题")
            
            # 显示前几个问题预览
            print("前5个查询问题预览:")
            for i, q in enumerate(questions[:5], 1):
                print(f"  {i}. {q[:80]}..." if len(q) > 80 else f"  {i}. {q}")
            if len(questions) > 5:
                print(f"  ... 还有 {len(questions) - 5} 个问题")
            
            return questions
            
        except json.JSONDecodeError as e:
            print(f"错误：JSON文件格式不正确: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"读取JSON文件时出错: {e}")
            sys.exit(1)
    
    def configure_parameters_manually(self) -> Dict[str, Any]:
        """
        手动配置检索参数
        
        Returns:
            配置好的参数字典
        """
        print("=" * 60)
        print("RAGFlow Retrieve Chunks 参数配置")
        print("=" * 60)
        
        params = {}
        
        # 1. 必需参数：问题
        print("\n1. 请选择查询问题输入方式:")
        print("   a) 手动输入查询问题")
        print("   b) 从JSON文件读取keywords作为查询问题")
        
        question_source = input("请选择 (a/b): ").lower().strip()
        
        if question_source == 'a':
            print("\n请输入查询问题 (必填):")
            question = input("> ").strip()
            if not question:
                print("错误：查询问题不能为空！")
                sys.exit(1)
            params["question"] = question
            
        elif question_source == 'b':
            print("\n请输入JSON文件绝对路径 (例如: /home/user/keywords.json):")
            json_path = input("> ").strip()
            if not json_path:
                print("错误：JSON文件路径不能为空！")
                sys.exit(1)
            
            questions = self.load_questions_from_json(json_path)
            
            print("\n请选择查询模式:")
            print("   a) 单问题模式：选择其中一个问题查询")
            print("   b) 批量模式：查询所有问题（多次请求）")
            print("   c) 合并模式：将所有问题合并为一个查询")
            
            query_mode = input("请选择 (a/b/c): ").lower().strip()
            
            if query_mode == 'a':
                print("\n请选择要查询的问题编号:")
                for i, q in enumerate(questions, 1):
                    print(f"  {i}. {q[:60]}..." if len(q) > 60 else f"  {i}. {q}")
                try:
                    choice = int(input(f"请输入编号 (1-{len(questions)}): ").strip())
                    if 1 <= choice <= len(questions):
                        params["question"] = questions[choice-1]
                        print(f"已选择问题: {params['question'][:100]}...")
                    else:
                        print(f"错误：编号必须在1-{len(questions)}之间")
                        sys.exit(1)
                except ValueError:
                    print("错误：请输入有效的数字")
                    sys.exit(1)
                    
            elif query_mode == 'b':
                # 批量模式：存储所有问题，后续单独处理
                params["question"] = questions[0]  # 先使用第一个问题
                params["all_questions"] = questions  # 存储所有问题
                params["query_mode"] = "batch"
                print(f"批量模式：将依次查询 {len(questions)} 个问题")
                
            elif query_mode == 'c':
                # 合并模式：将所有问题合并为一个查询问题
                combined_question = "；".join(questions)
                if len(combined_question) > 1000:
                    print(f"警告：合并后的问题较长 ({len(combined_question)} 字符)")
                    print("考虑使用批量模式或手动精简关键词")
                    
                params["question"] = combined_question
                params["query_mode"] = "combined"
                print(f"合并模式：将 {len(questions)} 个问题合并为一个查询")
                print(f"合并后问题预览: {combined_question[:200]}...")
            else:
                print("错误：无效的选择")
                sys.exit(1)
                
        else:
            print("错误：无效的选择")
            sys.exit(1)
        
        # 2. 数据源选择
        print("\n2. 选择数据源 (至少选择一种):")
        print("   a) 按数据集ID检索")
        print("   b) 按文档ID检索")
        print("   c) 同时使用两种方式")
        
        choice = input("请选择 (a/b/c): ").lower().strip()
        
        if choice in ['a', 'c']:
            print("请输入数据集ID (多个用逗号分隔，例如: id1,id2,id3):")
            dataset_input = input("> ").strip()
            if dataset_input:
                dataset_ids = [id.strip() for id in dataset_input.split(',') if id.strip()]
                params["dataset_ids"] = dataset_ids
        
        if choice in ['b', 'c']:
            print("请输入文档ID (多个用逗号分隔):")
            document_input = input("> ").strip()
            if document_input:
                document_ids = [id.strip() for id in document_input.split(',') if id.strip()]
                params["document_ids"] = document_ids
        
        if 'dataset_ids' not in params and 'document_ids' not in params:
            print("错误：必须至少指定一种数据源！")
            sys.exit(1)
        
        # 3. 分页参数
        print("\n3. 分页参数配置 (可选，直接回车使用默认值):")
        
        page = input("页码 (默认: 1): ").strip()
        if page:
            params["page"] = int(page)
        
        page_size = input("每页数量 (默认: 30): ").strip()
        if page_size:
            params["page_size"] = int(page_size)
        
        # 4. 相似度参数
        print("\n4. 相似度参数配置 (可选):")
        
        similarity_threshold = input("相似度阈值 (默认: 0.2, 范围 0-1): ").strip()
        if similarity_threshold:
            params["similarity_threshold"] = float(similarity_threshold)
        
        vector_weight = input("向量相似度权重 (默认: 0.3, 范围 0-1): ").strip()
        if vector_weight:
            params["vector_similarity_weight"] = float(vector_weight)
        
        top_k = input("候选chunks数量 (默认: 1024): ").strip()
        if top_k:
            params["top_k"] = int(top_k)
        
        # 5. 高级功能
        print("\n5. 高级功能配置 (可选):")
        
        use_kg = input("启用知识图谱检索? (y/n, 默认: n): ").lower().strip()
        if use_kg == 'y':
            params["use_kg"] = True
        
        toc_enhance = input("启用目录增强? (y/n, 默认: n): ").lower().strip()
        if toc_enhance == 'y':
            params["toc_enhance"] = True
        
        highlight = input("启用结果高亮? (y/n, 默认: n): ").lower().strip()
        if highlight == 'y':
            params["highlight"] = True
        
        # 6. 元数据过滤条件
        print("\n6. 是否添加元数据过滤条件? (y/n, 默认: n)")
        if input("> ").lower().strip() == 'y':
            params["metadata_condition"] = self.configure_metadata_condition()
        
        # 7. 跨语言检索
        print("\n7. 是否配置跨语言检索? (y/n, 默认: n)")
        if input("> ").lower().strip() == 'y':
            languages = input("请输入目标语言 (多个用逗号分隔，例如: en,zh,ja): ").strip()
            if languages:
                params["cross_languages"] = [lang.strip() for lang in languages.split(',')]
        
        print("\n✓ 参数配置完成！")
        print("=" * 60)
        
        return params
    
    def configure_metadata_condition(self) -> Dict[str, Any]:
        """
        配置元数据过滤条件
        
        Returns:
            元数据条件字典
        """
        print("配置元数据过滤条件:")
        
        condition = {}
        condition["logic"] = input("逻辑关系 (and/or, 默认: and): ").strip() or "and"
        
        conditions_list = []
        
        while True:
            print(f"\n添加条件 #{len(conditions_list) + 1}:")
            name = input("字段名称 (例如: author, tags): ").strip()
            if not name:
                break
            
            print("比较运算符 (is, not is, contains, not contains, in, not in, start with, end with, >, <, ≥, ≤, empty, not empty):")
            operator = input("> ").strip()
            
            value = None
            if operator not in ['empty', 'not empty']:
                value = input("比较值: ").strip()
            
            conditions_list.append({
                "name": name,
                "comparison_operator": operator,
                "value": value if value is not None else ""
            })
            
            add_more = input("是否继续添加条件? (y/n): ").lower().strip()
            if add_more != 'y':
                break
        
        condition["conditions"] = conditions_list
        return condition
    
    def retrieve_chunks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行检索
        
        Args:
            params: 检索参数
            
        Returns:
            检索结果
        """
        try:
            print(f"\n正在发送请求到: {self.endpoint}")
            print(f"请求参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=params,
                timeout=30
            )
            
            print(f"\nHTTP状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                # 添加查询问题到结果中，便于后续处理
                if "question" in params:
                    result["query_question"] = params["question"]
                return result
            else:
                print(f"请求失败: {response.text}")
                return {"error": f"HTTP {response.status_code}", "message": response.text}
                
        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
            return {"error": "network_error", "message": str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {"error": "json_parse_error", "message": str(e)}
    
    def batch_retrieve_chunks(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        批量执行检索（针对多个查询问题）
        
        Args:
            params: 基础检索参数，包含all_questions字段
            
        Returns:
            检索结果列表
        """
        if "all_questions" not in params:
            print("错误：参数中没有all_questions字段")
            return []
        
        all_questions = params["all_questions"]
        results = []
        
        print(f"\n开始批量检索，共 {len(all_questions)} 个问题")
        print("=" * 60)
        
        for i, question in enumerate(all_questions, 1):
            print(f"\n[{i}/{len(all_questions)}] 正在查询: {question[:80]}...")
            
            # 复制参数并更新问题
            query_params = params.copy()
            query_params["question"] = question
            
            # 执行检索
            result = self.retrieve_chunks(query_params)
            
            # 添加序号和原始问题
            result["query_index"] = i
            result["original_question"] = question
            results.append(result)
            
            # 短暂暂停，避免请求过频
            import time
            if i < len(all_questions):
                time.sleep(0.5)
        
        print(f"\n批量检索完成，共处理 {len(results)} 个查询")
        return results
    
    def display_results(self, result: Dict[str, Any]):
        """
        格式化显示检索结果
        
        Args:
            result: 检索结果
        """
        print("\n" + "=" * 60)
        print("检索结果")
        print("=" * 60)
        
        if "error" in result:
            print(f"错误: {result['error']}")
            print(f"详细信息: {result.get('message', '无')}")
            return
        
        if "code" in result:
            if result["code"] != 0:
                print(f"API错误代码: {result['code']}")
                print(f"错误信息: {result.get('message', '无')}")
                return
            else:
                print("✓ API调用成功")
        
        # 显示查询问题（如果有）
        if "query_question" in result:
            question = result["query_question"]
            print(f"查询问题: {question[:200]}..." if len(question) > 200 else f"查询问题: {question}")
        
        if "data" not in result:
            print("错误: 响应中没有data字段")
            return
        
        data = result["data"]
        
        # 显示统计信息
        print(f"\n1. 统计信息:")
        print(f"   - 匹配chunks总数: {data.get('total', 0)}")
        
        if "doc_aggs" in data:
            print(f"   - 文档分布:")
            for doc in data["doc_aggs"]:
                print(f"     - {doc.get('doc_name', '未知文档')}: {doc.get('count', 0)} 个chunks")
        
        # 显示chunks详情
        if "chunks" in data and data["chunks"]:
            chunks = data["chunks"]
            print(f"\n2. 检索到的chunks ({len(chunks)} 个):")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\n   {'-' * 40}")
                print(f"   Chunk #{i}")
                print(f"   {'-' * 40}")
                
                # 基本信息
                print(f"   ID: {chunk.get('id', 'N/A')}")
                print(f"   文档: {chunk.get('document_name', chunk.get('document_keyword', '未知'))}")
                print(f"   文档ID: {chunk.get('document_id', 'N/A')}")
                print(f"   数据集ID: {chunk.get('kb_id', 'N/A')}")
                
                # 相似度分数
                print(f"   综合相似度: {chunk.get('similarity', 0):.4f}")
                if 'vector_similarity' in chunk:
                    print(f"   向量相似度: {chunk.get('vector_similarity', 0):.4f}")
                if 'term_similarity' in chunk:
                    print(f"   词项相似度: {chunk.get('term_similarity', 0):.4f}")
                
                # 内容
                print(f"\n   内容:")
                content = chunk.get('content', '')
                if 'highlight' in chunk and chunk['highlight']:
                    print(f"     (高亮版): {chunk['highlight'][:500]}...")
                else:
                    print(f"     {content[:500]}..." if len(content) > 500 else f"     {content}")
                
                # 关键词和位置信息
                if 'important_keywords' in chunk and chunk['important_keywords']:
                    print(f"   关键词: {', '.join(chunk['important_keywords'])}")
                
                if 'positions' in chunk and chunk['positions']:
                    print(f"   位置信息: {chunk['positions']}")
        else:
            print("\n2. 未检索到相关chunks")
        
        print("\n" + "=" * 60)
        print("检索完成")
        print("=" * 60)
    
    def display_batch_results(self, results: List[Dict[str, Any]]):
        """
        显示批量检索结果
        
        Args:
            results: 批量检索结果列表
        """
        print("\n" + "=" * 60)
        print("批量检索结果汇总")
        print("=" * 60)
        
        successful_queries = 0
        total_chunks = 0
        
        for i, result in enumerate(results, 1):
            print(f"\n{'='*40}")
            print(f"查询 #{i}: {result.get('original_question', '未知问题')[:100]}...")
            print(f"{'='*40}")
            
            if "error" in result:
                print(f"   ✗ 失败: {result['error']}")
                continue
            
            if "code" in result and result["code"] != 0:
                print(f"   ✗ API错误: {result.get('message', '无')}")
                continue
            
            if "data" in result:
                data = result["data"]
                chunks_count = len(data.get("chunks", []))
                total_chunks += data.get("total", 0)
                
                print(f"   ✓ 成功检索到 {chunks_count} 个chunks")
                print(f"   ✓ 总匹配数: {data.get('total', 0)}")
                successful_queries += 1
        
        print(f"\n{'='*60}")
        print(f"批量检索总结:")
        print(f"  总查询数: {len(results)}")
        print(f"  成功查询: {successful_queries}")
        print(f"  失败查询: {len(results) - successful_queries}")
        print(f"  总匹配chunks数: {total_chunks}")
        print(f"{'='*60}")
    
    def save_results_to_file(self, result: Dict[str, Any], filename: str = "retrieve_results.json"):
        """
        将结果保存到文件
        
        Args:
            result: 检索结果
            filename: 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 结果已保存到文件: {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    
    def save_batch_results_to_file(self, results: List[Dict[str, Any]], filename: str = "batch_retrieve_results.json"):
        """
        将批量检索结果保存到文件
        
        Args:
            results: 批量检索结果列表
            filename: 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_results": results,
                    "summary": {
                        "total_queries": len(results),
                        "successful_queries": sum(1 for r in results if "error" not in r and r.get("code", 1) == 0),
                        "failed_queries": sum(1 for r in results if "error" in r or r.get("code", 1) != 0),
                        "total_chunks": sum(r.get("data", {}).get("total", 0) for r in results if "data" in r)
                    }
                }, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 批量结果已保存到文件: {filename}")
        except Exception as e:
            print(f"保存批量结果文件时出错: {e}")

def main():
    """
    主函数：运行RAGFlow检索程序
    """
    print("=" * 60)
    print("RAGFlow Retrieve Chunks 检索程序")
    print("=" * 60)
    
    # 配置连接信息
    print("\n配置RAGFlow连接信息:")
    base_url = input("RAGFlow服务器地址 (例如: http://localhost:9380): ").strip()
    if not base_url:
        base_url = "http://localhost:9380"
        print(f"使用默认地址: {base_url}")
    
    api_key = input("API密钥: ").strip()
    if not api_key:
        print("错误：API密钥不能为空！")
        sys.exit(1)
    
    # 创建检索器
    retriever = RAGFlowRetriever(base_url, api_key)
    
    # 配置参数
    params = retriever.configure_parameters_manually()
    
    # 检查是否是批量模式
    if params.get("query_mode") == "batch" and "all_questions" in params:
        print("\n开始批量检索...")
        results = retriever.batch_retrieve_chunks(params)
        
        # 显示批量结果汇总
        retriever.display_batch_results(results)
        
        # 询问是否保存结果
        save_option = input("\n是否将批量结果保存到文件? (y/n): ").lower().strip()
        if save_option == 'y':
            filename = input("文件名 (默认: batch_retrieve_results.json): ").strip() or "batch_retrieve_results.json"
            retriever.save_batch_results_to_file(results, filename)
            
        # 询问是否查看单个结果
        view_detail = input("\n是否查看某个查询的详细结果? (输入编号或n): ").strip()
        if view_detail.lower() != 'n':
            try:
                idx = int(view_detail) - 1
                if 0 <= idx < len(results):
                    retriever.display_results(results[idx])
                    
                    # 询问是否保存该结果
                    save_single = input(f"\n是否将查询#{idx+1}的结果单独保存? (y/n): ").lower().strip()
                    if save_single == 'y':
                        filename = f"query_{idx+1}_results.json"
                        retriever.save_results_to_file(results[idx], filename)
                else:
                    print(f"错误：编号必须在1-{len(results)}之间")
            except ValueError:
                print("错误：请输入有效的数字")
    
    else:
        # 单次检索模式
        print("\n正在执行检索...")
        result = retriever.retrieve_chunks(params)
        
        # 显示结果
        retriever.display_results(result)
        
        # 询问是否保存结果
        save_option = input("\n是否将结果保存到文件? (y/n): ").lower().strip()
        if save_option == 'y':
            filename = input("文件名 (默认: retrieve_results.json): ").strip() or "retrieve_results.json"
            retriever.save_results_to_file(result, filename)
    
    print("\n程序执行完毕！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()