import json
import pandas as pd
from tqdm import tqdm

# 这个代码用来从原来的数据集里面获取数据形成json文档

def create_dataset():
    # 读取原始数据文件
    print("Reading files...")
    
    # 读取查询文件
    queries_df = pd.read_csv("/home/duansf/dachuang/hjmtest/T2Ranking/data/queries.dev.tsv", 
                            sep='\t', header=0)
    
    # 读取段落集合
    collection_df = pd.read_csv("/home/duansf/dachuang/hjmtest/T2Ranking/data/collection.tsv",
                               sep='\t', header=0)
    
    # 读取查询-段落对应关系
    qrels_df = pd.read_csv("/home/duansf/dachuang/hjmtest/T2Ranking/data/qrels.dev.tsv",
                          sep='\t', header=0)
    
    # 创建结果列表
    result = []
    
    # 对每个唯一的qid进行处理
    unique_qids = qrels_df['qid'].unique()
    
    print("Processing data...")
    for qid in tqdm(unique_qids):
        # 获取当前qid的问题
        question = queries_df[queries_df['qid'] == qid]['text'].iloc[0]
        
        # 获取当前qid对应的所有pid和相关性分数
        current_qrels = qrels_df[qrels_df['qid'] == qid][['pid', 'rel']]
        
        # 获取所有pid对应的段落和相关性分数
        passage_list = {}
        for _, row in current_qrels.iterrows():
            pid = row['pid']
            rel_score = int(row['rel'])  # 相关性分数
            passage = collection_df[collection_df['pid'] == pid]['text'].iloc[0]
            passage_list[str(pid)] = {
                "text": passage,
                "relevance": rel_score
            }
        
        # 创建当前问题的数据项
        item = {
            "qid": int(qid),
            "question": question,
            "passagelist": passage_list
        }
        
        result.append(item)
    
    # 保存为JSON文件
    print("Saving to JSON file...")
    output_path = "/home/duansf/dachuang/hjmtest/T2Ranking/data/processed_dev_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset created successfully! Total {len(result)} items.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    create_dataset()