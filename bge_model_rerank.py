from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import torch
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import time
from rerank_res_eval import evaluate_ranking_quality
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BGEReranker:
    def __init__(self, model_path: str = "./model/bge"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def calculate_scores(self, query: str, passages: List[str], batch_size: int = 128) -> List[float]:
        """批量计算相关性分数"""
        scores = []
        
        # 分批处理避免内存溢出
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            
            # 构造模型输入
            inputs = self.tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                max_length=256,  # 减小最大长度
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 推理计算
            with torch.no_grad():
                batch_scores = self.model(**inputs).logits
                # 确保输出始终是一维张量
                if batch_scores.dim() == 0:  # 如果是标量
                    batch_scores = batch_scores.unsqueeze(0)
                elif batch_scores.dim() == 2:  # 如果是 [batch_size, 1]
                    batch_scores = batch_scores.squeeze(-1)
                
                # 转换为列表
                scores.extend(batch_scores.cpu().tolist())
        
        return scores

    def rerank_passages(self, query: str, passages: dict) -> List[Tuple[int, float]]:
        """对段落进行重排序"""
        # 提取文本和ID
        passage_texts = [p['text'] for p in passages.values()]
        passage_ids = list(passages.keys())
        
        # 计算分数
        scores = self.calculate_scores(query, passage_texts)
        
        # 组合ID和分数并排序
        scored_passages = list(zip(passage_ids, scores))
        return sorted(scored_passages, key=lambda x: x[1], reverse=True)

def main():
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 设置日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/rerank_eval_bge_{timestamp}.log"
    result_file = f"results/test_result_bge_{timestamp}.tsv"
    
    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),      # 文件处理器
            logging.StreamHandler()             # 控制台处理器
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"日志路径: {log_file}")
    logger.info(f"结果路径: {result_file}")
    logger.info("开始BGE重排序评估任务")
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 初始化重排序模型
    reranker = BGEReranker()
    logger.info("模型加载完成")
    
    # 加载数据集
    with open("dataset/processed_dev_dataset.json", 'r') as f:
        dataset = json.load(f)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个查询")
    
    # 处理查询
    with open(result_file, 'w') as f:
        f.write("qid\tpid\n")  # 写入标题行
        
        for item in tqdm(dataset, desc="处理查询"):
            qid = item['qid']
            question = item['question']
            passages = item['passagelist']
            
            # 重排序
            ranked_passages = reranker.rerank_passages(question, passages)
            
            # 写入结果
            for pid, _ in ranked_passages:
                f.write(f"{qid}\t{pid}\n")
    
    # 评估结果
    reference_file = "dataset/qrels.retrieval.dev.tsv"
    metrics = evaluate_ranking_quality(reference_file, result_file)
    
    logger.info("\n评估结果:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()