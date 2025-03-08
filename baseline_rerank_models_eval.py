### 在/home/duansf/dachuang/hjmtest/T2ranker_hjm/baseline_rerank_model_eval.py的基础上面，实现对各个model的加载和测试
### 分别是 
### 1. bm25
### 2. cross
### 3. hard


# import argparse
# args = argparse.ArgumentParser()
# args.add_argument("--model_name", type=str, default="bm25")
# ## 可选参数 bm25 cross hard
# ## 对应path 分别是
# ## bm25: /home/duansf/dachuang/hjmtest/T2ranker_hjm/model/cross-encoder.p
# ## cross: /home/duansf/dachuang/hjmtest/T2ranker_hjm/model/dual-encoder-trained-with-bm25-negatives.p
# ## hard: /home/duansf/dachuang/hjmtest/T2ranker_hjm/model/dual-encoder-trained-with-hard-negatives.p
# args = args.parse_args()



import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from tqdm import tqdm
from modeling import DualEncoder, Reranker
from utils import HParams
from rerank_res_eval import evaluate_ranking_quality
import os
from datetime import datetime
import argparse
import logging
import random
import numpy as np

# 设置随机数种子，确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置可见
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logger(model_name, log_dir='logs'):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'rerank_eval_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    # 打印日志路径
    logger.info(f"日志路径: {log_file}")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_model(args, device):
    """根据模型类型加载对应的模型"""
    model_paths = {
        "bm25": "model/cross-encoder.p",
        "cross": "model/dual-encoder-trained-with-bm25-negatives.p",
        "hard": "model/dual-encoder-trained-with-hard-negatives.p",
        "rkm": "model/rkm"  # reranker模型路径
    }
    
    model_path = model_paths[args.model_name]
    logger.info(f"加载模型: {model_path}")
    
    if args.model_name == "rkm":
        # 加载reranker模型
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif args.model_name == "bm25":
        # 加载cross-encoder模型
        model_args = HParams(
            model_name_or_path="model/bert-base-chinese",
            max_seq_len=160,
            gradient_checkpoint=False
        )
        model = Reranker(model_args)
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        # 清理state_dict中的多余键
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # 移除'module.'前缀
            k = k.replace('module.', '')
            # 跳过'position_ids'
            if 'position_ids' in k:
                continue
            cleaned_state_dict[k] = v
        
        # 使用strict=False允许跳过不匹配的键
        model.load_state_dict(cleaned_state_dict, strict=False)
        tokenizer = AutoTokenizer.from_pretrained("model/bert-base-chinese")
    else:
        # 加载dual-encoder模型
        model_args = HParams(
            retriever_model_name_or_path="model/bert-base-chinese",
            max_seq_len=160,
            q_max_seq_len=160, 
            p_max_seq_len=160,
            untie_encoder=True,
            add_pooler=False,
            gradient_checkpoint=False,
            negatives_x_device=False,
            negatives_in_device=False,
            sample_num=8
        )
        model = DualEncoder(model_args)
        state_dict = torch.load(model_path, map_location=device)
        # 清理state_dict
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if 'position_ids' in k:
                continue
            cleaned_state_dict[k] = v
        model.load_state_dict(cleaned_state_dict, strict=False)
        tokenizer = AutoTokenizer.from_pretrained("model/bert-base-chinese")
    
    model.to(device)
    model.eval()
    return model, tokenizer

def encode_text_batch(texts, tokenizer, max_length, model, device, is_query=False):
    """批量编码文本"""
    # 1. 处理输入格式
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # 2. 根据不同模型类型进行处理
    with torch.no_grad():
        if isinstance(model, AutoModelForSequenceClassification):  # rkm模型
            outputs = model(**inputs)
            return outputs.logits
        elif isinstance(model, DualEncoder):  # cross/hard模型
            if is_query:
                return model.encode_query(inputs)
            else:
                return model.encode_passage(inputs)
        else:  # bm25模型 (Reranker)
            return model(inputs)

def rank_passages(question, passages, model, tokenizer, device, batch_size=32):
    """使用批处理对段落进行排序"""
    passage_items = list(passages.items())
    passage_scores = []
    
    if isinstance(model, AutoModelForSequenceClassification):  # rkm模型
        for i in range(0, len(passage_items), batch_size):
            batch = passage_items[i:i + batch_size]
            batch_texts = [(question, p[1]['text']) for p in batch]
            
            # 为rkm模型特别处理输入
            inputs = tokenizer(
                [question] * len(batch),  # queries
                [p[1]['text'] for p in batch],  # passages
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                scores = model(**inputs).logits
            
            for j, (pid, _) in enumerate(batch):
                passage_scores.append((pid, scores[j][1].item()))
    
    elif isinstance(model, DualEncoder):  # cross/hard模型
        q_embedding = encode_text_batch([question], tokenizer, 160, model, device, is_query=True)
        
        for i in range(0, len(passage_items), batch_size):
            batch = passage_items[i:i + batch_size]
            batch_texts = [p[1]['text'] for p in batch]
            
            p_embeddings = encode_text_batch(batch_texts, tokenizer, 160, model, device, is_query=False)
            scores = torch.matmul(q_embedding, p_embeddings.T)
            
            for j, (pid, _) in enumerate(batch):
                passage_scores.append((pid, scores[0, j].item()))
    
    else:  # bm25模型 (Reranker)
        for i in range(0, len(passage_items), batch_size):
            batch = passage_items[i:i + batch_size]
            batch_texts = [(question, p[1]['text']) for p in batch]
            
            scores = encode_text_batch(batch_texts, tokenizer, 160, model, device)
            
            for j, (pid, _) in enumerate(batch):
                passage_scores.append((pid, scores[j].item()))
    
    return sorted(passage_scores, key=lambda x: x[1], reverse=True)

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bm25", choices=["bm25", "cross", "hard","rkm"])
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(args.model_name)


    logger.info(f"开始评估模型: {args.model_name}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model, tokenizer = load_model(args, device)
    
    # 加载数据集
    logger.info("加载数据集...")
    with open("dataset/processed_dev_dataset.json", 'r') as f:
        dataset = json.load(f)
    
    # 创建结果文件
    test_file = f"results/test_result_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    logger.info(f"结果文件路径: {test_file}")
    with open(test_file, 'w') as f:
        f.write("qid\tpid\n")
    
    # 生成排序结果
    logger.info(f"开始处理 {len(dataset)} 个查询...")
    for item in tqdm(dataset, desc="处理查询"):
        qid = item['qid']
        question = item['question']
        passages = item['passagelist']
        
        # 对段落进行排序
        ranked_passages = rank_passages(question, passages, model, tokenizer, device)
        
        # 立即将当前查询的排序结果写入文件
        with open(test_file, 'a') as f:
            for pid, score in ranked_passages:
                f.write(f"{qid}\t{pid}\n")
    
    # 评估排序质量
    logger.info("计算评估指标...")
    reference_file = "dataset/qrels.retrieval.dev.tsv"
    metrics = evaluate_ranking_quality(reference_file, test_file)
    
    logger.info(f"\n模型 {args.model_name} 的排序质量评估结果:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()