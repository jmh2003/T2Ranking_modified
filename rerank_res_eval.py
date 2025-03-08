import numpy as np  

def evaluate_ranking_quality(reference_file, candidate_file):
    """
    评估排序质量
    reference_file: 标准排序文件 (qid\tpid)
    candidate_file: 待评估的排序文件 (qid\tpid)
    """
    # 读取标准排序
    qrels = {}
    with open(reference_file, 'r') as f:
        # 跳过标题行
        header = f.readline()  # 读取第一行
        for line in f:
            try:
                qid, pid = line.strip().split('\t')
                qid = int(qid)
                pid = int(pid)
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(pid)
            except ValueError:
                continue  # 跳过无法转换为整数的行
    
    # 读取待评估排序
    rankings = {}
    with open(candidate_file, 'r') as f:
        # 跳过标题行
        header = f.readline()  # 读取第一行
        for line in f:
            try:
                qid, pid = line.strip().split('\t')
                qid = int(qid)
                pid = int(pid)
                if qid not in rankings:
                    rankings[qid] = []
                rankings[qid].append(pid)
            except ValueError:
                continue  # 跳过无法转换为整数的行
    
    if not rankings:
        print("警告：没有读取到有效的排序数据")
        return {}
        
    metrics = {}
    num_queries = len(rankings)
    
    # 1. MRR (Mean Reciprocal Rank) - 添加MRR@10
    mrr = 0.0
    mrr_10 = 0.0
    for qid in rankings:
        if qid in qrels:
            for rank, pid in enumerate(rankings[qid], 1):
                if pid in qrels[qid]:
                    mrr += 1.0 / rank
                    if rank <= 10:  # MRR@10
                        mrr_10 += 1.0 / rank
                    break
    metrics['MRR'] = mrr / num_queries if num_queries > 0 else 0
    metrics['MRR@10'] = mrr_10 / num_queries if num_queries > 0 else 0
    
    # 2. NDCG (Normalized Discounted Cumulative Gain) - 添加NDCG@20和NDCG@100
    def dcg_at_k(rel_list, k):
        dcg = 0
        for i, rel in enumerate(rel_list[:k], 1):
            dcg += (2 ** rel - 1) / np.log2(i + 1)
        return dcg

    ndcg_5 = 0.0
    ndcg_10 = 0.0
    ndcg_20 = 0.0  # 新增
    ndcg_100 = 0.0  # 新增
    
    for qid in rankings:
        if qid in qrels:
            # 获取排序结果的相关性列表
            rel_list = [1 if pid in qrels[qid] else 0 for pid in rankings[qid]]
            # 获取理想排序的相关性列表
            ideal_rel_list = sorted([1] * len(qrels[qid]) + [0] * (len(rankings[qid]) - len(qrels[qid])), reverse=True)
            
            # 计算各个k值的DCG和IDCG
            dcg5 = dcg_at_k(rel_list, 5)
            idcg5 = dcg_at_k(ideal_rel_list, 5)
            dcg10 = dcg_at_k(rel_list, 10)
            idcg10 = dcg_at_k(ideal_rel_list, 10)
            dcg20 = dcg_at_k(rel_list, 20)  # 新增
            idcg20 = dcg_at_k(ideal_rel_list, 20)  # 新增
            dcg100 = dcg_at_k(rel_list, 100)  # 新增
            idcg100 = dcg_at_k(ideal_rel_list, 100)  # 新增
            
            # 计算各个k值的NDCG
            ndcg_5 += dcg5 / idcg5 if idcg5 > 0 else 0
            ndcg_10 += dcg10 / idcg10 if idcg10 > 0 else 0
            ndcg_20 += dcg20 / idcg20 if idcg20 > 0 else 0  # 新增
            ndcg_100 += dcg100 / idcg100 if idcg100 > 0 else 0  # 新增
    
    metrics['NDCG@5'] = ndcg_5 / num_queries
    metrics['NDCG@10'] = ndcg_10 / num_queries
    metrics['NDCG@20'] = ndcg_20 / num_queries  # 新增
    metrics['NDCG@100'] = ndcg_100 / num_queries  # 新增
    
    # 3. Precision@k
    for k in [1, 3, 5]:
        precision = 0.0
        for qid in rankings:
            if qid in qrels:
                hits = sum(1 for pid in rankings[qid][:k] if pid in qrels[qid])
                precision += hits / k
        metrics[f'P@{k}'] = precision / num_queries
    
    # 4. Average Precision (AP)
    mean_ap = 0.0
    for qid in rankings:
        if qid in qrels:
            ap = 0.0
            hits = 0
            for rank, pid in enumerate(rankings[qid], 1):
                if pid in qrels[qid]:
                    hits += 1
                    ap += hits / rank
            ap = ap / len(qrels[qid]) if len(qrels[qid]) > 0 else 0
            mean_ap += ap
    metrics['MAP'] = mean_ap / num_queries
    
    return metrics

def main():
    # 标准排序文件
    reference_file = "/home/duansf/dachuang/hjmtest/T2ranker_hjm/dataset/qrels.retrieval.dev.tsv"
    test_file = "/home/duansf/dachuang/hjmtest/T2ranker_hjm/test_result_20250308_160431.tsv"

    # 评估排序质量
    metrics = evaluate_ranking_quality(reference_file, test_file)
    
    # 添加论文结果对比
    paper_metrics = {
        'BM25': {'MRR@10': 0.5184, 'NDCG@20': 0.4401, 'NDCG@100': 0.4696},
        'DE': {'MRR@10': 0.5520, 'NDCG@20': 0.5149, 'NDCG@100': 0.5571}
    }
    
    if metrics:
        print("\n排序质量评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\n与论文结果对比:")
        print("论文中的结果:")
        print(f"BM25: MRR@10={paper_metrics['BM25']['MRR@10']:.4f}, NDCG@20={paper_metrics['BM25']['NDCG@20']:.4f}, NDCG@100={paper_metrics['BM25']['NDCG@100']:.4f}")
        print(f"DE: MRR@10={paper_metrics['DE']['MRR@10']:.4f}, NDCG@20={paper_metrics['DE']['NDCG@20']:.4f}, NDCG@100={paper_metrics['DE']['NDCG@100']:.4f}")
    else:
        print("评估失败：没有获取到有效的评估指标")

if __name__ == "__main__":
    main()