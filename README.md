# T2Ranking_modified
## based on T2Ranking
T2Ranking: https://github.com/THUIR/T2Ranking
The datasets/models/article can be found in https://github.com/THUIR/T2Ranking

dataset/  文件夹下面放置数据集，来自上文的T2Ranking
processed_dev_dataset.json  qrels.retrieval.dev.tsv
qrels.dev.tsv               readme.md

使用extract_dataset.py 从dataset/qrels.dev.tsv 中提取出processed_dev_dataset.json

model/ 文件夹下面放置模型，来自上文的T2Ranking
/home/duansf/dachuang/hjmtest/T2Ranking_modified/model/bert-base-chinese  huggingface的google-bert  bert-base-chinese模型
bert-base-chinese  dual-encoder-trained-with-bm25-negatives.p
bge                dual-encoder-trained-with-hard-negatives.p
cross-encoder.p

bge model 是rerank  model  https://huggingface.co/BAAI/bge-reranker-v2-m3


/home/duansf/dachuang/hjmtest/T2ranker_hjm/dataset/processed_dev_dataset.json
当中的数据格式是
[
  {
    "qid": 0,
    "question": "问题内容",
    "passagelist": {
      "159474": {
        "text": "段落内容",
        "relevance": 3
      },
      "159475": {
        "text": "段落内容",
        "relevance": 3
      }
      // ... 其他段落
    }
  }
]
其中qid是问题id，question是问题内容，passagelist是段落列表，159474是段落id，text是段落内容，relevance是相关性得分。
当前的数据是基于T2Ranking的标注数据集，处理得来，用于单独衡量一个model的排序能力。

数据集/home/duansf/dachuang/hjmtest/T2ranker_hjm/dataset/qrels.dev.tsv当中，qid是问题的id，pid是段落的id，rel是相关性得分。
qid	-	pid	rel
0	0	159474	3
0	0	159475	3
0	0	219110	2
0	0	45280	1
0	0	269554	1
0	0	0	0
0	0	20367	0
0	0	103438	0
0	0	231595	0
0	0	391409	0
0	0	591217	0
0	0	486307	0
0	0	486306	0
1	0	221806	2
数据集/home/duansf/dachuang/hjmtest/T2ranker_hjm/dataset/qrels.retrieval.dev.tsv选取的是，数据集/home/duansf/dachuang/hjmtest/T2ranker_hjm/dataset/qrels.dev.tsv当中，相关度高的段落，也就是rel=2，3的段落，舍去了rel=0，1的段落，并且是按照相关度从高到低来排序的，保留了qid和pid的对应关系。




