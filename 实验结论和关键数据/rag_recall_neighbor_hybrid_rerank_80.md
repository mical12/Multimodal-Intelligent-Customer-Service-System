# 专家回复 RAG 召回覆盖实验：neighbor / hybrid / reranker

实验目标：回答质量问题先定位到专家回复阶段的 RAG 召回是否足够。固定同一批需要手册回答的随机 80 题，用 LLM 只评估“召回片段是否足以支撑正确回答”，不评估表达风格。

## 结果汇总

| 方案 | top_k | neighbor_count | 额外策略 | full | partial | miss | 结论 |
|---|---:|---:|---|---:|---:|---:|---|
| baseline embedding | 6 | 1 | 无 | 60 | 16 | 4 | 原始专家 RAG 覆盖率可用，但有明显缺口 |
| embedding + neighbor+1 | 6 | 2 | 无 | 65 | 14 | 1 | 正收益最明确，miss 明显减少 |
| embedding + neighbor+2 | 6 | 3 | 无 | 67 | 12 | 1 | 继续小幅提升，且 miss 不增加 |
| embedding + neighbor+3 | 6 | 4 | 无 | 68 | 9 | 3 | full 最高，但 miss 增加，开始出现噪声 |
| hybrid + neighbor+1 | 6 | 2 | BM25 0.6 + embedding 0.5 | 57 | 18 | 5 | 负收益，BM25 把一些泛问题拉偏 |
| reranker + neighbor+1 | 6 | 2 | embedding top8 后 Qwen3-Reranker 重排 | 65 | 13 | 2 | 与 neighbor+1 基本打平，但更慢且有少量退化 |

## 逐题变化

baseline -> neighbor+1：
- 变好 12 题，变差 4 题，持平 64 题。
- 主要收益：把 3 个 miss 拉到 partial，把多道 partial 拉到 full。
- 仍然 miss：`285`，相机快门按钮拆除问题，手册片段主要讲使用快门按钮，不讲拆除。

neighbor+1 -> hybrid：
- 变好 4 题，变差 13 题，持平 63 题。
- hybrid 对目录标题、通用词、泛安全问题更敏感，容易把精确步骤挤下去。
- 不建议作为专家回复阶段的主检索策略。

neighbor+1 -> reranker：
- 变好 5 题，变差 6 题，持平 69 题。
- 救回了 `100、175、190、224、420` 等题，但把 `86、192、296、326、353、359` 等题排差。
- 当前轻量配置为 `candidate_pool=8, max_length=512, batch_size=4`。效果没有明显超过 embedding+neighbor2，同时耗时显著增加。

## 当前建议

专家回复阶段优先采用：

```yaml
agents:
  expert:
    retrieval:
      top_k: 6
      neighbor_count: 3
```

`neighbor_count=3` 比 `2` 又多出 2 个 full，miss 仍保持 1 个；`neighbor_count=4` 虽然 full 最高，但 miss 增加到 3 个，说明上下文过宽后开始引入噪声。暂不引入 hybrid；reranker 可以保留为后续离线实验方向，但不建议现在进入在线主流程。下一步更值得做的是检查 partial/miss 题的 chunk 切分质量、图片依赖和答案生成提示词，而不是继续堆检索策略。

## 输出文件

- `rag_recall_random80.csv`
- `rag_recall_embedding_neighbor2.csv`
- `rag_recall_embedding_neighbor3.csv`
- `rag_recall_embedding_neighbor4.csv`
- `rag_recall_hybrid_neighbor2.csv`
- `rag_recall_rerank_neighbor2.csv`
- `evaluate_rag_recall_variants.py`
