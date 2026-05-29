# Section-aware Chunk 召回实验

目标：验证“按 `#` 小节切分 + 短小节相邻聚类 + 长小节保留完整语义”是否能改善 baseline partial 问题。

## 切分规则

- 先按 `#` 标题切出原始小节。
- 相邻短小节按顺序聚类，目标接近 512 chars。
- 聚类合并后不能超过 512 chars。
- 原始长小节不硬切，避免破坏步骤、表格和完整语义。
- 英文汇总手册按意图识别得到的产品分别切分与建索引。

## 输出文件

- `section_chunks_by_product_new.csv`
- `section_chunking_stats_new.csv`
- `section_vs_current_chunking_stats_new.csv`
- `section_chunks_over2000_new.csv`
- `section_chunks_over2000_new_preview.md`
- `section_chunk_recall_baseline_partial.csv`
- `section_chunk_embedding_cache/`

## 重测结果

重测对象：`rag_recall_random80.csv` 中 baseline 判为 partial 的 16 道题。

| 方法 | full | partial | miss |
|---|---:|---:|---:|
| baseline 固定长度 chunk | 0 | 16 | 0 |
| embedding + neighbor_count=3 | 8 | 8 | 0 |
| section-aware chunk top2 | 6 | 10 | 0 |
| section-aware chunk top3 | 6 | 10 | 0 |
| section-aware chunk top4 | 10 | 5 | 1 |
| section-aware chunk top6 | 8 | 8 | 0 |

section-aware 拉成 full 的题：

- `175` 摩托艇滑航速度急转弯
- `177` 摩托艇中低速转弯油门控制
- `192` 温控器安全更换电池
- `292` 相机删除单张图片
- `312` 传真安全注意事项
- `318` 烤架泄漏测试
- `346` WaveRunner 清洁 intake / impeller
- `359` riding lawn mower 后避震调节

top_k 对比：

- top2 和 top3 数量相同，都是 `6 full / 10 partial / 0 miss`。
- top4 数量最好，为 `10 full / 5 partial / 1 miss`。
- top6 反而回落到 `8 full / 8 partial / 0 miss`，说明更多片段可能会给 judge 或生成模型引入噪声。
- `296` 耳机组件在 top4 被判 miss，核心原因是召回到了外观部件、维护和蓝牙说明，但没有包装清单；这更像手册/问题匹配问题，不是 top_k 问题。

仍为 partial 的题：

- `100` 洗碗机可折叠下层篮架：手册主要写折叠/展开，缺安装或预装说明。
- `224` 烤箱烤盘：召回覆盖用途与建议，但缺具体温度/功能/预热步骤。
- `294` 相机 CP Direct 打印：召回有打印相关内容，但缺 CP Direct 的完整步骤。
- `296` 耳机组件：召回有外观部件，但缺明确包装清单。
- `314` 传真搬运：召回有勿倾斜等警告，但缺完整搬运流程。
- `332` WaveRunner 两种 lever：召回有部件用途，但缺具体操作步骤。
- `353` XL490/XL495 handset 安装：召回只说明撕电池贴，缺放置/充电/配置步骤。
- `368` 微波炉 Favorite Recipe：召回解释功能，但缺设置步骤正文。

## 结论

section-aware chunk 的方向是对的，能把一部分由固定长度切断导致的 partial 拉成 full。但在这 16 道里，效果和 `neighbor_count=3` 数量上打平。

后续重点不是继续加 hybrid/reranker，而是：

- 把 section-aware chunk 接入主 RAG 后在完整 80/400 题上评估。
- 对仍 partial 的题做手册全文核验，确认是否“手册本身没有更详细步骤”。
- 对目录、包装清单、表格、图片说明这类特殊结构加规则。

## 当前主程序 chunk 策略统计

后续已把旧的固定长度 RAG 替换为 section-aware chunk，并重新建立缓存。当前策略：

- 先按 `#` 小节切分。
- 过滤明显目录页：`目录` / `Contents` / 大量点线页码。
- 小于 100 chars 的小节向后合并。
- 相邻短小节按顺序聚类，目标接近 512 chars。
- 超过 512 chars 的长小节按句子二次切分，尽量接近 512，同时保留父 chunk 信息：`parent_start`、`parent_end`、`child_index`、`child_count`。
- 中文单产品手册统一使用 `.all.pkl` 缓存；英文汇总手册继续按产品单独缓存。

当前 chunk 长度分布：

| 指标 | 数值 |
|---|---:|
| chunk 总数 | 3157 |
| 最短 | 101 |
| 最长 | 573 |
| 平均 | 380.6 |
| p50 | 416 |
| p75 | 473 |
| p90 | 501 |
| p95 | 509 |
| p99 | 512 |
| >1000 | 0 |
| >2000 | 0 |

长度区间：

| 区间 | 数量 |
|---|---:|
| 0-100 | 0 |
| 100-200 | 345 |
| 200-300 | 422 |
| 300-400 | 664 |
| 400-512 | 1733 |
| 513-700 | 13 |
| 701-1000 | 0 |
| >1000 | 0 |

切分来源：

| strategy | 数量 |
|---|---:|
| section_sentence_split | 1621 |
| section | 665 |
| section_cluster | 478 |
| section_tiny_merged | 305 |
| section_word_split | 88 |

结论：当前 chunk 大小已经比较稳定，绝大多数落在 400-512 chars；没有再出现几千字的大块，也没有 100 字以下的孤立标题块。

## 专家阶段 top4 召回统计

基于 352 道进入专家阶段的问题统计 top4 召回文本总量：

| 指标 | 数值 |
|---|---:|
| 问题数 | 352 |
| 最小 top4 总字符数 | 555 |
| 最大 top4 总字符数 | 2019 |
| 平均 top4 总字符数 | 1615.6 |
| p50 | 1632.5 |
| p75 | 1762 |
| p90 | 1850 |
| p95 | 1893 |
| >2200 | 0 |

这说明 `top_k=4` 下输入给专家 agent 的上下文量可控，通常在 1.6k 字左右，没有明显爆 prompt。

top4 涉及父章节数量：

| unique_parent_count | 问题数 |
|---:|---:|
| 1 | 3 |
| 2 | 12 |
| 3 | 67 |
| 4 | 270 |

其中 79/352 道题的 top4 中至少有 2 个 chunk 来自同一个父章节。这个现象是合理的：二次切分后，同一章节的多个子 chunk 可能同时被召回。后续如果要进一步优化，重点不是微调 top1 排序，而是考虑“命中某个父章节后是否需要补齐相邻子 chunk 或父章节内缺失子 chunk”。

top4 中章节形态：

| 类型 | 问题数 |
|---|---:|
| only_single_chunk_sections | 149 |
| mixed_single_and_split_sections | 163 |
| only_split_sections | 40 |

按父章节行统计：

| 类型 | 数量 |
|---|---:|
| split parent | 417 |
| single parent | 891 |

结论：大多数问题不需要跨很多章节回答；现在的 partial 风险主要不是“完全没召回”，而是“召回到父章节的一部分，但父章节里相邻子 chunk 没被一起带上”。这和之前人工看 partial 的判断一致。

## 后续覆盖度判断口径

判断 RAG 是否覆盖问题时，不能只把 chunk 文本交给 judge 模型。很多手册信息在图片里，文本中只保留了 `<PIC>` 占位，例如装配步骤、按钮位置、部件图、包装清单、错误示例等。如果 judge 看不到图片，会把这类召回误判为 partial 或 miss。

当前新增脚本：

```text
judge_retrieval_coverage_with_images.py
```

评估方式：

- 按当前主程序 RAG 重新召回 topK。
- 将 chunk 中的 `<PIC>` 标注为 `<PIC_序号: 图片名>`。
- 收集这些 `<PIC>` 对应的真实图片，并作为 `image_url` 一起传给 API 大模型。
- 如果图片触发平台 `DataInspectionFailed` 拦截，则自动去掉图片重发，只保留文本和图片标签，并在输出中用 `without_images=1` 标记。
- judge 输出 `full` / `partial` / `miss`，并记录原因、关键证据、使用到的图片名。

示例：

```powershell
D:\anaconda3\envs\work\python.exe judge_retrieval_coverage_with_images.py --top-k 4 --limit 20
```

默认输出：

```text
retrieval_coverage_judge_with_images.csv
```

这比纯文本 judge 更接近真实专家回答阶段，因为专家 agent 本来也会同时收到文本片段和候选图片。

## 语言门控路由规则

多模态 judge 的 miss/error 复盘里发现一类非 RAG 问题：英文问题有时会被中文单品手册吸走。例如英文 `grill / LP Tank / regulator` 问题被路由到中文 `烤箱`，以及 `T-rail mounting` 被路由到单反相机而不是英文汇总里的 `Camera Installation Guide`。

已在意图识别阶段加入语言门控：

- 英文问题：规则命中、全局 RAG、摘要 LLM 只看 `英文汇总手册` 的英文子产品。
- 中文问题：规则命中、全局 RAG、摘要 LLM 只看中文单产品手册。
- 混合问题只要包含中文，就按中文问题处理，避免型号、英文缩写把中文问题带到英文汇总。
- 全局 RAG 路由阶段也会把候选片段中的图片随 prompt 一起传给路由模型；若图片触发 `DataInspectionFailed`，自动去掉图片重发，仅保留文本和图片标签。

相关配置：

```yaml
agents:
  intent:
    global_rag_assist:
      max_images: 20
```

验证样例：

| 问题 | 当前候选/命中 |
|---|---|
| `When using the grill, how to connect regulator to the LP Tank?` | `Grill / 英文汇总手册` |
| `What are the detailed instructions for T-rail mounting of a camera or equipment?` | 全局 RAG 第一候选 `Camera Installation Guide` |
| `如何使用烤箱的烤架？` | `烤箱 / 烤箱手册` |
| `如何为混合即时相机安装肩带？` | `相机 / 相机手册` |
