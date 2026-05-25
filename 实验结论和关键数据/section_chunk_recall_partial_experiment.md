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
