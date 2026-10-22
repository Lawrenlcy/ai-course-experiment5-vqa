# 参考文献与代码来源

| 来源 | 用途 | 本项目是否实际使用 |
|---|---|---|
| Li et al. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML 2022. https://arxiv.org/abs/2201.12086 | ViT、文本编码/解码和视觉语言预训练原理 | 使用BLIP VQA预训练模型 |
| https://huggingface.co/Salesforce/blip-vqa-base | 模型卡与权重，BSD-3-Clause | 实际下载，固定revision |
| https://github.com/salesforce/BLIP/blob/main/configs/vqa.yaml | 原始训练数据及推理配置 | 保存配置快照；train_files含vqa_val |
| Goyal et al. Making the V in VQA Matter. CVPR 2017. https://arxiv.org/abs/1612.00837 | VQAv2平衡答案和视觉证据动机 | 实际使用val2014固定500题 |
| https://visualqa.org/evaluation.html | 官方答案格式、规范化与留一评分 | 遵循官方协议 |
| https://github.com/GT-Vision-Lab/VQA | 官方VQA评测源码 | third_party/vqaEval.py只改两处print语法，保留BSD许可和原始快照 |
| Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. https://arxiv.org/abs/1405.0312 | 图像数据来源 | 下载所选图像，保留COCO图像ID |
| Bai et al. Qwen2.5-VL Technical Report. 2025. https://arxiv.org/abs/2502.13923 | 现代多模态生成模型的后续路线 | 仅调研，没有加载或报告其性能 |

## 许可、来源与结果边界
官方VQA evaluator版权属于Aishwarya Agrawal等上游作者，许可见third_party/VQA_LICENSE.txt，不应将它描述为本项目自行提出的评分算法。模型与数据各遵循原许可，MIT仅覆盖原创代码。VQA数据条款见https://visualqa.org/terms.html；COCO条款见https://cocodataset.org/#termsofuse。COCO图像来自各自作者，仓库案例保留文件名、image_id及下载源，不能宣称拥有这些照片版权。

BLIP官方配置包含VQAv2 validation；当前具体检查点的全部训练谱系未独立证明，因此按可能污染处理。使用相同500题比较greedy/beam3/灰图是诊断，不是标准test-dev排行榜。官方原始BLIP还包含候选答案排序推理；本项目使用Transformers开放生成，不能把它与上游ranking成绩不加区分地对比。

## 本项目与参考项目的关系
不是复制整套上游项目作为原创。使用Transformers加载公开权重；数据准备、运行配置与校验、实验组织、可视化、报告材料和交互入口由本项目实现。借鉴代码单独记录于third_party并保留许可。旧版notebook只作历史来源，不复用旧运行数值作为本机成绩。

## 资源选择
在8GB显存下优先完成可复现、能核验的基线和对照；没有将基础模型论文的SOTA成绩当成本项目成绩。较新架构列入文献调研，不声称已经运行。这里的“调研”不是截至2026年的完整排行榜。
