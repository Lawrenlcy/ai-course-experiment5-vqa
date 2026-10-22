# 实验5：本地结果分析与PPT素材

固定 500 题、495 张图像，随机种子42。模型为Salesforce/blip-vqa-base，没有进行项目微调。全部预测来自本机实际运行。

**这是可能存在训练重叠的VQAv2 validation诊断，不是独立测试集成绩。** 官方BLIP配置含vqa_val；不能据此声称达到论文的未见测试性能。

## 准确率
| method      | group   |   n |   accuracy_percent |
|:------------|:--------|----:|-------------------:|
| beam3       | overall | 500 |             85     |
| beam3       | yes/no  | 219 |             97.854 |
| beam3       | number  |  53 |             67.547 |
| beam3       | other   | 228 |             76.711 |
| blank_image | overall | 500 |             47.56  |
| blank_image | yes/no  | 219 |             67.534 |
| blank_image | number  |  53 |             29.811 |
| blank_image | other   | 228 |             32.5   |
| greedy      | overall | 500 |             85.76  |
| greedy      | yes/no  | 219 |             98.037 |
| greedy      | number  |  53 |             67.547 |
| greedy      | other   | 228 |             78.202 |

评分调用官方VQA评测器，只修改两处Python 2 print语法。保留数字、冠词、标点和缩写规范化，以及十位标注者留一法。每题0、30、60、90、100分，整体为平均值；不是单一标准答案的严格匹配率。

## 对照设计
greedy与beam3只改变束宽；blank_image保持问题和图像尺寸，用均匀灰图替换原图，检测图像证据缺失后的行为。灰图属于分布外干预，其下降不等于严格的因果贡献，也不是重新训练的纯文本基线。每种设置独立预热一个批次。

配对统计：
```json
{
  "greedy": {
    "delta_vs_greedy_pp": 0.0,
    "ci95": [
      0.0,
      0.0
    ],
    "improved": 0,
    "unchanged": 500,
    "worsened": 0,
    "bootstrap": "3000 image-cluster paired resamples; seed 42"
  },
  "beam3": {
    "delta_vs_greedy_pp": -0.76,
    "ci95": [
      -1.653225806451613,
      0.07984431137724553
    ],
    "improved": 5,
    "unchanged": 482,
    "worsened": 13,
    "bootstrap": "3000 image-cluster paired resamples; seed 42"
  },
  "blank_image": {
    "delta_vs_greedy_pp": -38.2,
    "ci95": [
      -42.384911256776505,
      -33.9874749498998
    ],
    "improved": 20,
    "unchanged": 239,
    "worsened": 241,
    "bootstrap": "3000 image-cluster paired resamples; seed 42"
  }
}
```
按图像聚类bootstrap，95%区间若包含0，不宣称稳定改善。解码消融没有针对这些题的答案调参。

## 性能
| method      |   generation_seconds_per_question |   e2e_seconds_per_question |
|:------------|----------------------------------:|---------------------------:|
| greedy      |                            0.0135 |                     0.0269 |
| beam3       |                            0.0164 |                     0.0295 |
| blank_image |                            0.0126 |                     0.0234 |
单位为秒/题的批量摊销成本。Generation仅计同步GPU生成；E2E还包括图像读取、预处理和解码，不包括下载、模型加载和评分。不能称为单请求响应延迟。

## 错误案例
|   question_id | question                         | answer       | majority                     |
|--------------:|:---------------------------------|:-------------|:-----------------------------|
|       1903001 | What is in the lid?              | lid          | nothing                      |
|      34950010 | What color is the train?         | red and blue | red, yellow, orange and blue |
|      50125003 | How many tires are visible?      | 10           | 7                            |
|      71724004 | How many people are in the boat? | 2            | 3                            |
|     323951000 | Does he have a hat on?           | yes          | no                           |
|     483165001 | Is the man wearing a helmet?     | yes          | no                           |
按答案类别和题号固定选取前两个零分案例。多数票不一定是唯一真值，应结合per_question_results.csv中的十人答案分布解释。不能仅凭输出断言模型内部原因。

## 可用素材
figures/中7组PNG和SVG；case_images/保留12个左右可复核案例；predictions_*.jsonl是原始预测；submission_*.json为官方格式；per_question_results.csv是逐题评分；run_config、environment和image_hashes提供完整来源。

PPT建议：任务与架构 → 数据抽样 → 官方计分 → 总体与类别结果 → 解码成本 → 灰图对照 → 成功/失败案例 → 重叠限制与改进。引用见docs/REFERENCES.md。
