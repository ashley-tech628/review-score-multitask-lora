# 整理与延伸交付

项目署名：团队项目，Ashley Liu (Xinying Liu) 担任组长、承担主要工作；不声称全部独立完成。具体队友姓名与分工尚待补充。

这次新增了有实际结果的工程延伸：缺失标签掩码、商品分组拆分、文本去重、岭回归与均值基线、商品聚类 bootstrap、不含原始评论的评估记录、模型导出与恢复预测、合成数据演示及测试。

新实验：本地数据前 20,000 条，测试集 3,559 条 / 76 个未见商品；岭回归 macro MSE 0.01615，均值基线 0.01840。只能描述为这份前缀样本上的结果，不能当作 LoRA 提升或全量成绩。

LoRA 延伸代码已补回归头的训练与保存、缺失标签 loss、验证集选模型、加载后一致性检查和预测入口；随后已安装依赖，13 项测试全部通过；2,000 条记录的一轮 CPU LoRA 训练、保存、加载一致性及预测已完成。测试集仅 46 条，结果用于流程验证。

可用于简历的表述：

> Led a team project on multi-aspect review scoring with DistilBERT and LoRA; extended the pipeline with product-held-out evaluation, missing-label masking, checkpoint auditing, and reproducible baseline inference.

其中 extension 是此次 AI 辅助新增工作，应在你理解、运行并能解释实现后用于面试。原始 checkpoint 缺少回归头，不能声称已恢复原 LoRA 预测。

本地已整理，尚未创建 GitHub 仓库或推送。公开前补齐团队署名与共享代码发布范围。建议仓库名 review-score-multitask-lora。
