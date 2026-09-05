# judge-redteam 中文导读

这个项目研究的是：**当 AI 给答案打分时，会不会被排版、语气、篇幅或看起来很有道理的错误推理带偏？**

你可以把它理解成一套“检查 AI 裁判是否公正”的实验工具。它先准备有明确正确答案的题目，再改变答案的呈现方式，比较裁判改判的方向，并用不做干预的重复评判估计裁判本身的随机波动。

目前仓库中的结果来自人为设置偏差的**模拟裁判**。这说明工具能够走通实验和报告流程；尚未完成真实模型实验，也没有关于 GPT、Claude 或其他真实模型的研究结论。

## 怎么读图

![模拟裁判的方向性效果量和置信区间。横线表示估计的不确定性，H5 长度对照以灰色菱形区分。](figures/demo/effect_sizes.svg)

点越靠右，表示这一组数据里“从正确改判成错误”相对“从错误改判成正确”越突出。横线表示不确定性。各区间未经多重比较校正，不能单凭横线替代正式统计判断；不同实验轴也不能直接当作真实模型的能力排名。

![正确改错与错误改对的比例，以及不做干预时的改错参考值。所有结果来自模拟裁判。](figures/demo/directional_flips.svg)

橙色表示改错，绿色表示改对；菱形是同样的题目、同样的呈现重复评判时的改错参考值。这个参考帮助判断变化是否超过裁判自身的波动。实际判断还需要项目中的按题目配对的统计检验。

## 零费用试跑

```bash
git clone https://github.com/richardgshen-hub/judge-redteam
cd judge-redteam
python -m pip install -e ".[viz]"
python scripts/demo.py --figures
```

需要 Python 3.10 或以上。这个命令只运行模拟裁判，不需要 API 密钥。
报告位于 `results/demo_report.md`，图片与配套数据位于 `docs/figures/demo/`。
想只看文字报告，可以省略安装可视化依赖，直接执行 `python scripts/demo.py`。

[完整英文说明](../README.md) · [示例报告](../results/demo_report.md) · [图表生成与口径](FIGURES.md) · [图表原始数值 CSV](figures/demo/axis_results.csv)
