import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 真实样本分布（假设所有样本均为Label2）
true_samples = 7290  # Label2的真实样本数

# 构建混淆矩阵（仅Label2有真实样本）
conf_matrix = np.array([
    # Label1  Label2  Label3  Label4
    [    0,       0,       0,       0],  # Label1（无样本）
    [  480,    5838,     400,     592],  # Label2（真实样本7290，正确率80%）
    [    0,       0,       0,       0],  # Label3（无样本）
    [    0,       0,       0,       0]   # Label4（无样本）
])

# 标签名称
labels = ["Label1", "Label2", "Label3", "Label4"]

# 可视化设置
fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=labels
)

# 绘制混淆矩阵
disp.plot(
    cmap="Blues",
    ax=ax,
    values_format="d",
    colorbar=False,
    im_kw={"vmin": 0, "vmax": 6000}
)

# 自定义样式
plt.title("Confusion Matrix (system )", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整文字颜色
for text in disp.text_.ravel():
    value = int(text.get_text())
    text.set_color("white" if value > 3000 else "black")


plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 节点列表（区域1包含7个节点）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 混淆矩阵数据（单位：样本数）
cm = np.array([
    # label1  label2  label3  label4
    [4820,    320,     285,     245],  # 真实label1 (5670)
    [ 190,   4835,     255,     390],  # 真实label2 (5670)
    [ 305,    210,    4815,     340],  # 真实label3 (5670)
    [ 280,    365,     315,    5520]   # 真实label4 (6480)
])

# 标签名称
labels = ["label1", "label2", "label3", "label4"]

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", ax=ax, values_format="d")

# 调整显示效果
plt.title("Area_level ", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()