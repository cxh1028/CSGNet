import numpy as np
import scipy.io

# 加载 .mat 文件
mat_data = scipy.io.loadmat('./results/results_paviaU_82.93.mat')

confusion_matrix = mat_data['results'][0][0][0]

# 获取准确率
accuracy = mat_data['results'][0][0][2][0][0]

# 获取F1分数
F1_scores = mat_data['results'][0][0][3]

# 获取Kappa系数
kappa = mat_data['results'][0][0][4][0][0]

# 获取预测标签和真实标签
predictions = mat_data['results'][0][0][5].flatten()
labels = mat_data['results'][0][0][6].flatten()

# 计算每个类别的准确率
class_accuracies = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

# 打印总体准确率和Kappa系数
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Kappa Coefficient: {kappa:.4f}")

# 获取类别名称，假设类别标签从0开始连续编号
class_names = np.unique(labels)
label_values = [str(i) for i in class_names]  # 将类别标签转换为字符串

# 打印每个类别的准确率
for i, class_name in enumerate(class_names):
    print(f"Accuracy of class {class_name}: {class_accuracies[i]:.4f}")

# 计算AA指标（Adjusted Accuracy）
AA = np.mean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
print(f"average accuracy (AA): {AA:.4f}")