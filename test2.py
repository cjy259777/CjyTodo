import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def load_local_data(file_path):
    """加载本地CSV文件数据"""
    try:
        # 学生需要完善：使用pandas读取CSV文件
        data = pd.read_csv(file_path)
        print("本地数据加载成功！")
        print("数据前5行：")
        print(data.head())
        print(f"数据形状: {data.shape}")

        # 检查数据基本信息
        print("\n数据信息:")
        print(data.info())

        # 检查缺失值
        print("\n缺失值统计:")
        print(data.isnull().sum())

        # 假设最后一列是标签
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # 处理特征中的字符串数据（如果有）
        for i in range(X.shape[1]):
            if isinstance(X[0, i], str):
                le_feature = LabelEncoder()
                X[:, i] = le_feature.fit_transform(X[:, i])

        # 如果标签是字符串，转换为数字
        if y.dtype == 'object' or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_names = le.classes_
        else:
            target_names = np.unique(y)

        return X, y, target_names
    except Exception as e:
        print(f"本地数据加载失败: {e}")
        return None, None, None


def load_iris_data():
    """加载内置的Iris数据集"""
    # 学生需要完善：加载sklearn内置的iris数据集
    iris = load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    print("\nIris数据加载成功！")
    print("特征前5行：")
    print(pd.DataFrame(X, columns=iris.feature_names).head())

    return X, y, target_names

def preprocess_data(X_train, X_test):
    """标准化数据"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def split_dataset(X, y, test_size=0.2):
    """使用留出法划分数据集为训练集和测试集"""
    # 学生需要完善：使用train_test_split划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print(f"\n数据集划分完成:")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}")
    print(f"类别分布 - 测试集: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


def cross_validation(X, y, model_class, k=5):
    """使用k折交叉验证法评估模型稳定性"""
    # 学生需要完善：创建KFold对象
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        print(f"折 {fold + 1} 精度: {round(acc, 4)}")

    print(f"\n{k}折交叉验证结果:")
    print(f"各折精度: {[round(s, 4) for s in scores]}")
    print(f"平均精度: {round(np.mean(scores), 4)} ± {round(np.std(scores), 4)}")

    return np.mean(scores)


def bootstrap_method(X, y, model_class, n_iterations=100, random_state=42):
    """使用自助法评估模型性能"""
    np.random.seed(random_state)
    scores = []
    n_samples = X.shape[0]

    for i in range(n_iterations):
        # 学生需要完善：有放回地抽样创建训练集
        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True
        )
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]

        # 标准化数据
        scaler = StandardScaler()
        X_bootstrap = scaler.fit_transform(X_bootstrap)

        # 使用未被抽中的样本作为测试集
        oob_indices = [i for i in range(n_samples) if i not in indices]
        if len(oob_indices) > 0:  # 确保有测试样本
            X_test = X[oob_indices]
            y_test = y[oob_indices]
            X_test = scaler.transform(X_test)  # 使用相同的标准化器

            model = model_class()
            model.fit(X_bootstrap, y_bootstrap)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

    print(f"\n自助法评估结果 ({n_iterations}次迭代):")
    print(f"平均准确率: {round(np.mean(scores), 4)} ± {round(np.std(scores), 4)}")

    return np.mean(scores)


def evaluate_performance(y_true, y_pred, target_names):
    """计算并打印分类性能指标"""
    # 学生需要完善：计算各种评估指标
    error_rate = 1 - accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n模型性能评估:")
    print(f"错误率: {round(error_rate, 4)}")
    print(f"准确率: {round(accuracy, 4)}")
    print(f"精确率: {round(precision, 4)}")
    print(f"召回率: {round(recall, 4)}")
    print(f"F1分数: {round(f1, 4)}")

    # 打印每个类别的详细指标
    if len(target_names) > 2:
        print("\n各类别详细指标:")
        # 学生需要完善：计算每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i, name in enumerate(target_names):
            print(f"类别 {name}: 精确率={round(precision_per_class[i], 4)}, "
                  f"召回率={round(recall_per_class[i], 4)}, F1={round(f1_per_class[i], 4)}")


def main():
    print("请选择数据加载方式：")
    print("1. 加载本地CSV文件")
    print("2. 加载内置Iris数据集")

    choice = input("请输入选项(1/2)：")

    if choice == '1':
        file_path = input("请输入CSV文件路径：")
        X, y, target_names = load_local_data(file_path)
    elif choice == '2':
        X, y, target_names = load_iris_data()
    else:
        print("无效选项！")
        return

    if X is None or y is None:
        return

    # 数据集划分
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 数据标准化
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # 使用留出法评估模型
    print("\n=== 留出法评估 ===")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    evaluate_performance(y_test, y_pred, target_names)

    # 使用交叉验证法评估模型稳定性
    print("\n=== 交叉验证评估 ===")
    cross_validation(X, y, lambda: KNeighborsClassifier(n_neighbors=3), k=5)

    # 使用自助法评估模型
    print("\n=== 自助法评估 ===")
    bootstrap_method(X, y, lambda: KNeighborsClassifier(n_neighbors=3), n_iterations=100)


if __name__ == "__main__":
    main()