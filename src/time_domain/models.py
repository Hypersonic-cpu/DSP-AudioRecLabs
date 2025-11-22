"""
分类器模块：包含传统机器学习分类器和MLP神经网络
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ==================== 传统机器学习分类器 ====================

class TraditionalClassifier:
    """传统分类器的统一封装"""

    def __init__(self, classifier_type='knn', **kwargs):
        """
        Args:
            classifier_type: 分类器类型
                - 'knn': K近邻
                - 'naive_bayes': 朴素贝叶斯
                - 'decision_tree': 决策树
                - 'svm': 支持向量机
            **kwargs: 分类器的超参数
        """
        self.classifier_type = classifier_type

        if classifier_type == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 3)
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif classifier_type == 'naive_bayes':
            self.model = GaussianNB()

        elif classifier_type == 'decision_tree':
            max_depth = kwargs.get('max_depth', None)
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        elif classifier_type == 'svm':
            C = kwargs.get('C', 1.0)
            kernel = kwargs.get('kernel', 'rbf')
            self.model = SVC(C=C, kernel=kernel, random_state=42)

        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")

    def fit(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """预测"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': report,
            'confusion_matrix': cm
        }


# ==================== MLP神经网络 ====================

class MLPClassifier(nn.Module):
    """多层感知器神经网络"""

    def __init__(self, input_size, hidden_layers, num_classes, dropout=0.3):
        """
        Args:
            input_size: 输入特征维度
            hidden_layers: 隐藏层维度列表，如 [64, 32]
            num_classes: 类别数
            dropout: Dropout比例
        """
        super(MLPClassifier, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPTrainer:
    """MLP训练器"""

    def __init__(self, input_size, hidden_layers, num_classes,
                 learning_rate=0.001, epochs=100, batch_size=16,
                 device=None):
        """
        Args:
            input_size: 输入特征维度
            hidden_layers: 隐藏层维度列表
            num_classes: 类别数
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批大小
            device: 设备（cuda或cpu）
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 构建模型
        self.model = MLPClassifier(input_size, hidden_layers, num_classes).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练历史
        self.train_losses = []
        self.train_accuracies = []

    def fit(self, X_train, y_train, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            verbose: 是否打印训练过程
        """
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        # 创建数据加载器
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # 记录
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    def predict(self, X_test):
        """预测"""
        self.model.eval()

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()

    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': report,
            'confusion_matrix': cm,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies
        }


# ==================== 分类器工厂函数 ====================

def create_classifier(classifier_type, **kwargs):
    """
    创建分类器的工厂函数

    Args:
        classifier_type: 分类器类型
            - 'knn', 'naive_bayes', 'decision_tree', 'svm': 传统分类器
            - 'mlp': MLP神经网络
        **kwargs: 分类器参数

    Returns:
        分类器对象
    """
    if classifier_type in ['knn', 'naive_bayes', 'decision_tree', 'svm']:
        return TraditionalClassifier(classifier_type, **kwargs)

    elif classifier_type == 'mlp':
        return MLPTrainer(**kwargs)

    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
