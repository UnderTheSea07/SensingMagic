"""
Material Classification using Magnetic Sensor Data (Random Forest + Enhanced Interpretability)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import pywt
from scipy import stats
from scipy.signal import welch
import shap
import joblib
import random
import warnings
from itertools import cycle
from sklearn.calibration import calibration_curve
import gc
import pickle
gc.collect()
# 忽略特定的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAMPLE_RATE = 100
WINDOW_SIZE = 50
STRIDE = 25

# Material name mapping
MATERIAL_MAPPING = {
    "杜邦纸": "Synthetic Nonwoven",
    "磨砂雨衣": "Matte-Finish Raincoat Fabric",
    "灯芯绒": "Corduroy Fabric",
    "针织罗纹布": "Ribbed Knit Fabric",
    "水洗皮": "Washed Faux Leather",
    "纱网布": "Mesh Fabric",
    "夜光反光布": "Luminous Reflective Fabric",
    "天鹅绒": "Velvet Fabric"
}

def safe_divide(a, b, fill_value=0):
    """安全除法，处理除以零的情况"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)
    return result

def extract_features(signal_xyz):
    """提取特征，处理异常值"""
    features = []
    for axis in range(3):
        signal = signal_xyz[:, axis]
        # 处理异常值
        signal = np.clip(signal, np.percentile(signal, 1), np.percentile(signal, 99))

        # 时域特征
        time_features = [
            np.mean(signal), np.std(signal), np.min(signal), np.max(signal), np.ptp(signal),
            stats.skew(signal), stats.kurtosis(signal), np.sum(signal ** 2),
            np.sum(np.abs(np.diff(signal))), np.sum(np.diff(signal) ** 2),
            np.sum(np.diff(np.signbit(signal))), np.percentile(signal, 25),
            np.percentile(signal, 75), np.median(signal), np.mean(np.abs(signal))
        ]

        # 频域特征
        try:
            f, psd = welch(signal, fs=SAMPLE_RATE, nperseg=min(256, len(signal)))
            total_power = np.sum(psd)
            freq_features = [
                total_power,
                np.sum(psd[f < 10]),
                np.sum(psd[(f >= 10) & (f < 50)]),
                np.sum(psd[f >= 50]),
                safe_divide(np.sum(psd[f < 10]), total_power),
                safe_divide(np.sum(psd[(f >= 10) & (f < 50)]), total_power),
                safe_divide(np.sum(psd[f >= 50]), total_power),
                f[np.argmax(psd)],
                -np.sum(psd * np.log2(psd + 1e-10)),
                np.mean(psd),
                np.std(psd),
                np.max(psd)
            ]
        except:
            freq_features = [0] * 12

        # 小波特征
        try:
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)
            level = min(2, max_level)
            coeffs = pywt.wavedec(signal, 'db4', level=level)
            wavelet_features = []
            for coeff in coeffs:
                wavelet_features.extend([
                    np.mean(coeff), np.std(coeff), np.min(coeff), np.max(coeff), np.sum(coeff ** 2)
                ])
            wavelet_features = wavelet_features[:15] if len(wavelet_features) >= 15 else wavelet_features + [0] * (
                    15 - len(wavelet_features))
        except:
            wavelet_features = [0] * 15

        features.extend(time_features + freq_features + wavelet_features)

    return np.array(features)

def prepare_dataset(base_path):
    try:
        with open("./prepare_data.pkl", "rb") as f:
            X, y, le, non_zero_var_mask = pickle.load(f)
            return X, y, le, non_zero_var_mask
    except:
        pass
    """准备数据集，处理异常值"""
    materials = {
        "杜邦纸": "杜邦纸",
        "磨砂雨衣": "muoshayuyi",
        "灯芯绒": "dengxinrong",
        "针织罗纹布": "luowenbu",
        "水洗皮": "水洗皮",
        "纱网布": "shawang",
        "夜光反光布": "fanguangbu",
        "天鹅绒": "tianerong"
    }

    all_features, all_labels = [], []
    for material_name, material_code in materials.items():
        pattern = f"*_{material_code}_*.csv"
        files = glob.glob(str(Path(base_path) / f'**/{pattern}'), recursive=True)
        if not files:
            print(f"警告: 未找到材料 {material_name} 的文件")
            continue

        for file in files:
            try:
                df = pd.read_csv(file)
                if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                    continue

                data = df[['X', 'Y', 'Z']].values
                # 处理异常值
                for col in range(3):
                    data[:, col] = np.clip(
                        data[:, col],
                        np.percentile(data[:, col], 1),
                        np.percentile(data[:, col], 99)
                    )

                for i in range(0, len(data) - WINDOW_SIZE, STRIDE):
                    window = data[i:i + WINDOW_SIZE]
                    features = extract_features(window)
                    all_features.append(features)
                    all_labels.append(material_name)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")

    if not all_features:
        raise ValueError("没有提取到任何特征")

    X = np.array(all_features)
    y = np.array(all_labels)

    # 移除零方差特征
    non_zero_var_mask = np.var(X, axis=0) > 0
    X = X[:, non_zero_var_mask]

    le = LabelEncoder()
    y = le.fit_transform(y)
    with open("prepare_data.pkl","wb") as f:
        pickle.dump((X, y, le, non_zero_var_mask),f)
    return X, y, le, non_zero_var_mask

def prepare_train_dataset(base_path):
    try:
        with open("./prepare_train_data.pkl", "rb") as f:
            train_X, train_y, test_X, test_y = pickle.load(f)
            return train_X, train_y, test_X, test_y
    except:
        pass
    """准备数据集，处理异常值"""
    materials = {
        "杜邦纸": "杜邦纸",
        "磨砂雨衣": "muoshayuyi",
        "灯芯绒": "dengxinrong",
        "针织罗纹布": "luowenbu",
        "水洗皮": "水洗皮",
        "纱网布": "shawang",
        "夜光反光布": "fanguangbu",
        "天鹅绒": "tianerong"
    }

    train_features, train_labels = [], []
    test_features, test_labels = [], []
    for material_name, material_code in materials.items():
        pattern = f"*_{material_code}_*.csv"
        files = glob.glob(str(Path(base_path) / f'**/{pattern}'), recursive=True)
        if not files:
            print(f"警告: 未找到材料 {material_name} 的文件")
            continue

        for file in files:
            try:
                df = pd.read_csv(file)
                if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                    continue

                data = df[['X', 'Y', 'Z']].values
                # 处理异常值
                for col in range(3):
                    data[:, col] = np.clip(
                        data[:, col],
                        np.percentile(data[:, col], 1),
                        np.percentile(data[:, col], 99)
                    )
                train_data = data[:int(len(data) * 0.8)]
                test_data = data[int(len(data) * 0.8):]

                for i in range(0, len(train_data) - WINDOW_SIZE, STRIDE):
                    window = data[i:i + WINDOW_SIZE]
                    features = extract_features(window)
                    train_features.append(features)
                    train_labels.append(material_name)
                for i in range(0, len(test_data) - WINDOW_SIZE, STRIDE):
                    window = data[i:i + WINDOW_SIZE]
                    features = extract_features(window)
                    test_features.append(features)
                    test_labels.append(material_name)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")

    if not train_features:
        raise ValueError("没有提取到任何特征")

    train_X = np.array(train_features)
    train_y = np.array(train_labels)
    test_X = np.array(test_features)
    test_y = np.array(test_labels)

    # 移除零方差特征
    non_zero_var_mask = np.var(train_X, axis=0) > 0
    train_X = train_X[:, non_zero_var_mask]
    non_zero_var_mask = np.var(test_X, axis=0) > 0
    test_X = test_X[:, non_zero_var_mask]

    le = LabelEncoder()
    train_y = le.fit_transform(train_y)
    test_y = le.fit_transform(test_y)
    with open("prepare_train_data.pkl","wb") as f:
        pickle.dump((train_X, train_y, test_X, test_y),f)
    return train_X, train_y, test_X, test_y


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    # Convert class names to English
    english_classes = [MATERIAL_MAPPING.get(cls, cls) for cls in classes]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=english_classes, yticklabels=english_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(clf, X_test, y_test, classes):
    """Plot ROC curves"""
    plt.figure(figsize=(10, 8))

    # Get prediction probabilities for each class
    y_score = clf.predict_proba(X_test)

    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Convert class names to English
    english_classes = [MATERIAL_MAPPING.get(cls, cls) for cls in classes]

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{english_classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Classes')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(clf, feature_names):
    """Plot feature importance"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(14, 6))
    plt.title("Feature Importance (Top 20)")
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_correlation(X, feature_names):
    """Plot feature correlation heatmap"""
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curve(clf, X, y):
    """Plot learning curve"""
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_probability(clf, X_test, y_test, classes):
    """Plot prediction probability distribution"""
    y_pred_proba = clf.predict_proba(X_test)

    plt.figure(figsize=(12, 6))
    # Convert class names to English
    english_classes = [MATERIAL_MAPPING.get(cls, cls) for cls in classes]

    for i in range(len(classes)):
        plt.hist(y_pred_proba[y_test == i, i], bins=50, alpha=0.5, label=english_classes[i])

    plt.xlabel('Prediction Probability')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Probability Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_probability.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_values(clf, X_test, feature_names):
    """Plot SHAP values analysis"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(14, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_values.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_over_time(clf, X, y, feature_names):
    """Plot feature importance over time"""
    # Split data by time
    n_splits = 5
    split_size = len(X) // n_splits

    importance_over_time = []
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)

        # Train model
        temp_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_clf.fit(X[start_idx:end_idx], y[start_idx:end_idx])

        # Get feature importance
        importance_over_time.append(temp_clf.feature_importances_)

    # Plot feature importance over time
    plt.figure(figsize=(14, 6))
    top_n = 10
    top_indices = np.argsort(clf.feature_importances_)[-top_n:]

    for idx in top_indices:
        plt.plot(range(n_splits), [imp[idx] for imp in importance_over_time],
                 label=feature_names[idx], marker='o')

    plt.xlabel('Time Segment')
    plt.ylabel('Feature Importance')
    plt.title('Top 10 Feature Importance Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('feature_importance_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_path = "."
    X, y, le, feature_mask = prepare_dataset(base_path)
    print(f"数据集大小: {X.shape}")

    # 生成特征名
    all_feature_names = []
    for axis in ['X', 'Y', 'Z']:
        all_feature_names.extend([f'{axis}_t{i}' for i in range(15)])
        all_feature_names.extend([f'{axis}_f{i}' for i in range(12)])
        all_feature_names.extend([f'{axis}_w{i}' for i in range(15)])

    # 只保留非零方差特征的特征名
    feature_names = [name for i, name in enumerate(all_feature_names) if feature_mask[i]]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train,  y_train, X_test, y_test = prepare_train_dataset(base_path)

    # 训练随机森林
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 保存模型和转换器
    joblib.dump(clf, 'material_classifier.joblib')
    joblib.dump(scaler, 'material_scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(feature_mask, 'feature_mask.joblib')

    # 评估
    y_pred = clf.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 生成所有可视化图表
    plot_confusion_matrix(y_test, y_pred, le.classes_)
    plot_roc_curves(clf, X_test, y_test, le.classes_)
    plot_feature_importance(clf, feature_names)
    plot_feature_correlation(X_scaled, feature_names)
    plot_prediction_probability(clf, X_test, y_test, le.classes_)
    plot_shap_values(clf, X_test, feature_names)
    plot_feature_importance_over_time(clf, X_scaled, y, feature_names)
    plot_learning_curve(clf, X_scaled, y)

    print("\n所有分析图表已保存。")
    print(f"使用的特征数量: {len(feature_names)}")


if __name__ == "__main__":
    main()
