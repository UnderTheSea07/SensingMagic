"""
Material Classification using Magnetic Sensor Data (Random Forest + Interpretability)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

# 忽略特定的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SAMPLE_RATE = 100
WINDOW_SIZE = 50
STRIDE = 25

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
            stats.skew(signal), stats.kurtosis(signal), np.sum(signal**2),
            np.sum(np.abs(np.diff(signal))), np.sum(np.diff(signal)**2),
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
            freq_features = [0]*12
            
        # 小波特征
        try:
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)
            level = min(2, max_level)
            coeffs = pywt.wavedec(signal, 'db4', level=level)
            wavelet_features = []
            for coeff in coeffs:
                wavelet_features.extend([
                    np.mean(coeff), np.std(coeff), np.min(coeff), np.max(coeff), np.sum(coeff**2)
                ])
            wavelet_features = wavelet_features[:15] if len(wavelet_features) >= 15 else wavelet_features + [0]*(15-len(wavelet_features))
        except:
            wavelet_features = [0]*15
            
        features.extend(time_features + freq_features + wavelet_features)
    
    return np.array(features)

def prepare_dataset(base_path):
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
    
    return X, y, le, non_zero_var_mask

def plot_feature_importance(clf, feature_names):
    """绘制特征重要性图"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(14, 6))
    plt.title("特征重要性 (前20)")
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300)
    plt.close()

def plot_feature_correlation(X, feature_names):
    """绘制特征相关性热力图，处理异常值"""
    # 使用pandas计算相关性，它会自动处理异常值
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig('rf_feature_correlation.png', dpi=300)
    plt.close()

def plot_shap_summary(clf, X, feature_names):
    """绘制SHAP值分析图"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(14, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('rf_shap_summary.png', dpi=300)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
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
    joblib.dump(clf, 'rf_material_classifier.joblib')
    joblib.dump(scaler, 'rf_material_scaler.joblib')
    joblib.dump(le, 'rf_label_encoder.joblib')
    joblib.dump(feature_mask, 'rf_feature_mask.joblib')
    
    # 评估
    y_pred = clf.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png', dpi=300)
    plt.close()
    
    # 可解释性分析
    plot_feature_importance(clf, feature_names)
    plot_feature_correlation(X_scaled, feature_names)
    plot_shap_summary(clf, X_test, feature_names)
    
    print("\n特征重要性、相关性、SHAP分析图已保存。")
    print(f"使用的特征数量: {len(feature_names)}")

if __name__ == "__main__":
    main() 