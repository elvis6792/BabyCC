import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config

def main():
    print("==========================================================")
    print("啟動多模態資料庫建置管線 (Multi-modal Dataset Builder)")
    print("==========================================================")

    # 1. 載入與清理基礎標籤
    df_part = pd.read_csv('./data/UK_COVID19/participant_metadata.csv')
    
    if 'respiratory_condition_asthma' not in df_part.columns:
        raise ValueError("[致命錯誤] 找不到 respiratory_condition_asthma 欄位")
        
    df_part['asthma_label'] = df_part['respiratory_condition_asthma'].apply(
        lambda x: 1.0 if str(x).lower() in ['true', '1', 'yes', '1.0'] else 0.0
    )

    # 目前暫無純數值特徵，保留 numeric_features 陣列以利未來擴充 (如 BMI, 體重等)
    numeric_features = []
    categorical_features = ['age', 'gender', 'smoker_status', 'healthcare_worker']
    
    # 防呆：確認特徵欄位是否存在
    missing_cols = [c for c in numeric_features + categorical_features if c not in df_part.columns]
    if missing_cols:
        print(f"[Warning] 警告：找不到以下欄位，將自特徵中移除: {missing_cols}")
        numeric_features = [c for c in numeric_features if c in df_part.columns]
        categorical_features = [c for c in categorical_features if c in df_part.columns]

    df_filtered = df_part[['participant_identifier', 'asthma_label'] + numeric_features + categorical_features].copy()
    
    # 剔除完全沒有特徵或 PID 為空的無效數據
    df_filtered = df_filtered.dropna(subset=['participant_identifier'])
    
    print(f"有效病患總數: {len(df_filtered)}")
    print(f"氣喘正樣本比例: {df_filtered['asthma_label'].mean():.2%}")

    # 2. 嚴格分層切分 (Stratified Split) - 解決 1:8 不平衡帶來的分佈漂移
    print("\n執行分層切分 (Train: 80%, Val: 20%)...")
    X = df_filtered.drop(columns=['asthma_label'])
    y = df_filtered['asthma_label']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 建立 Scikit-learn Pipeline 防止 Data Leakage
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    print("擬合特徵轉換器 (僅限 Train Set 以防止 Data Leakage)...")
    train_tabular_np = preprocessor.fit_transform(X_train)
    val_tabular_np = preprocessor.transform(X_val)

    # 4. 轉換為 PyTorch 字典並持久化
    # 結構: PID -> {'tabular': Tensor, 'label': float}
    train_dict = {}
    for i, pid in enumerate(X_train['participant_identifier']):
        train_dict[pid] = {
            'tabular': torch.tensor(train_tabular_np[i], dtype=torch.float32),
            'label': torch.tensor([y_train.iloc[i]], dtype=torch.float32)
        }

    val_dict = {}
    for i, pid in enumerate(X_val['participant_identifier']):
        val_dict[pid] = {
            'tabular': torch.tensor(val_tabular_np[i], dtype=torch.float32),
            'label': torch.tensor([y_val.iloc[i]], dtype=torch.float32)
        }

    output_dir = Path(Config.PROJECT_ROOT) / "processed_fusion_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(train_dict, output_dir / "tabular_train.pt")
    torch.save(val_dict, output_dir / "tabular_val.pt")
    
    # 儲存 preprocessor 以供未來 Inference 使用
    import joblib
    joblib.dump(preprocessor, output_dir / "tabular_preprocessor.joblib")

    print(f"\n轉換完成！")
    print(f"- Tabular Feature 維度: {train_tabular_np.shape[1]}")
    print(f"- Train set 數量: {len(train_dict)}")
    print(f"- Val set 數量: {len(val_dict)}")
    print(f"檔案儲存於: {output_dir}")

if __name__ == "__main__":
    main()