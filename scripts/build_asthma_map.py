# scripts/build_asthma_map.py
"""
建立 UK 資料集的「音檔名稱 → 受試者氣喘狀態」映射表。
執行一次即可，產出 asthma_map.csv 供 Dataset 使用。
"""
import pandas as pd
import os

AUDIO_META = './data/UK_COVID19/audio_metadata.csv'
PART_META = './data/UK_COVID19/participant_metadata.csv'
OUTPUT = './data/asthma_map.csv'


def main():
    print("讀取 audio_metadata.csv ...")
    audio_df = pd.read_csv(AUDIO_META)
    print(f"   共 {len(audio_df)} 筆音檔記錄")

    print("讀取 participant_metadata.csv ...")
    part_df = pd.read_csv(PART_META)
    print(f"   共 {len(part_df)} 位受試者")

    # ── 建立 participant → 臨床資訊 的字典 ──
    part_info = part_df.set_index('participant_identifier')[[
        'respiratory_condition_asthma',
        'symptom_shortness_of_breath',
        'symptom_cough_any',
        'smoker_status',
        'age',
        'gender',
    ]].to_dict('index')

    # ── 展開音檔：一個 participant 可能有多個音檔 ──
    rows = []

    for _, row in audio_df.iterrows():
        pid = row.get('participant_identifier')
        if pd.isna(pid):
            continue

        info = part_info.get(pid, {})
        asthma = info.get('respiratory_condition_asthma', 0.0)
        sob = info.get('symptom_shortness_of_breath', 0.0)  # shortness of breath
        cough_symptom = info.get('symptom_cough_any', 0.0)
        smoker = info.get('smoker_status', 0.0)
        age = info.get('age', 0.0)
        gender = info.get('gender', 0.0)

        # 處理 NaN
        asthma = 0.0 if pd.isna(asthma) else float(asthma)
        sob = 0.0 if pd.isna(sob) else float(sob)
        cough_symptom = 0.0 if pd.isna(cough_symptom) else float(cough_symptom)

        # 展開三種音檔類型
        for col, sound_type in [
            ('exhalation_file_name', 'exhale'),
            ('cough_file_name', 'cough'),
            ('three_cough_file_name', 'cough'),
        ]:
            fname = row.get(col)
            if pd.notna(fname) and str(fname).strip():
                stem = os.path.splitext(str(fname))[0]
                rows.append({
                    'file_stem': stem,
                    'sound_type': sound_type,
                    'participant_id': pid,
                    'asthma': asthma,
                    'shortness_of_breath': sob,
                    'cough_symptom': cough_symptom,
                    'smoker': smoker,
                    'age': age,
                    'gender': gender,
                })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT, index=False)

    # ── 統計 ──
    print(f"\n映射表已儲存：{OUTPUT}")
    print(f"   總音檔數：{len(result_df)}")
    print(f"   咳嗽：{(result_df['sound_type'] == 'cough').sum()}")
    print(f"   哈氣：{(result_df['sound_type'] == 'exhale').sum()}")
    print(f"\n氣喘分佈：")
    print(result_df['asthma'].value_counts().to_string())
    print(f"\n呼吸困難分佈：")
    print(result_df['shortness_of_breath'].value_counts().to_string())

    # 交叉統計
    print(f"\n氣喘 × 呼吸困難 交叉表：")
    cross = pd.crosstab(
        result_df['asthma'],
        result_df['shortness_of_breath'],
        rownames=['asthma'],
        colnames=['shortness_of_breath']
    )
    print(cross.to_string())


if __name__ == '__main__':
    main()