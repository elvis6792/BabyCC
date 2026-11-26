# BabyCC (Cough & Wheeze Acoustic Analyzer)

BabyCC 是一個基於深度學習（MobileNetV3）與 C++ 高效能聲學特徵提取器（Slaney Mel Filterbank + HPSS 軟遮罩）的離線嬰幼兒與成人呼吸音（氣喘喘鳴音 Wheeze / 咳嗽聲 Cough）實時監測系統。

---

## 核心特色 (Core Features)

1. **雙模式切換 (Dual Monitoring Modes)**
   - **`PHONE` 手機模式**：適用於距離嘴巴/胸口 30cm 外的居家 15 秒快速監測。
   - **`STETHO` 聽診器模式**：適用於連接電子聽診器或手機麥克風直接貼胸口進行長時連續監測。
2. **完全離線純單機運算 (100% On-Device Offline & Private)**
   - 使用 ONNX Runtime C++ API 與 NDK JNI，所有 AI 模型與 STFT 聲學特徵提取均在手機端實時計算，無需連接網路或外部伺服器。
3. **智慧咳嗽暴衝遮罩 (Smart Cough Masking)**
   - 二元雙頭神經網路（`wheeze_head` 與 `cough_head`）獨立預測，精準將咳嗽爆音與咽喉摩擦聲遮罩歸零，達至極低誤報率。
4. **雙層會話決策引擎 (Session-Level Clinical Decision Engine)**
   - 自動統計時間軸上的連續喘鳴段落與擊中次數，提供臨床等級的 `ALARM (氣喘發作)`、`WARNING` 與 `NORMAL` 分級預警。
5. **即時動態火紅頻譜圖 (Real-Time Fire Spectrogram)**
   - 包含自訂 Compose UI 頻譜元件，實時繪製音訊 Log-Mel 特徵圖。

---

## 專案架構 (Project Architecture)

```text
BabyCC/
├── app/                          # Android 應用程式模組 (Kotlin + Jetpack Compose)
│   ├── src/main/cpp/             # C++ 原生聲學特徵提取器與 JNI 介面
│   │   ├── FeatureExtractor.cpp   # Slaney Log-Mel & HPSS 諧波打擊音分離
│   │   └── RespiratoryAnalyzer.cpp# ONNX Runtime C++ 推理與即時音訊處理
│   ├── src/main/java/com/hsu/baby_cc/
│   │   ├── audio/                # 錄音器 (SafeAudioRecorder) 與 聲納硬體健檢 (SonarCalibrator)
│   │   ├── engine/               # 雙層會話決策引擎 (AcousticDecisionEngine)
│   │   └── ui/                   # ViewModels, Compose UI 與 實時頻譜繪製 (Fire Spectrogram)
│   └── src/main/assets/          # 打包的 ONNX 模型 (asthma_monitor_final_merged.onnx)
├── scripts/                      # Python 訓練、驗證與評估腳本
│   ├── train_final.py            # 主模型全監督 Focal Loss 訓練腳本
│   ├── infer_session.py          # 會話層滑窗推理驗證腳本
│   ├── config.py                 # 聲學與路徑配置檔
│   └── run_regression.sh         # 回歸驗收測試套件
├── best_asthma_monitor.pt        # PyTorch 最佳模型權重
├── asthma_monitor_final.onnx     # 導出的單一檔 ONNX 模型
└── README.md
```

---

## 快速開始 (Quick Start)

### 1. 直接下載安裝 APK（GitHub Releases）

* **系統需求**：Android 8.0（API 26）或以上版本
* **下載頁面**：前往本專案的 [GitHub Releases](https://github.com/elvis6792/BabyCC/releases) 下載最新版本的 `BabyCC.apk`。
* **安裝方式**：
  1. **手機直接安裝**：在 Android 手機下載 `BabyCC.apk` 後點擊開啟，依系統提示允許「安裝未知應用程式」即可完成安裝。
  2. **透過 ADB 安裝**（電腦連接手機並開啟 USB 偵錯）：
     ```bash
     adb install -r BabyCC.apk
     ```

---

### 2. 從原始碼編譯 Android App

使用 Android Studio 開啟本專案根目錄，或使用 Gradle 命令行建置：

```bash
# 建置 Debug APK
./gradlew app:assembleDebug
```

---

### 3. Python 訓練與評估環境配置

環境需求：Python 3.10+，PyTorch，librosa，onnxruntime

```bash
# 安裝依賴
pip install torch torchvision librosa soundfile onnxruntime scikit-learn pandas tqdm python-dotenv

# 執行模型推理測試
python scripts/infer_session.py --file app/src/main/assets/synth_wheeze_exhale_00008.wav --mode phone

# 執行全套回歸測試
bash scripts/run_regression.sh
```

---

## 醫療免責聲明 (Medical Disclaimer)

> **本專案僅供學術研究、技術展示與居家輔助觀察參考，不構成任何形式之醫療診斷、醫療建議、治療方案或醫療器材認證。**
>
> - 本應用程式與模型輸出之結果（包含 `ALARM`、`WARNING`、`NORMAL` 等分級提示）可能受環境噪音、收音距離、硬體麥克風差異或模型誤差影響，無法保證 100% 準確性，亦不得取代專業醫師的臨床聽診與診斷。
> - 若嬰幼兒或使用者出現呼吸急促、呼吸困難、胸凹、嘴唇發紫、持續劇烈咳嗽或疑似氣喘發作等症狀，**請立即就醫或尋求緊急醫療協助**，切勿因本系統之顯示結果而延誤治療。
> - 作者與貢獻者不對任何因使用或無法使用本軟體所導致之直接、間接或衍生性損害承擔責任。

---

## 授權協議 (License)

本專案原始碼採用 MIT 授權協議開源，詳細條款請參閱 [LICENSE](LICENSE) 檔案。

Copyright (c) 2026 HSU CHIH YUAN

### 第三方元件授權聲明 (Third-Party Notices)

本專案包含或依賴以下第三方開源元件，完整授權條款與版權聲明請參閱 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：
* **KissFFT** (`app/src/main/cpp/kiss_fft.*`)：BSD-3-Clause License — Copyright (c) 2003-2010, Mark Borgerding
* **ONNX Runtime** (`app/src/main/cpp/include/`, `app/src/main/cpp/Libs/*/libonnxruntime*.so`)：MIT License — Copyright (c) Microsoft Corporation
* **OpenCV** (`app/src/main/cpp/Libs/*/libopencv_java4.so`)：Apache License 2.0 — Copyright (c) OpenCV Foundation & Contributors

---

## 致謝 (Acknowledgements)

本專案的模型訓練使用到了以下公開開源與學術資料庫，特此致謝所有資料貢獻者與研究單位：
* **ICBHI 2017 Respiratory Sound Database**: 提供聽診器錄製之各類臨床異常呼吸音（Wheezes, Crackles）。
* **UK Audio (UK Respiratory Sound Database)**: 提供大量臨床與社群收集的呼吸音資料。
* **SprSound (Open-Source Pediatric Respiratory Sound Database)**: 提供專門針對兒童及嬰幼兒的優質臨床錄音資料。
* **CoughVID**: 提供大規模群眾外包的咳嗽聲學數據集。
