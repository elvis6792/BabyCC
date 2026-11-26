import requests
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/api/test_client.py <wav檔案路徑>")
        sys.exit(1)

    wav_path = Path(sys.argv[1]).resolve()
    if not wav_path.exists():
        print(f"[Error] 找不到檔案: {wav_path}")
        sys.exit(1)

    url = "http://127.0.0.1:8000/api/v1/analyze"
    
    try:
        with open(wav_path, "rb") as f:
            files = {"file": (wav_path.name, f, "audio/wav")}
            print(f"發送音檔至 {url} ...")
            response = requests.post(url, files=files)

        if response.status_code == 200:
            print("\n[OK] API 測試成功，回傳結果：")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"\n[Error] API 回傳錯誤 (HTTP {response.status_code}):")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"[Error] 連線失敗。請確認 API 伺服器是否已在 {url} 啟動。")
    except Exception as e:
        print(f"[Error] 發生未知的客戶端錯誤: {str(e)}")

if __name__ == "__main__":
    main()