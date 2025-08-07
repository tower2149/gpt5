
#!/usr/bin/env python3
import argparse, json, os, sys, time
import requests
from pathlib import Path

API_URL = "https://api.openai.com/v1/responses"
MODEL   = "gpt-5"  # gpt-5 / gpt-5-mini 等も可

def load_history(dirpath, session):
    path = Path(dirpath) / f"{session}.json"
    if not path.exists(): return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(dirpath, session, history):
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    path = Path(dirpath) / f"{session}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def make_payload(history, prompt, use_search=True):
    # Responses APIは input に履歴＋新規発話を渡せる
    msgs = history + [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL,
        "input": msgs,
    }
    if use_search:
        payload["tools"] = [{"type": "web_search"}]
    return payload

def call_api(payload, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()

def extract_text(resp_json):
    # SDKと同様のヘルパが無いので素直に辿る（output_textがあれば最優先）
    if "output_text" in resp_json and resp_json["output_text"]:
        return resp_json["output_text"]
    # 汎用フォールバック
    try:
        items = resp_json.get("output", []) or resp_json.get("response", {}).get("output", [])
        for it in items:
            if it.get("type") == "message":
                for c in it.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
        # だめなら最終手段
        return json.dumps(resp_json)
    except Exception:
        return json.dumps(resp_json)

def main():
    p = argparse.ArgumentParser(description="GPT-5 CLI with Web Search + history")
    p.add_argument("prompt", nargs="*", help="質問/指示。未指定なら標準入力から")
    p.add_argument("--session", "-s", default="default", help="履歴ファイル名")
    p.add_argument("--history-dir", default=os.path.expanduser("~/.gpt5_cli"))
    p.add_argument("--no-search", action="store_true", help="Web検索を無効化")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY が未設定です。export OPENAI_API_KEY=... を実行してください。", file=sys.stderr)
        sys.exit(1)

    # 入力取得
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("空のプロンプトです。", file=sys.stderr); sys.exit(1)

    # 履歴読込→API→結果→履歴保存
    history = load_history(args.history_dir, args.session)
    payload = make_payload(history, prompt, use_search=(not args.no_search))
    resp = call_api(payload, api_key)
    text = extract_text(resp)

    # 出力
    print(text.strip())

    # 履歴更新（ユーザー発話＋アシスタント返答）
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": text})
    save_history(args.history_dir, args.session, history)

if __name__ == "__main__":
    main()
