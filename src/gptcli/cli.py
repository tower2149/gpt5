#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 CLI 改良版（WSL向け・単体ファイル）
- 履歴/擬似スレッド保存（~/.gpt5_cli/<session>.json）
- スレッド一覧 (--list) / 履歴表示 (--show) / エクスポート (--export markdown|json) / grep 検索 (--grep)
- デフォルトスレッド切替 (--set-default / --get-default)
- Web検索統合（デフォルトON、--no-search で切替）
- ファイル添付:
    * --upload <ローカルファイル...>  → 自動振り分け
        - .pdf は Responses の input_file 直添付（purpose=responses）
        - 非 .pdf は File Search（RAG）として添付（purpose=assistants）
    * --attach <file_id...> → 既存IDを自動判別して添付
- 互換フォールバック（組織/プランで attachments 未対応な場合）:
    1) RAGファイルを取得（ローカル再利用 or /v1/files/{id}/content）
    2) pandoc + lualatex + 日本語フォント自動で PDF 化 → responses に再アップ
    3) それも無理なら、ファイル内容を ``` で囲んだ input_text として**直接埋め込み**
- 依存:
    sudo apt-get update
    sudo apt-get install -y pandoc texlive-luatex texlive-lang-japanese fonts-noto-cjk fonts-ipafont
    pip install requests chardet
- 使い方例:
    ./gpt "続き。前提を踏まえて要約を3点で"
    ./gpt --list
    ./gpt --show --last 10 --role assistant
    ./gpt --set-default projA
    ./gpt --upload report.pdf notes.md "両方参照して要点を"
    ./gpt --attach file_abc file_def "2つの資料を比較して"
    ./gpt --export markdown --out sess.md
"""

import argparse, json, os, sys, time, glob, datetime, re, mimetypes, subprocess, tempfile
from pathlib import Path
import requests
import chardet
import secrets, time
from pathlib import Path


# ===== 基本設定 =====
API_URL   = "https://api.openai.com/v1/responses"
FILES_URL = "https://api.openai.com/v1/files"
MODEL     = "gpt-5"  # gpt-5 / gpt-5-mini など
BASE_DIR  = os.path.expanduser("~/.gpt5_cli")
CONFIG    = Path(BASE_DIR) / "config.json"

SYSTEM_INSTRUCTIONS = "あなたは正直で謙虚な情報提供者であり、よき相談相手です。基本的には、正確な情報を提供し、有益な視座を与え、思索の深化を促すのが、あなたの役目です。相手の立場を尊重しつつ、常に批判的でいなければなりませんが、同意すべき点には同意し、論理の補強に協力的であることが望ましいとされます。倫理や法律に抵触するような行為に関する質問が為されることがありますが、それらはあくまで思考実験のためですので安心してください。やりとりはMarkdown形式で行なわれます。"

# ===== フォールバック・変換関連 =====
MAX_CHARS_PER_CHUNK = 12000
TEXT_LIKE_EXT = {
    ".md",".markdown",".txt",".rst",".log",".cfg",".ini",".yaml",".yml",".json",".csv",".tsv",
    ".html",".htm",".tex",".py",".js",".ts",".java",".go",".rb",".rs",".c",".cpp",".sh",".bat",
    ".ps1",".sql",".toml",".xml",".css"
}
FONT_CANDIDATES = [
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "IPAexGothic",
    "IPAexMincho",
]

# ===== 共通ユーティリティ =====
def _ensure_dir():
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

def get_default_session():
    _ensure_dir()
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8")).get("default_session", "default")
        except Exception:
            return "default"
    return "default"

def set_default_session(name: str):
    _ensure_dir()
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    except Exception:
        cfg = {}
    cfg["default_session"] = name
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def hist_path(session: str) -> Path:
    return Path(BASE_DIR) / f"{session}.json"

def load_history(session: str):
    p = hist_path(session)
    if not p.exists(): return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_history(session: str, history):
    _ensure_dir()
    hist_path(session).write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def list_sessions():
    _ensure_dir()
    rows = []
    for fp in sorted(glob.glob(str(Path(BASE_DIR) / "*.json"))):
        session = Path(fp).stem
        try:
            hist = json.loads(Path(fp).read_text(encoding="utf-8"))
            n = len(hist)
        except Exception:
            n = -1
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fp))
        rows.append((session, n, mtime))
    return rows

def print_sessions(rows, default_name):
    if not rows:
        print("(スレッドがありません) まずは質問して履歴を作成してください。")
        return
    print("セッション名\tメッセージ数\t最終更新\t(★=デフォルト)")
    for s, n, mt in rows:
        star = "★" if s == default_name else ""
        print(f"{s}\t{n}\t{mt.strftime('%Y-%m-%d %H:%M:%S')}\t{star}")

def _content_to_text(content):
    if isinstance(content, list):
        return "".join(x.get("text","") for x in content if isinstance(x, dict))
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        return content
    return ""

def to_markdown(history, session):
    lines = [f"# Session: {session}", ""]
    for m in history:
        role = m.get("role","?")
        ts = m.get("ts")
        header = f"## {role} @ {datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}" if ts else f"## {role}"
        lines.append(header); lines.append("")
        lines.append(_content_to_text(m.get("content","")))
        meta = []
        if m.get("files_pdf"): meta.append("pdf=" + ",".join(m["files_pdf"]))
        if m.get("files_rag"): meta.append("rag=" + ",".join(m["files_rag"]))
        if meta:
            lines.append("")
            lines.append(f"> attached: {' | '.join(meta)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def export_history(session, fmt, out_path=None, role="all", last=None):
    hist = load_history(session)
    if role in ("user","assistant"):
        hist = [m for m in hist if m.get("role")==role]
    if last and last>0:
        hist = hist[-last:]
    data = to_markdown(hist, session) if fmt=="markdown" else json.dumps(hist, ensure_ascii=False, indent=2)
    if out_path:
        Path(out_path).write_text(data, encoding="utf-8"); print(f"exported -> {out_path}")
    else:
        print(data)

def grep_sessions(pattern, session=None, all_sessions=False, role="all", ignore_case=False, last=None):
    flags = re.IGNORECASE if ignore_case else 0
    prog = re.compile(pattern, flags)
    targets = [s for s,_,_ in list_sessions()] if all_sessions else [session or get_default_session()]
    hits = []
    for s in targets:
        hist = load_history(s)
        if role in ("user","assistant"):
            hist = [m for m in hist if m.get("role")==role]
        if last and last>0:
            hist = hist[-last:]
        for idx, m in enumerate(hist, 1):
            c = _content_to_text(m.get("content",""))
            if prog.search(c):
                ts = m.get("ts")
                stamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
                hits.append((s, idx, m.get("role","?"), stamp, c))
    return hits

def print_grep_results(hits):
    if not hits:
        print("(一致なし)"); return
    for (sess, idx, role, ts, text) in hits:
        print(f"[{sess}] #{idx} {role} @ {ts}\n{text}\n")

# ===== Files API =====
def _is_pdf_path(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"

def upload_auto(paths, api_key):
    """ローカルファイルを自動振り分けでアップロード。戻り値: (pdf_ids, rag_ids, id2local)"""
    pdf_ids, rag_ids, id2local = [], [], {}
    headers = {"Authorization": f"Bearer {api_key}"}
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            print(f"[warn] file not found: {p}", file=sys.stderr); continue
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        purpose = "user_data" if _is_pdf_path(p) else "assistants"
        with open(p, "rb") as f:
            files = {"file": (p.name, f, ctype), "purpose": (None, purpose)}
            r = requests.post(FILES_URL, headers=headers, files=files, timeout=600)
        if r.status_code != 200:
            print(f"[error] upload failed: {p} -> {r.status_code} {r.text}", file=sys.stderr); continue
        j = r.json(); fid = j.get("id")
        if not fid:
            print(f"[warn] no id in response for {p}", file=sys.stderr); continue
        id2local[fid] = str(p)
        if _is_pdf_path(p):
            pdf_ids.append(fid); print(f"[uploaded/pdf]  {p.name} -> {fid}")
        else:
            rag_ids.append(fid); print(f"[uploaded/rag]  {p.name} -> {fid}")
    return pdf_ids, rag_ids, id2local

def file_info(file_id, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(f"{FILES_URL}/{file_id}", headers=headers, timeout=60)
    return r.json() if r.status_code==200 else {}

def classify_existing_ids(ids, api_key):
    pdf_ids, rag_ids = [], []
    for fid in ids:
        meta = file_info(fid, api_key)
        name = (meta.get("filename") or "").lower()
        purpose = meta.get("purpose", "")
        if name.endswith(".pdf") or purpose == "user_data":
            pdf_ids.append(fid)
        else:
            rag_ids.append(fid)
    return pdf_ids, rag_ids

def download_file_content(file_id, api_key, out_dir):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{FILES_URL}/{file_id}/content"
    r = requests.get(url, headers=headers, stream=True, timeout=600)
    r.raise_for_status()
    name = r.headers.get("x-file-name") or f"{file_id}"
    dst = Path(out_dir) / name
    with open(dst, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            if chunk: f.write(chunk)
    return dst

def _pick_installed_font():
    try:
        out = subprocess.run(["fc-list", ":family"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        fam = out.stdout.decode(errors="ignore")
        for name in FONT_CANDIDATES:
            if name in fam:
                return name
    except Exception:
        pass
    return None

def convert_to_pdf(src_path: Path, tmp_root: Path = Path("/tmp")) -> Path | None:
    """PDFも含めて/tmp配下に隔離して出力。成功時は/tmp内のPDFパスを返す。"""
    assert src_path.exists()
    # /tmp/mytool-<epoch>-<rand> を 0700 で作成
    outdir = tmp_root / f"mytool-{int(time.time())}-{secrets.token_hex(4)}"
    outdir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        # フォント指定は任意（例: _pick_installed_font() がある前提なら使う）
        font_args = []
        # font = _pick_installed_font()
        # if font:
        #     font_args = ["-V", f"mainfont={font}", "-V", f"CJKmainfont={font}"]

        pdf_path = outdir / (src_path.stem + ".pdf")
        cmd = [
            "pandoc", str(src_path),
            "-o", str(pdf_path),
            "--pdf-engine=lualatex",
            "--resource-path", str(src_path.parent),
            f"--pdf-engine-opt=-output-directory={outdir}",
            # MiKTeX の場合は必要に応じて:
            # f"--pdf-engine-opt=-aux-directory={outdir}",
        ] + font_args

        # 作業CWDも/tmp側にする（万一の追加ファイルもここへ）
        subprocess.run(
            cmd, cwd=str(outdir), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if not pdf_path.exists():
            sys.stderr.write(f"[warn] PDFが生成されませんでした: {src_path.name}\n")
            return None

        return pdf_path  # ここで返るのは /tmp/.../xxx.pdf
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[warn] pandoc変換失敗: {src_path.name}\n{e.stderr.decode(errors='ignore')}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"[warn] 変換処理中に例外: {e}\n")
        return None

def upload_pdf_for_responses(pdf_path: Path, api_key) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {"purpose": "user_data"}
        r = requests.post(FILES_URL, headers=headers, files=files, data=data, timeout=600)
    if r.status_code != 200:
        # フォールバック（user_dataが使えない環境）
        if r.status_code == 400 and "purpose" in r.text:
            with open(pdf_path, "rb") as f2:
                files2 = {"file": (pdf_path.name, f2, "application/pdf")}
                data2 = {"purpose": "assistants"}
                r2 = requests.post(FILES_URL, headers=headers, files=files2, data=data2, timeout=600)
            if r2.status_code != 200:
                sys.stderr.write(f"[warn] PDFアップロード失敗: {pdf_path.name} -> {r2.status_code} {r2.text}\n")
                return None
            return r2.json().get("id")
        else:
            sys.stderr.write(f"[warn] PDFアップロード失敗: {pdf_path.name} -> {r.status_code} {r.text}\n")
            return None
    return r.json().get("id")


def _read_text_safely(path: Path) -> str | None:
    """UTF-8優先、ダメなら chardet 推測で読む"""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            raw = path.read_bytes()
        except Exception:
            return None
        try:
            enc = chardet.detect(raw).get("encoding") or "utf-8"
        except Exception:
            enc = "utf-8"
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            return None

def _chunk_text(s: str, size=MAX_CHARS_PER_CHUNK):
    return [s[i:i+size] for i in range(0, len(s), size)]

# ===== メッセージ整形 =====
def sanitize_history(history):
    """履歴の独自メタを落として API 仕様に整形"""
    clean = []
    for m in history:
        if not isinstance(m, dict): continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            norm = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t == "input_file" and "file_id" in c:
                        norm.append({"type":"input_file","file_id":c["file_id"]})
                    elif t in ("input_text","text") and "text" in c:
                        norm.append({"type":"input_text","text":c["text"]})
                    else:
                        norm.append({"type":"input_text","text":json.dumps(c, ensure_ascii=False)})
                elif isinstance(c, str):
                    norm.append({"type":"input_text","text":c})
            content = norm or ""
        elif isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        clean.append({"role": role, "content": content})
    return clean

def build_user_content(prompt, pdf_file_ids):
    if pdf_file_ids:
        content = [{"type":"input_text","text": prompt}]
        for fid in pdf_file_ids:
            content.append({"type":"input_file","file_id": fid})
        return content
    return prompt

def make_payload(history, prompt, use_search, pdf_ids, rag_ids):
    msgs = sanitize_history(history)
    user_msg = {"role":"user", "content": build_user_content(prompt, pdf_ids)}
    msgs.append(user_msg)

    payload = {
        "model": MODEL,
        "input": msgs,
        "instructions": SYSTEM_INSTRUCTIONS,  # ← 追加
        # 任意の安定化パラメータ（好みで有効化）
        # "temperature": 0.2,
        # "truncation": "auto",
    }

    tools = []
    if use_search:
        tools.append({"type":"web_search"})
    if rag_ids:
        tools.append({"type":"file_search"})
        payload["attachments"] = [{"file_id": fid, "tools":[{"type":"file_search"}]} for fid in rag_ids]
    if tools:
        payload["tools"] = tools
    return payload

def extract_text(resp_json):
    if resp_json.get("output_text"):
        return resp_json["output_text"]
    items = resp_json.get("output", []) or resp_json.get("response", {}).get("output", [])
    chunks = []
    for it in items:
        if it.get("type") == "message":
            for c in it.get("content", []):
                if c.get("type") in ("output_text","text") and "text" in c:
                    chunks.append(c["text"])
    if chunks: return "".join(chunks)
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        cont = msg.get("content")
        if isinstance(cont, str): return cont
        if isinstance(cont, list):
            return "".join(x.get("text","") for x in cont if isinstance(x, dict))
    return ""

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

# ===== APIコール（フォールバック込み） =====
def call_api(payload, api_key, local_paths=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=600)
    if resp.status_code == 200:
        return resp.json()

    j = safe_json(resp)
    err = (j or {}).get("error", {})
    if err.get("param") == "attachments":
        rag_ids = [a.get("file_id") for a in payload.get("attachments", [])] if "attachments" in payload else []
        payload.pop("attachments", None)
        if "tools" in payload:
            payload["tools"] = [t for t in payload["tools"] if t.get("type") != "file_search"]
            if not payload["tools"]:
                payload.pop("tools", None)

        new_pdf_ids = []
        inline_text_blocks = []

        with tempfile.TemporaryDirectory() as td:
            for fid in rag_ids:
                src = Path(local_paths.get(fid)) if local_paths and fid in local_paths else None
                if not src or not src.exists():
                    try:
                        src = download_file_content(fid, api_key, td)
                    except Exception as e:
                        sys.stderr.write(f"[warn] ダウンロード不可: {fid} ({e})\n")
                        continue

                text = None
                if src.suffix.lower() in TEXT_LIKE_EXT or src.suffix == "":
                    text = _read_text_safely(src)
                pdf = convert_to_pdf(src)

                if text:
                    chunks = _chunk_text(text)
                    header = f"【INLINE from {src.name}】\n"
                    for i, chunk in enumerate(chunks):
                        prefix = header if i == 0 else f"【INLINE cont. {src.name}】\n"
                        fenced = f"```\n{chunk}\n```"
                        inline_text_blocks.append(prefix + fenced)
                elif pdf:
                    nid = upload_pdf_for_responses(pdf, api_key)
                    if nid:
                        new_pdf_ids.append(nid)
                        sys.stderr.write(f"[compat/pdf] {src.name} -> {nid}\n")
                    continue
                else:
                    sys.stderr.write(f"[warn] アップロード失敗: {src.name}\n")

        inputs = payload.get("input", [])
        if inputs and isinstance(inputs[-1], dict) and inputs[-1].get("role") == "user":
            content = inputs[-1].get("content")
            if isinstance(content, str):
                content = [{"type":"input_text","text":content}]
            inputs[-1]["content"] = content
        else:
            content = [{"type":"input_text","text":"添付資料を参照して回答してください。"}]
            inputs.append({"role":"user","content":content})
            payload["input"] = inputs

        for np in new_pdf_ids:
            content.append({"type":"input_file","file_id":np})
        for block in inline_text_blocks:
            content.append({"type":"input_text","text":block})

        resp2 = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=600)
        if resp2.status_code != 200:
            raise RuntimeError(f"API error {resp2.status_code}: {resp2.text}")
        return resp2.json()

    raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(
        prog="gpt",
        description="ChatGPT CLI",
        usage="%(prog)s PROMPT... [OPTIONS]",
        epilog=(
            "注意:  OPTIONSの後にPROMPTを置く場合は、-- で区切ってください。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 通常
    ap.add_argument("prompt", nargs="*", help="質問/指示。未指定なら標準入力から")
    ap.add_argument("--session", "-s", help="履歴名（スレッド）。未指定はデフォルト")
    ap.add_argument("--no-search", action="store_true", help="Web検索を無効化")

    # 一覧/表示/エクスポート/grep/デフォルト
    ap.add_argument("--list", action="store_true", help="スレッド一覧")
    ap.add_argument("--show", action="store_true", help="このセッションの履歴表示")
    ap.add_argument("--last", type=int, help="--show/--export/--grep: 末尾N件")
    ap.add_argument("--role", choices=["all","user","assistant"], default="all", help="--show/--export/--grep ロール絞り込み")
    ap.add_argument("--export", choices=["markdown","json"], help="履歴を出力")
    ap.add_argument("--out", help="--export の出力先ファイル")
    ap.add_argument("--grep", help="履歴内を正規表現で検索")
    ap.add_argument("--all", action="store_true", help="--grep: 全スレッド横断")
    ap.add_argument("--ignore-case", "-i", action="store_true", help="--grep: 大文字小文字無視")
    ap.add_argument("--set-default", metavar="SESSION", help="デフォルトセッションを設定")
    ap.add_argument("--get-default", action="store_true", help="現在のデフォルトセッションを表示")

    # ファイル（既存APIのまま）
    ap.add_argument("--upload", nargs="+", help="ローカルファイルをアップロード（PDFは直添付、非PDFはRAG）")
    ap.add_argument("--attach", nargs="+", help="既存 file_id を添付（自動判別）")

    args = ap.parse_args()

    # デフォルトセッション操作
    if args.get_default:
        print(get_default_session()); return
    if args.set_default:
        set_default_session(args.set_default); print(f"default session -> {args.set_default}"); return

    # 一覧
    if args.list:
        print_sessions(list_sessions(), get_default_session()); return

    # セッション決定
    session = args.session or get_default_session()

    # 表示
    if args.show and not args.export and not args.grep:
        hist = load_history(session)
        if args.role in ("user","assistant"):
            hist = [m for m in hist if m.get("role")==args.role]
        if args.last and args.last>0:
            hist = hist[-args.last:]
        if not hist:
            print(f"(セッション '{session}' の履歴はありません)"); return
        for i, m in enumerate(hist, 1):
            r = m.get("role","?"); c = _content_to_text(m.get("content",""))
            print(f"[{i}] {r}:\n{c}")
            meta = []
            if m.get("files_pdf"): meta.append("pdf=" + ",".join(m["files_pdf"]))
            if m.get("files_rag"): meta.append("rag=" + ",".join(m["files_rag"]))
            if meta: print(f"(attached: {' | '.join(meta)})")
            print()
        return

    # エクスポート
    if args.export:
        export_history(session, args.export, out_path=args.out, role=args.role, last=args.last); return

    # grep
    if args.grep:
        hits = grep_sessions(args.grep, session=session, all_sessions=args.all,
                             role=args.role, ignore_case=args.ignore_case, last=args.last)
        print_grep_results(hits); return

    # ---- 通常問い合わせ ----
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY が未設定です。export OPENAI_API_KEY=... を実行してください。", file=sys.stderr); sys.exit(1)

    # 入力
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = sys.stdin.read().strip()
    if not prompt and not args.upload and not args.attach:
        print("空のプロンプトです（ファイルだけ投げる場合でも簡単な指示文を推奨）", file=sys.stderr); sys.exit(1)

    # アップロード＆既存ID分類
    pdf_ids, rag_ids, id2local = [], [], {}
    if args.upload:
        p_pdf, p_rag, mapping = upload_auto(args.upload, api_key)
        pdf_ids += p_pdf; rag_ids += p_rag; id2local.update(mapping)
    if args.attach:
        a_pdf, a_rag = classify_existing_ids(args.attach, api_key)
        pdf_ids += a_pdf; rag_ids += a_rag

    history = load_history(session)
    payload = make_payload(history, prompt or "ファイルを参照して回答してください。", use_search=(not args.no_search),
                           pdf_ids=pdf_ids, rag_ids=rag_ids)

    resp = call_api(payload, api_key, local_paths=id2local)
    text = extract_text(resp) or "(no text)"
    print(text.strip())

    now_ts = int(time.time())
    history.append({"role": "user", "content": (prompt or ""), "files_pdf": pdf_ids, "files_rag": rag_ids, "ts": now_ts})
    history.append({"role": "assistant", "content": text, "ts": now_ts})
    save_history(session, history)

if __name__ == "__main__":
    raise SystemExit(main())
