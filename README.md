# gpt-cli — ChatGPT をターミナルから使う軽量 CLI

ローカルのファイルを添付しつつ、プロンプトで ChatGPT に指示できるコマンドラインツールです。pipx で隔離インストールでき、システムの Python を汚しません。

- 推奨: Python 3.9 以上
- コマンド名: gpt

## インストール

pipx を推奨します。

- ローカルのソースから
```
pipx install .
```

- Git リポジトリから（例）
```
pipx install git+https://github.com/you/gpt-cli.git
```

- PyPI に公開後
```
pipx install gpt-cli
```

アップグレード
```
pipx upgrade gpt-cli
```

アンインストール
```
pipx uninstall gpt-cli
```

インストール確認
```
pipx list
gpt --version
```

## セットアップ（API キー）

OpenAI API を利用する場合の例です。別プロバイダを使う場合は環境変数名を読み替えてください。

- シェル（bash/zsh 等）
```
export OPENAI_API_KEY="sk-..."
# 必要なら
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_MODEL="gpt-4o-mini"
```

- Windows PowerShell
```
setx OPENAI_API_KEY "sk-..."
```

環境変数は gpt 起動時に読み込まれます。

## 使い方

基本形（プロンプトを先頭に）
```
gpt "この資料を要約して" --upload report.pdf notes.md
```

usage（ヘルプにも表示されます）
```
gpt PROMPT... [--upload FILE [FILE ...]] [OPTIONS]
```

重要: --upload の後ろにプロンプトを置く場合は、-- で区切ってください。
```
gpt --upload report.pdf notes.md -- "この資料を要約して"
```

よく使う例
- 複数ファイルを添付してコードレビュー
```
gpt "この2つの差分で壊れそうな箇所を指摘して" --upload src/main.py src/utils.py
```

- 日本語→英語の改善案
```
gpt "下記文の英訳と改善案を出して\n---\n明日の会議を来週に延期したいです。"
```

- 端末からの一行プロンプト（クォート推奨）
```
gpt "要点だけ3つに箇条書きで"
```

ヘルプ
```
gpt --help
```

## 振る舞いの要点

- プロンプトは位置引数（必須）です。曖昧さを避けるため、原則としてコマンドの先頭に置いてください。
- --upload は複数ファイルに対応します（nargs="+"). シェル展開を使う場合はファイル数が多いときに便利です。
- --upload の後ろにプロンプトを書く場合、argparse の仕様上プロンプトが引数として吸い込まれます。必ず -- を挟んで区切ってください。

## 退出コード

- 正常終了: 0
- 入力やネットワーク等のエラー: 非 0

## トラブルシューティング

- pipx で「Could not find a version that satisfies the requirement glob / datetime」等が出る
  - 標準ライブラリ名を依存関係に入れているのが原因です。pyproject.toml の dependencies から glob, datetime, argparse, json などの標準ライブラリを削除してください。
  - 再インストール: pipx uninstall gpt-cli; pipx install .

- 実行時に「API key not set」
  - OPENAI_API_KEY（または利用プロバイダの API キー）を環境変数で設定してください。新しいシェルを開き直すと反映されます。

- 端末で日本語が化ける
  - ターミナルの文字コードを UTF-8 に設定してください。

## 開発

クローンしてローカルインストール
```
git clone https://github.com/you/gpt-cli.git
cd gpt-cli
pipx install --suffix=@dev .
# 実行: gpt@dev --help
```

変更を反映させたいとき
```
pipx reinstall gpt-cli --force
# もしくは
pipx install --force .
```

プロジェクト構成（例）
```
src/
  gptcli/
    __init__.py
    cli.py      # main() を定義
pyproject.toml
README.md
```

エントリポイント（pyproject.toml の例）
```
[project.scripts]
gpt = "gptcli.cli:main"
```

## ライセンス

MIT License

## 謝辞

- 本ツールは OpenAI API 等の外部サービスを利用します。各サービスの利用規約に従ってお使いください。

