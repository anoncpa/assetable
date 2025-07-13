# Assetable

`assetable`は、スキャンされた本やPDFドキュメントを、AIと人間の両方が読みやすいデジタル資産に変換するためのコマンドラインツールです。PDFをページごとに画像に分割し、Ollamaを利用したAI処理によって、テキスト、図、表などを抽出・構造化し、最終的にMarkdown形式のファイルとして出力します。

## 主な機能

- **PDFの分割**: PDFの各ページを高品質な画像ファイル（PNG）に変換します。
- **AIによる構造分析**: 各ページのレイアウトを分析し、テキスト、図、表などの要素を識別します。
- **資産の抽出**: ページ内から図や表などの具体的なアセットを抽出します。
- **Markdown生成**: 分析結果を元に、構造化されたMarkdownファイルを生成します。
- **パイプライン処理**: 上記の処理をパイプラインとして一括で実行できます。

## インストール

リポジトリをクローンし、Poetryを使用して依存関係をインストールします。

```bash
git clone https://github.com/your-username/assetable.git
cd assetable
pip install .
```

## 使い方

`assetable`コマンドは、いくつかのサブコマンドを持っています。

### 1. AIシステムのチェック (`check-ai`)

まず、AIシステム（Ollama）が正しく設定されているか確認します。

```bash
assetable check-ai
```

Ollamaが起動しており、必要なモデルがインストールされているかどうかが表示されます。

### 2. PDFの分割 (`split`)

PDFファイルをページごとの画像に分割します。

```bash
assetable split path/to/your/book.pdf
```

- `--dpi`: 画像の解像度（DPI）を指定します。（デフォルト: 300）
- `--output`: 出力先ディレクトリを指定します。
- `--force`: 既存のファイルがあっても強制的に再生成します。

### 3. パイプラインの実行 (`pipeline`)

PDFの分割からMarkdown生成までの一連の処理を実行します。

```bash
assetable pipeline path/to/your/book.pdf
```

- `--stages`: 実行するステージを指定します。（例: `split,structure`）
- `--pages`: 処理するページをカンマ区切りで指定します。（例: `1,2,5`）
- `--force`: 既存のファイルを無視して強制的に再処理します。

### 4. 処理状況の確認 (`status`)

パイプラインの処理状況を確認できます。

```bash
assetable status path/to/your/book.pdf
```

### 5. PDF情報の表示 (`info`)

PDFファイルのメタデータやページ数などの情報を表示します。

```bash
assetable info path/to/your/book.pdf
```

### 6. 生成ファイルのクリーンアップ (`cleanup`)

生成されたファイルを削除します。

```bash
assetable cleanup path/to/your/book.pdf --stage all
```

- `--stage`: クリーンアップするステージを指定します。（`split`, `all`など）
- `--yes`: 確認プロンプトをスキップします。

## 開発者向けコマンド

- `analyze`: 特定のページを詳細に分析し、デバッグ情報を表示します。
- `process-single`: パイプラインの各ステップを単一ページに対して実行します。
- `test-ai`: AI処理のテストを実行します。