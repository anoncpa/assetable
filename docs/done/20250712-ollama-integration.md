# 2025年07月12日: Ollama実行クラスの実装

## 完了した項目

- [x] **Ollamaクライアントの実装 (`src/assetable/ai/ollama_client.py`)**
  - [x] Ollamaサーバーとの通信を処理する`OllamaClient`クラスを作成しました。
  - [x] Visionモデルを使用した画像とプロンプトの送信機能 (`chat_with_vision`) を実装しました。
  - [x] Pydanticモデルを使用した構造化出力のサポートを追加しました。
  - [x] 接続エラー、モデルエラー、レスポンスエラーなどのためのカスタム例外クラスを定義しました。
  - [x] 利用可能なモデルのリストの取得とキャッシング機能を実装しました。
  - [x] 指数バックオフによるリトライメカニズムを実装しました。
  - [x] 処理統計（リクエスト数、合計処理時間など）の追跡機能を追加しました。

- [x] **Visionプロセッサの実装 (`src/assetable/ai/vision_processor.py`)**
  - [x] `VisionProcessor`クラスを作成し、3段階のAI処理アプローチ（構造解析、資産抽出、Markdown生成）を実装しました。
  - [x] `analyze_page_structure`メソッドを実装し、ページ画像からテキスト、表、図、画像、相互参照を検出します。
  - [x] `extract_assets`メソッドを実装し、検出された資産（表、図、画像）を構造化データに変換します。
  - [x] `generate_markdown`メソッドを実装し、ページ全体を表現するMarkdownコンテンツを生成します。
  - [x] 各処理段階で異なるAIモデルを使用できるように設定可能にしました。

- [x] **AIパイプラインステップの実装 (`src/assetable/pipeline/ai_steps.py`)**
  - [x] `AIStructureAnalysisStep`、`AIAssetExtractionStep`、`AIMarkdownGenerationStep`の3つのパイプラインステップを作成しました。
  - [x] 各ステップが`VisionProcessor`を使用して、対応するAI処理を実行するようにしました。
  - [x] パイプラインエンジンとの統合のために、依存関係と処理ステージを正しく定義しました。

- [x] **パイプラインエンジンの更新 (`src/assetable/pipeline/engine.py`)**
  - [x] 新しいAIパイプラインステップをデフォルトのステップリストに統合しました。
  - [x] 循環インポートの問題を解決するために、インポートのタイミングを調整しました。

- [x] **CLIコマンドの拡張 (`src/assetable/cli.py`)**
  - [x] `check-ai`コマンドを追加し、Ollamaサーバーの接続状態と利用可能なモデルを確認できるようにしました。
  - [x] `test-ai`コマンドを追加し、単一ページで特定のAI処理ステージをテストできるようにしました。

- [x] **テストコードの実装**
  - [x] `tests/test_ollama_client.py`を作成し、`OllamaClient`の包括的なテストを実装しました。
  - [x] `tests/test_vision_processor.py`を作成し、`VisionProcessor`の3つの処理段階をテストしました。
  - [x] `tests/test_ai_pipeline_steps.py`を作成し、AIステップとパイプラインフレームワークの統合をテストしました。
  - [x] `tests/conftest.py`を設定し、Ollamaに依存するテストを適切にマーキングしてスキップできるようにしました。

- [x] **プロジェクト設定の更新**
  - [x] `pyproject.toml`に`ollama`ライブラリの依存関係を追加しました。
  - [x] パッケージが正しくインストールされ、テストが実行できるように、`pip install -e .`を実行しました。
