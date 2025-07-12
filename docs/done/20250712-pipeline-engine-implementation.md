# 2025年07月12日: パイプライン実行制御機構の実装

## 目的

PDF分割から最終的なMarkdown生成まで、全ての処理ステップを統合し、順序立てて実行する責務を担うパイプライン実行制御機構を実装する。

## 完了した項目

- [x] **パイプライン実行エンジンの実装 (`src/assetable/pipeline/engine.py`)**
  - [x] **基本構造の実装**
    - [x] `PipelineError`, `PipelineStepError`, `PipelineConfigError` のカスタム例外クラスを定義。
    - [x] `PipelineStep` 抽象ベースクラスを定義し、各パイプラインステップの共通インターフェース（`step_name`, `processing_stage`, `dependencies`, `execute_page`, `execute_document`等）を確立。
  - [x] **パイプラインステップの実装**
    - [x] `PDFSplitStep`: PDF分割処理を行うステップ。`PDFSplitter`を内部で使用し、ドキュメントレベルでの実行を実装。
    - [x] `StructureAnalysisStep`: 構造解析のプレースホルダーステップを実装。
    - [x] `AssetExtractionStep`: 資産抽出のプレースホルダーステップを実装。
    - [x] `MarkdownGenerationStep`: Markdown生成のプレースホルダーステップを実装。
  - [x] **パイプラインエンジンの実装 (`PipelineEngine`)**
    - [x] デフォルトのパイプラインステップ（PDF分割、構造解析、資産抽出、Markdown生成）を定義。
    - [x] `execute_pipeline` メソッドを実装し、パイプライン全体の実行を制御。
    - [x] `execute_single_step` メソッドを実装し、単一ステップの実行を可能に。
    - [x] `_initialize_document` メソッドで、既存の `DocumentData` の読み込みまたは新規作成を実装。
    - [x] `get_pipeline_status` メソッドで、現在の処理状況や進捗を確認する機能を提供。
    - [x] `add_step`, `remove_step`, `get_step` といったステップ管理機能を追加。
    - [x] 依存関係の検証 (`_validate_pipeline`) と実行ステップのフィルタリング (`_filter_steps`) ロジックを実装。
  - [x] **コンビニエンス関数の実装**
    - [x] `run_pipeline`: `PipelineEngine`をインスタンス化してパイプライン全体を実行するヘルパー関数。
    - [x] `run_single_step`: `PipelineEngine`をインスタンス化して単一ステップを実行するヘルパー関数。

- [x] **CLIコマンドの拡張 (`src/assetable/cli.py`)**
  - [x] **`pipeline` コマンドの実装**
    - [x] `typer.Argument` を使用して必須のPDFパス引数を定義。
    - [x] `--stages`, `--pages`, `--force`, `--output`, `--debug` のオプションを追加し、柔軟な実行制御を実現。
    - [x] 入力されたステージ名 (`split`, `structure`等) を `ProcessingStage` Enumにマッピング。
    - [x] `asyncio.run` を使用して非同期の `execute_pipeline` メソッドを実行。
    - [x] `typer.progressbar` を使用して処理の進捗を視覚的に表示。
    - [x] 実行前後のステータスを表示し、処理結果をユーザーにフィードバック。
  - [x] **`run-step` コマンドの実装**
    - [x] 実行するステップ名を引数として受け取るように定義。
    - [x] `pipeline` コマンドと同様に、`--pages`, `--force`, `--debug` オプションをサポート。
    - [x] ステップ名を対応する `PipelineStep` クラスにマッピング。
    - [x] `run_single_step` ヘルパー関数を呼び出して単一ステップを実行。
    - [x] 実行結果と進捗ステータスを表示。

- [x] **モデルの更新への追従**
  - [x] `DocumentData` モデルの変更（`source_pdf` -> `source_pdf_path`, `total_pages` の削除）に伴い、`engine.py` 内での `DocumentData` のインスタンス化と属性アクセスを修正。
  - [x] `document_id` をPDFファイル名から生成するように実装。

- [x] **ファイルベースの状態管理との連携**
  - [x] プレースホルダーのパイプラインステップ (`StructureAnalysisStep`, `MarkdownGenerationStep`) が、`FileManager` がステージ完了の指標とするための空のファイル（`structure.json`, `markdown.md`）を生成するように修正。
  - [x] `StructureAnalysisStep` が、後続の `AssetExtractionStep` の完了チェックをパスできるように、有効な（ただし最小限の）`PageStructure` オブジェクトをJSONファイルに保存するように修正。
