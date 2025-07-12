# プロジェクトセットアップとsrcレイアウトへの移行 (2024-07-24)

## 完了したタスク

### フェーズ1: 初期プロジェクトセットアップ

- [x] **ディレクトリ構造の作成**
  - [x] `assetable` パッケージディレクトリ (`assetable/`)
  - [x] `assetable` 内の `pipeline` サブディレクトリ (`assetable/pipeline/`)
  - [x] `tests` ディレクトリ (`tests/`)
- [x] **Pythonファイルの作成**
  - [x] `assetable/__init__.py` (空)
  - [x] `assetable/cli.py` (Typer CLIエントリポイント、`split` コマンド定義)
  - [x] `assetable/config.py` (空、Pydantic設定用プレースホルダ)
  - [x] `assetable/models.py` (空、Pydanticデータモデル用プレースホルダ)
  - [x] `assetable/pipeline/__init__.py` (空)
  - [x] `assetable/pipeline/pdf_splitter.py` (PyMuPDFによるPDF分割処理の骨格、`SplitConfig` モデル定義)
  - [x] `assetable/pipeline/ai_executor.py` (空、ollama連携用プレースホルダ)
  - [x] `assetable/pipeline/pipeline.py` (空、処理パイプライン制御用プレースホルダ)
  - [x] `tests/test_pdf_splitter.py` (空、テスト用プレースホルダ)
- [x] **設定ファイルの作成**
  - [x] `ruff.toml` (Lint設定: line-length, target-version, select, ignore)
  - [x] `pyrightconfig.json` (型チェック設定: typeCheckingMode, pythonVersion, exclude)
  - [x] `.gitignore` (標準的なPythonプロジェクト用。既存のものを確認し、流用)
- [x] **Python環境のセットアップ (`uv` 使用)**
  - [x] 仮想環境作成 (`uv venv .venv`) および有効化 (試行)
  - [x] `pyproject.toml` の手動初期化 (`[project]` テーブル作成)
  - [x] ランタイム依存関係の追加 (`uv add typer pydantic pymupdf ollama`)
    - `typer`
    - `pydantic`
    - `pymupdf`
    - `ollama`
  - [x] 開発用依存関係の追加 (`uv add --dev ruff pyright pytest`)
    - `ruff`
    - `pyright`
    - `pytest`
- [x] **`pyproject.toml` の編集**
  - [x] プロジェクトメタ情報追加 (`description`, `requires-python`)
  - [x] `[project.scripts]` 追加 (`assetable = "assetable.cli:app"`)
  - [x] `dependencies` と `dev-dependencies` (`[dependency-groups.dev]`) が `uv add` により自動設定されることを確認
- [x] **動作確認**
  - [x] `uv run python -m assetable.cli --help` を実行し、CLIヘルプメッセージが表示されることを確認

### フェーズ2: `src` レイアウトへの移行

- [x] **ディレクトリ構造の変更**
  - [x] `src` ディレクトリ作成
  - [x] `assetable` ディレクトリを `src/` 配下に移動 (`src/assetable/`)
- [x] **`pyproject.toml` の修正**
  - [x] `[tool.setuptools.packages.find]` 追加 (`where = ["src"]`)
  - [x] `[build-system]` 追加 (setuptoolsベースのビルド設定)
  - [x] 開発依存関係の指定を `[dependency-groups.dev]` に統一 (当初 `[project.optional-dependencies.dev]` に変更したが、`uv sync --dev` との互換性のため元に戻した)
- [x] **`pyrightconfig.json` の修正**
  - [x] `include` パスを `["src"]` に変更
- [x] **`ruff.toml` の修正**
  - [x] `src = ["src"]` を追加
- [x] **テストファイルのインポート確認**
  - [x] `tests/test_pdf_splitter.py` 内のインポート文は `src` レイアウトでも変更不要であることを確認 (理論上)
- [x] **依存関係の再同期**
  - [x] `uv sync --dev` を実行し、プロジェクト自体が編集可能モードでインストールされ、かつ開発依存関係が正しくインストールされることを確認
- [x] **動作確認**
  - [x] `uv run python -m assetable.cli --help` を実行し、`src` レイアウト変更後もCLIヘルプメッセージが表示されることを確認
