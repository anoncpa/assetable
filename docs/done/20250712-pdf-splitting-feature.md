# PDF分割機能の実装 (2025年07月12日)

以下のタスクが完了しました。

## 1. PDF分割機能のコア実装

- [x] **`src/assetable/pipeline/pdf_splitter.py` の作成と実装**
    - [x] `PDFSplitter` クラスを実装し、PyMuPDF (fitz) を使用してPDFページを画像に変換する機能を提供。
    - [x] DPI設定 (`config.pdf_split.dpi`) に基づいて画像の解像度を調整。
    - [x] 設定された画像フォーマット (`config.pdf_split.image_format`) に従って画像を保存 (PNG, JPG/JPEG対応)。
    - [x] `DocumentData` および `PageData` モデルを使用して処理状態とメタデータを管理。
    - [x] `FileManager` を使用して、生成された画像やメタデータJSONファイルを適切なディレクトリ構造に保存。
    - [x] **エラーハンドリング**:
        - `PDFNotFoundError`: PDFファイルが見つからない場合。
        - `PDFCorruptedError`: PDFファイルが破損している、または開けない場合 (0ページPDFも含む)。
        - `ImageConversionError`: 画像変換中にエラーが発生した場合。
        - `PDFSplitterError`: その他の予期せぬエラー。
    - [x] **既存ファイルのスキップ/強制再生成**:
        - `force_regenerate` パラメータによる画像の強制再生成機能。
        - `config.processing.skip_existing_files` 設定に基づき、既に処理済みのページをスキップする機能。
    - [x] **PDF情報取得機能**:
        - `get_pdf_info`: PDFの基本情報 (ページ数、メタデータ、ファイルサイズ等) を取得。
        - `get_processing_status`: PDFの処理状況 (完了ページ数、保留ページ数、進捗率) を取得。
    - [x] **クリーンアップ機能**:
        - `cleanup_split_files`: 指定されたPDFに関連する生成済み画像ファイルを削除。全ページまたは特定ページ番号の指定が可能。
    - [x] **CLIラッパー関数**:
        - `split_pdf_cli`: `PDFSplitter.split_pdf` を呼び出すCLI向け関数。

## 2. CLIコマンドの統合

- [x] **`src/assetable/cli.py` の更新**
    - [x] **`split` コマンド**:
        - PDFファイルを指定し、ページごとに画像へ分割。
        - オプション: `--force` (強制再生成), `--dpi` (解像度指定), `--output` (出力先ディレクトリ指定), `--debug` (デバッグモード有効化)。
        - 処理前にPDF情報を表示。
        - 既存ページの処理状況を表示し、`--force` がない場合はスキップすることを通知。
        - `typer.progressbar` を使用した進捗表示。
        - 処理結果 (完了ページ数、出力ディレクトリ) を表示。
    - [x] **`info` コマンド**:
        - 指定されたPDFのメタデータ、ページ数、ファイルサイズ、処理ステータス等を表示。
    - [x] **`status` コマンド**:
        - 指定されたPDFの各処理ステージごとの進捗状況 (完了ページ数、保留ページ数、進捗率) を表示。
    - [x] **`cleanup` コマンド**:
        - 生成されたファイルをクリーンアップ。
        - `--stage` オプションでクリーンアップ対象ステージを指定 (`split`, `all` など)。
        - `--yes` オプションで確認プロンプトをスキップ。
    - [x] 設定のオーバーライド: CLIオプションで指定された値を `AssetableConfig` に反映。

## 3. 設定ファイルの更新・利用

- [x] **`src/assetable/config.py` の利用**
    - [x] `PDFSplitConfig` でDPIや画像フォーマットを設定可能にした。
    - [x] `OutputConfig` で出力ディレクトリ構造やファイル名パターンを管理。
    - [x] `get_page_image_path` メソッドを更新し、設定された画像フォーマットに基づいて動的にファイル拡張子を決定するように修正。

## 4. モデルの更新・利用

- [x] **`src/assetable/models.py` の利用**
    - [x] `DocumentData` モデルを更新し、`document_id` と `source_pdf_path` を必須フィールドとした。
    - [x] `total_pages` フィールドを `DocumentData` から削除 (処理概要の計算時に渡されるように変更)。
    - [x] `PDFSplitter` がこれらの変更に対応するように修正。
    - [x] `PageData` モデルで各ページの処理状態 (`current_stage`, `completed_stages`) や関連ファイルパス (`image_path` 等) を管理。

この実装により、PDFファイルを高品質な画像に分割し、後続のAI処理パイプラインへの入力を提供する基盤が整いました。CLIからの操作も可能です。
