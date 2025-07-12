# 2025年07月12日: 設定クラスの実装

以下のタスクが完了しました。

- [x] `src/assetable/config.py` に設定クラスを実装
    - [x] `PDFSplitConfig` クラスの実装とバリデーション
        - `dpi`: 72以上600以下
        - `image_format`: "png", "jpg", "jpeg" のいずれか (小文字に変換)
    - [x] `AIConfig` クラスの実装とバリデーション
        - `ollama_host`, `structure_analysis_model`, `asset_extraction_model`, `markdown_generation_model` のデフォルト値設定
        - `max_retries`, `timeout_seconds` のデフォルト値設定
        - `temperature`: 0.0以上2.0以下
        - `top_p`: 0.0以上1.0以下
    - [x] `OutputConfig` クラスの実装とバリデーション
        - `input_directory`, `output_directory` のデフォルト値設定と絶対パスへの変換
        - 各種サブディレクトリ名 (`pdf_split_subdir`, `structure_subdir` など) のデフォルト値設定
        - 各種ファイル命名パターン (`page_image_pattern` など) のデフォルト値設定
    - [x] `ProcessingConfig` クラスの実装とバリデーション
        - `skip_existing_files`, `debug_mode`, `save_intermediate_results` のデフォルト値設定
        - `max_parallel_pages`: 1以上10以下
        - `min_table_rows`, `min_figure_elements` のデフォルト値設定
    - [x] `AssetableConfig` メイン設定クラスの実装
        - 上記サブ設定クラスの集約
        - `version` 情報のデフォルト値設定
        - `from_env()` クラスメソッドによる環境変数からの設定読み込み機能
            - `ASSETABLE_OLLAMA_HOST`, `ASSETABLE_STRUCTURE_MODEL`, `ASSETABLE_ASSET_MODEL`, `ASSETABLE_MARKDOWN_MODEL`, `ASSETABLE_DPI`, `ASSETABLE_INPUT_DIR`, `ASSETABLE_OUTPUT_DIR`, `ASSETABLE_DEBUG` に対応
        - 各種出力パス生成メソッドの実装
            - `get_document_output_dir`, `get_pdf_split_dir`, `get_structure_dir`, `get_markdown_dir`, `get_csv_dir`, `get_images_dir`, `get_figures_dir`
            - `get_page_image_path`, `get_structure_json_path`, `get_markdown_path`, `get_asset_path`
        - `create_output_directories` メソッドによる出力ディレクトリ一括作成機能
        - `to_dict()` メソッドによる設定の辞書変換機能 (Pydantic V1/V2互換)
        - `save_to_file()` メソッドによるJSONファイルへの設定保存機能 (Pydantic V1/V2互換)
        - `load_from_file()` クラスメソッドによるJSONファイルからの設定読み込み機能
    - [x] グローバル設定インスタンス管理機能の実装
        - `get_config()`, `set_config()`, `reset_config()`
- [x] 作成されたファイルの内容確認
- [x] ブランチ `feature/config-class` で変更をコミット
    - コミットメッセージ: "feat: Implement configuration class\n\nAdded a comprehensive configuration management system using Pydantic.\n\nFeatures:\n- Type-safe configuration with validation.\n- Hierarchical structure for different components (PDF split, AI, Output, Processing).\n- Support for overriding settings via environment variables.\n- Centralized path management for output files and directories.\n- Methods for saving and loading configuration to/from JSON files.\n- Global access to configuration instance via `get_config()`."
