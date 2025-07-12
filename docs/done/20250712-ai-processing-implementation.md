# AI処理実装

3段階のAI処理（構造解析、資産抽出、Markdown生成）を詳細に実装しました。

- [x] `src/assetable/ai/prompts.py` の実装
  - [x] 構造解析、資産抽出、Markdown生成のためのプロンプトテンプレートを実装
- [x] `src/assetable/ai/vision_processor.py` の更新
  - [x] 3段階のAI処理を実装
  - [x] `EnhancedVisionProcessor` を実装し、エラーハンドリング、リトライロジック、パフォーマンスモニタリング、品質バリデーションを強化
- [x] `src/assetable/pipeline/ai_steps.py` の更新
  - [x] `EnhancedVisionProcessor` を使用するようにパイプラインステップを更新
  - [x] `EnhancedAIStructureAnalysisStep`, `EnhancedAIAssetExtractionStep`, `EnhancedAIMarkdownGenerationStep` を実装
- [x] `src/assetable/cli.py` の拡張
  - [x] `analyze` コマンドを追加し、特定のページに対してAI処理を個別実行できるようにした
  - [x] `process-single` コマンドを追加し、単一ページをパイプラインで処理できるようにした
- [x] `src/assetable/pipeline/engine.py` の更新
  - [x] `PipelineEngine` が新しい拡張AIステップを使用するように更新
