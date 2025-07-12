# 2025年07月12日: CrossPageReference の reference_type を具体的な型に分離

以下のタスクが完了しました。

- [x] `src/assetable/models.py` の `CrossPageReference` クラスを修正
    - [x] `ReferenceType` Enum を新しく追加
        - `PAGE`: ページへの参照
        - `HEADING`: 見出しへの参照
        - `TABLE`: 表への参照
        - `FIGURE`: 図への参照
        - `IMAGE`: 画像への参照
    - [x] `CrossPageReference.reference_type` の型ヒントを `str` から `ReferenceType` に変更
- [x] 変更内容の確認
- [x] ブランチ `feature/concrete-reference-types` で変更をコミット
    - コミットメッセージ: "Refactor: Separate CrossPageReference.reference_type into specific types\n\nAdded a new ReferenceType Enum and updated CrossPageReference to use it for reference_type.\nThis improves type safety and allows for more specific reference type definitions for AI model outputs."
