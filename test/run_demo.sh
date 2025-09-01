#!/bin/bash

# test/demo.cの実行専用スクリプト - 詳細な説明付き

set -e

echo "======================================================"
echo "  t9ccコンパイラ デモンストレーション"
echo "======================================================"
echo
echo "このスクリプトは以下の処理を実行します："
echo "1. test/demo.cをt9ccでコンパイル"
echo "2. 生成されたアセンブリをgccでビルド"
echo "3. orb上で実行してt9ccの機能をテスト"
echo

# demo.cの存在確認
if [ ! -f "test/demo.c" ]; then
    echo "エラー: test/demo.c が見つかりません"
    echo "プロジェクトルートから実行していることを確認してください"
    exit 1
fi

echo "--- demo.cの内容確認 ---"
echo "demo.cに含まれる機能："
echo "• 基本的な算術演算（+, -, *, /）"
echo "• 変数の宣言と代入"
echo "• 配列操作"
echo "• ポインタ操作"  
echo "• グローバル変数"
echo "• 制御構造（if, while, for）"
echo "• 関数呼び出し（再帰を含む）"
echo "• 文字列リテラル"
echo "• sizeof演算子"
echo

# buildディレクトリの作成
mkdir -p build

echo "=== コンパイル開始 ==="

# Step 1: コンパイラのビルド
echo "Step 1: t9ccコンパイラをビルド中..."
cargo build --release --quiet
echo "✓ コンパイラビルド完了"

# Step 2: ヘルパーファイルのコンパイル
echo "Step 2: ヘルパーファイルをコンパイル中..."
orb -m ubuntu exec gcc -c -o build/test_helper.o test/test_helper.c
echo "✓ ヘルパーコンパイル完了"

# Step 3: demo.cをアセンブリに変換
echo "Step 3: demo.c をアセンブリに変換中..."
target/release/t9cc test/demo.c > build/demo.s
echo "✓ アセンブリ変換完了"

# アセンブリファイルのサイズ表示
ASM_LINES=$(wc -l < build/demo.s)
echo "  生成されたアセンブリ: ${ASM_LINES}行"

# Step 4: 最終的な実行ファイルの生成
echo "Step 4: 実行ファイルを生成中..."
orb -m ubuntu exec gcc -o build/demo build/demo.s build/test_helper.o
echo "✓ 実行ファイル生成完了"

echo "=== コンパイル完了 ==="
echo

echo "=== 実行開始 ==="
echo "プログラムを実行します..."
echo

# 実行
echo "--- プログラム出力 ---"
if orb -m ubuntu exec build/demo; then
    EXIT_CODE=$?
else
    EXIT_CODE=$?
fi

echo "--- 実行終了 ---"
echo

# 終了コードを適切に解釈
if [ $EXIT_CODE -eq 255 ]; then
    echo "❌ プログラムがエラーで終了しました"
    echo "終了コード: $EXIT_CODE (システムエラー)"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ プログラムが正常に実行されました！"
    echo "終了コード: $EXIT_CODE"
else
    echo "✅ プログラムが正常に実行されました！"
    echo "計算結果（終了コード）: $EXIT_CODE"
    echo
    echo "🎉 t9ccコンパイラが正常に動作しています！"
fi

echo
echo "=== 詳細情報 ==="
echo "生成されたファイル:"
ls -la build/demo* build/test_helper.o 2>/dev/null || true
echo

echo "アセンブリファイルの先頭部分:"
echo "--- build/demo.s (最初の10行) ---"
head -10 build/demo.s
echo "..."

# クリーンアップオプション
echo
read -p "一時ファイルを削除しますか？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f build/demo.s build/demo
    echo "✓ 一時ファイルを削除しました"
else
    echo "一時ファイルを保持しました"
    echo "  アセンブリ: build/demo.s"
    echo "  実行ファイル: build/demo"
fi

echo
echo "======================================================"
echo "  デモンストレーション完了"
echo "======================================================"