#!/bin/bash

# t9ccコンパイラを使ってCファイルをorb上でコンパイル・実行するスクリプト

set -e  # エラー時に停止

if [ $# -ne 1 ]; then
    echo "使用法: $0 <Cファイル名>"
    echo "例: $0 fibonacci.c"
    exit 1
fi

INPUT_FILE="$1"
BASENAME=$(basename "$INPUT_FILE" .c)

# ファイルの存在確認
if [ ! -f "$INPUT_FILE" ]; then
    echo "エラー: ファイル '$INPUT_FILE' が見つかりません"
    exit 1
fi

# 一時ファイル名の設定
ASM_FILE="build/${BASENAME}.s"
EXE_FILE="build/${BASENAME}"
HELPER_OBJ="build/test_helper.o"

echo "=== t9ccコンパイラでのCファイルコンパイル ==="
echo "入力ファイル: $INPUT_FILE"
echo

# buildディレクトリの作成
mkdir -p build

# Step 1: t9ccコンパイラのビルド
echo "Step 1: t9ccコンパイラをビルド中..."
cargo build --release --quiet
echo "✓ コンパイラのビルド完了"

# Step 2: test_helper.cをコンパイル（必要な場合）
if [ ! -f "$HELPER_OBJ" ]; then
    echo "Step 2: test_helper.cをコンパイル中..."
    orb -m ubuntu exec gcc -c -o "$HELPER_OBJ" test/test_helper.c
    echo "✓ ヘルパーファイルのコンパイル完了"
else
    echo "Step 2: ヘルパーファイルは既にコンパイル済み"
fi

# Step 3: Cファイルをアセンブリコードにコンパイル
echo "Step 3: $INPUT_FILE をアセンブリコードに変換中..."
target/release/t9cc "$INPUT_FILE" > "$ASM_FILE"
echo "✓ アセンブリファイル生成完了: $ASM_FILE"

# Step 4: orb上でgccを使ってアセンブリをコンパイル
echo "Step 4: アセンブリをorb上でコンパイル中..."
orb -m ubuntu exec gcc -o "$EXE_FILE" "$ASM_FILE" "$HELPER_OBJ"
echo "✓ 実行ファイル生成完了: $EXE_FILE"

# Step 5: orb上で実行
echo "Step 5: プログラムを実行中..."
echo "--- 実行結果 ---"
if orb -m ubuntu exec "$EXE_FILE"; then
    EXIT_CODE=$?
else
    EXIT_CODE=$?
fi

echo "--- 実行完了 ---"

# 終了コードを適切に解釈
if [ $EXIT_CODE -eq 255 ]; then
    echo "❌ プログラムがエラーで終了しました"
    echo "終了コード: $EXIT_CODE (システムエラー)"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✓ プログラムが正常に実行されました"
    echo "終了コード: $EXIT_CODE"
else
    echo "✓ プログラムが正常に実行されました"
    echo "計算結果（終了コード）: $EXIT_CODE"
fi

# アセンブリファイルの内容を表示（オプション）
if [ "${SHOW_ASM:-}" = "1" ]; then
    echo
    echo "=== 生成されたアセンブリコード ==="
    cat "$ASM_FILE"
fi

# 一時ファイルのクリーンアップ（オプション）
if [ "${KEEP_FILES:-}" != "1" ]; then
    echo
    echo "一時ファイルをクリーンアップ中..."
    rm -f "$ASM_FILE" "$EXE_FILE"
    echo "✓ クリーンアップ完了"
fi

echo
echo "=== コンパイル・実行完了 ==="
echo "終了コード: $EXIT_CODE"