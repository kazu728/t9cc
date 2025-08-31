use rayon::prelude::*;
use serde::Deserialize;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Deserialize)]
struct TestCase {
    name: String,
    input: String,
    expected_output: i32,
}

#[derive(Debug, Deserialize)]
struct TestConfig {
    test_cases: Vec<TestCase>,
}

#[derive(Debug)]
struct TestResult {
    name: String,
    input: String,
    expected: i32,
    actual: Option<i32>,
    passed: bool,
    error: Option<String>,
}

fn main() {
    let test_file = fs::read_to_string("test/test_cases.toml")
        .expect("test/test_cases.tomlの読み込みに失敗しました");

    let config: TestConfig =
        toml::from_str(&test_file).expect("test_cases.tomlのパースに失敗しました");

    let total_tests = config.test_cases.len();

    println!("事前準備中...");
    if let Err(e) = setup_test_environment() {
        eprintln!("事前準備に失敗しました: {}", e);
        std::process::exit(1);
    }

    println!("e2e実行中...\n");

    let completed = AtomicUsize::new(0);
    let results: Vec<TestResult> = config
        .test_cases
        .par_iter()
        .map(|test_case| {
            let result = run_test(test_case);
            let count = completed.fetch_add(1, Ordering::Relaxed) + 1;

            if result.passed {
                println!(
                    "[{:3}/{:3}] 実行中: {} ... ✓",
                    count, total_tests, test_case.name
                );
            } else {
                println!(
                    "[{:3}/{:3}] 実行中: {} ... ✗",
                    count, total_tests, test_case.name
                );
            }

            result
        })
        .collect();

    print_results(&results);

    let passed_count = results.iter().filter(|r| r.passed).count();
    let failed_count = total_tests - passed_count;

    println!(
        "\n合計: {} テスト, 成功: {}, 失敗: {}",
        total_tests, passed_count, failed_count
    );

    if failed_count > 0 {
        std::process::exit(1);
    }
}

fn setup_test_environment() -> Result<(), String> {
    let helper_obj = "build/test_helper.o";

    let helper_compile = Command::new("orb")
        .args([
            "-m",
            "ubuntu",
            "exec",
            "gcc",
            "-c",
            "-o",
            helper_obj,
            "test/test_helper.c",
        ])
        .output()
        .map_err(|e| format!("test_helper.cのコンパイル失敗: {}", e))?;

    if !helper_compile.status.success() {
        return Err(format!(
            "test_helper.cのコンパイル失敗: {}",
            String::from_utf8_lossy(&helper_compile.stderr)
        ));
    }

    let build_output = Command::new("cargo")
        .args(["build", "--release"])
        .output()
        .map_err(|e| format!("cargo buildに失敗しました: {}", e))?;

    if !build_output.status.success() {
        return Err(format!(
            "cargo buildに失敗しました: {}",
            String::from_utf8_lossy(&build_output.stderr)
        ));
    }

    Ok(())
}

fn run_test(test_case: &TestCase) -> TestResult {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let thread_id = std::thread::current().id();
    let unique_id = format!("{}_{:?}", timestamp, thread_id);

    let source_file = format!("build/src_{}.c", unique_id);
    let asm_file = format!("build/tmp_{}.s", unique_id);
    let exe_file = format!("build/exe_{}", unique_id);
    let helper_obj = "build/test_helper.o";

    // テスト入力を一時ファイルに書き込む
    if let Err(e) = fs::write(&source_file, &test_case.input) {
        return TestResult {
            name: test_case.name.clone(),
            input: test_case.input.clone(),
            expected: test_case.expected_output,
            actual: None,
            passed: false,
            error: Some(format!("ソースファイル書き込み失敗: {}", e)),
        };
    }

    let output = match Command::new("target/release/t9cc")
        .arg(&source_file)
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            return TestResult {
                name: test_case.name.clone(),
                input: test_case.input.clone(),
                expected: test_case.expected_output,
                actual: None,
                passed: false,
                error: Some(format!("t9cc実行失敗: {}", e)),
            };
        }
    };

    if !output.status.success() {
        return TestResult {
            name: test_case.name.clone(),
            input: test_case.input.clone(),
            expected: test_case.expected_output,
            actual: None,
            passed: false,
            error: Some(format!(
                "コンパイラエラー: {}",
                String::from_utf8_lossy(&output.stderr)
            )),
        };
    }

    if let Err(e) = fs::write(&asm_file, &output.stdout) {
        return TestResult {
            name: test_case.name.clone(),
            input: test_case.input.clone(),
            expected: test_case.expected_output,
            actual: None,
            passed: false,
            error: Some(format!("アセンブリファイル書き込み失敗: {}", e)),
        };
    }

    let compile_output = match Command::new("orb")
        .args([
            "-m", "ubuntu", "exec", "gcc", "-o", &exe_file, &asm_file, helper_obj,
        ])
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            return TestResult {
                name: test_case.name.clone(),
                input: test_case.input.clone(),
                expected: test_case.expected_output,
                actual: None,
                passed: false,
                error: Some(format!("ccコンパイル失敗: {}", e)),
            };
        }
    };

    if !compile_output.status.success() {
        return TestResult {
            name: test_case.name.clone(),
            input: test_case.input.clone(),
            expected: test_case.expected_output,
            actual: None,
            passed: false,
            error: Some(format!(
                "アセンブラエラー: {}",
                String::from_utf8_lossy(&compile_output.stderr)
            )),
        };
    }

    let exe_output = match Command::new("orb")
        .args(["-m", "ubuntu", "exec", &exe_file])
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            return TestResult {
                name: test_case.name.clone(),
                input: test_case.input.clone(),
                expected: test_case.expected_output,
                actual: None,
                passed: false,
                error: Some(format!("実行失敗: {}", e)),
            };
        }
    };

    let exit_code = exe_output.status.code().unwrap_or(-1);
    let passed = exit_code == test_case.expected_output;

    let _ = fs::remove_file(&source_file);
    let _ = fs::remove_file(&asm_file);
    let _ = fs::remove_file(&exe_file);

    TestResult {
        name: test_case.name.clone(),
        input: test_case.input.clone(),
        expected: test_case.expected_output,
        actual: Some(exit_code),
        passed,
        error: if passed {
            None
        } else {
            Some(format!(
                "期待値: {}, 実際: {}",
                test_case.expected_output, exit_code
            ))
        },
    }
}

fn print_results(results: &[TestResult]) {
    println!("\n┌─────────────────────────┬───────────────────────────────────┬──────────┬──────────┬────────┐");
    println!("│ テスト名                │ 入力                              │ 期待値   │ 実際     │ 結果   │");
    println!("├─────────────────────────┼───────────────────────────────────┼──────────┼──────────┼────────┤");

    for result in results {
        let actual_str = result
            .actual
            .map(|v| v.to_string())
            .unwrap_or_else(|| "エラー".to_string());

        let status = if result.passed { "✓" } else { "✗" };

        println!(
            "│ {:<23} │ {:<33} │ {:>8} │ {:>8} │ {:^6} │",
            truncate(&result.name, 23),
            format_for_table(&result.input, 33),
            result.expected,
            actual_str,
            status
        );

        if let Some(error) = &result.error {
            println!("│ エラー: {:<91} │", truncate(error, 91));
        }
    }

    println!("└─────────────────────────┴───────────────────────────────────┴──────────┴──────────┴────────┘");
}

fn format_for_table(s: &str, max_len: usize) -> String {
    // Replace newlines and normalize whitespace for table display
    let normalized = s
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    
    if normalized.len() <= max_len {
        normalized
    } else {
        format!("{}...", &normalized[..max_len - 3])
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
