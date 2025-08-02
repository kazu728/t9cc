use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::process::Command;

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
    let test_file =
        fs::read_to_string("test/test_cases.toml").expect("test/test_cases.tomlの読み込みに失敗しました");

    let config: TestConfig =
        toml::from_str(&test_file).expect("test_cases.tomlのパースに失敗しました");

    let mut results = Vec::new();
    let total_tests = config.test_cases.len();

    println!("e2e実行中...\n");

    for test_case in config.test_cases {
        print!("実行中: {} ... ", test_case.name);
        std::io::stdout().flush().unwrap();

        let result = run_test(&test_case);

        if result.passed {
            println!("✓");
        } else {
            println!("✗");
        }

        results.push(result);
    }

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

fn run_test(test_case: &TestCase) -> TestResult {
    let asm_file = format!("build/{}.s", test_case.name);
    let exe_file = format!("build/{}", test_case.name);

    let output = match Command::new("cargo")
        .args(&["run", "--bin", "t9cc", "--", &test_case.input])
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
                error: Some(format!("cargo run失敗: {}", e)),
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

    if let Err(e) = fs::rename("build/tmp.s", &asm_file) {
        return TestResult {
            name: test_case.name.clone(),
            input: test_case.input.clone(),
            expected: test_case.expected_output,
            actual: None,
            passed: false,
            error: Some(format!("ファイル移動失敗: {}", e)),
        };
    }

    let compile_output = match Command::new("orb")
        .args(&["-m", "ubuntu", "exec", "gcc", "-o", &exe_file, &asm_file])
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
        .args(&["-m", "ubuntu", "exec", &exe_file])
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
            truncate(&result.input, 33),
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

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
