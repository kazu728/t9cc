use std::sync::OnceLock;

static FILENAME: OnceLock<String> = OnceLock::new();

pub fn set_filename(filename: &str) {
    FILENAME.set(filename.to_string()).ok();
}

pub fn error_at(loc: &str, input: &str, message: &str) {
    let pos_bytes = loc.as_ptr() as usize - input.as_ptr() as usize;
    let input_bytes = input.as_bytes();

    let mut line_start = pos_bytes;
    while line_start > 0 && input_bytes[line_start - 1] != b'\n' {
        line_start -= 1;
    }

    let mut line_end = pos_bytes;
    while line_end < input_bytes.len() && input_bytes[line_end] != b'\n' {
        line_end += 1;
    }

    let mut line_num = 1;
    for i in 0..line_start {
        if input_bytes[i] == b'\n' {
            line_num += 1;
        }
    }

    let default_filename = "<input>".to_string();
    let filename = FILENAME.get().unwrap_or(&default_filename);
    let line_content = &input[line_start..line_end];

    let indent = format!("{}:{}: ", filename, line_num).len();
    eprintln!("{}:{}: {}", filename, line_num, line_content);

    let pos_in_line = pos_bytes - line_start;
    let spaces = " ".repeat(indent + pos_in_line);
    eprintln!("{}^ {}", spaces, message);

    panic!("{}", message);
}
