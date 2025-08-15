pub fn error_at(loc: &str, input: &str, message: &str) {
    let pos_bytes = loc.as_ptr() as usize - input.as_ptr() as usize;
    let pos_chars = input[..pos_bytes].chars().count();

    eprintln!("{}", input);
    eprintln!("{:>1$}", "^", pos_chars + 1);

    panic!("{}", message);
}