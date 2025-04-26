use std::io::Write;
use t9cc::token::{expect_number, tokenize, Token, TokenKind};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    print!("{:?}", args);
    if args.len() != 2 {
        eprintln!("引数の個数が正しくありません");
        std::process::exit(1);
    }
    let input = args[1].as_str();
    let token = tokenize(input);
    let asm = generate_asm(token, input);

    write_asm("build/tmp.s", asm).unwrap();
}

fn write_asm(path: &str, asm: String) -> Result<(), std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::create(path)?;
    file.write_all(asm.as_bytes())?;
    Ok(())
}

fn generate_asm(token: Token, input: &str) -> String {
    let mut output = String::new();

    output.push_str(".intel_syntax noprefix\n");
    output.push_str(".global main\n");
    output.push_str("main:\n");

    let mut token = &mut Some(Box::new(token));

    while let Some(t) = token {
        let asm = match t.kind {
            TokenKind::Number(n) => format!("  mov rax, {}\n", n),
            TokenKind::Reserved('+') => format!("  add rax, {}\n", expect_number(t, input)),
            TokenKind::Reserved('-') => format!("  sub rax, {}\n", expect_number(t, input)),
            _ => unimplemented!(),
        };

        output.push_str(asm.as_str());
        token = &mut t.next;
    }

    output.push_str("  ret\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_asm() {
        let input = "5+20-4";

        let token = tokenize(input);

        let mut asm = String::new();
        asm.push_str(".intel_syntax noprefix\n");
        asm.push_str(".global main\n");
        asm.push_str("main:\n");
        asm.push_str("  mov rax, 5\n");
        asm.push_str("  add rax, 20\n");
        asm.push_str("  sub rax, 4\n");
        asm.push_str("  ret\n");

        assert_eq!(generate_asm(token, input), asm);
    }
}
