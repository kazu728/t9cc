use std::io::Write;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    print!("{:?}", args);
    if args.len() != 2 {
        eprintln!("引数の個数が正しくありません");
        std::process::exit(1);
    }
    let p = args[1].as_str();
    let asm = generate_asm(p);

    write_asm("build/tmp.s", asm).unwrap();
}

fn write_asm(path: &str, asm: String) -> Result<(), std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::create(path)?;
    file.write_all(asm.as_bytes())?;
    Ok(())
}

fn generate_asm(input: &str) -> String {
    let mut asm = String::new();
    asm.push_str(".intel_syntax noprefix\n");
    asm.push_str(".global main\n");
    asm.push_str("main:\n");

    let mut p = input.as_bytes().iter().peekable();
    while let Some(&c) = p.peek() {
        match c {
            b'0'..=b'9' => asm.push_str(&format!("  mov rax, {}\n", build_integer_string(&mut p))),
            b'+' => {
                asm.push_str("  add rax, ");
                p.next();
                asm.push_str(&format!("{}\n", build_integer_string(&mut p)));
            }
            b'-' => {
                asm.push_str("  sub rax, ");
                p.next();
                asm.push_str(&format!("{}\n", build_integer_string(&mut p)));
            }
            _ => unimplemented!(),
        }
    }

    asm.push_str("  ret\n");
    asm
}

fn build_integer_string(p: &mut std::iter::Peekable<std::slice::Iter<u8>>) -> String {
    let mut ascii_digit_string = String::new();

    while let Some(&c) = p.peek() {
        if c.is_ascii_digit() {
            ascii_digit_string.push(*c as char);
            p.next();
        } else {
            break;
        }
    }

    ascii_digit_string
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generate_asm() {
        let input = "5+20-4";

        let mut result = String::new();
        result.push_str(".intel_syntax noprefix\n");
        result.push_str(".global main\n");
        result.push_str("main:\n");
        result.push_str("  mov rax, 5\n");
        result.push_str("  add rax, 20\n");
        result.push_str("  sub rax, 4\n");
        result.push_str("  ret\n");

        assert_eq!(super::generate_asm(input), result);
    }
}
