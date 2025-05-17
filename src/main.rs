use std::io::Write;
use t9cc::asm_generator::gen_asm;
use t9cc::parser::program;
use t9cc::token::Tokenizer;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    print!("{:?}", args);
    if args.len() != 2 {
        eprintln!("引数の個数が正しくありません");
        std::process::exit(1);
    }
    let input = args[1].as_str();
    let token = Tokenizer::new(input).tokenize();

    let programs = program(&mut Some(Box::new(token)), input);

    let mut output = String::new();

    output.push_str(".intel_syntax noprefix\n");
    output.push_str(".global main\n");
    output.push_str("main:\n");

    output.push_str("  push rbp\n");
    output.push_str("  mov rbp, rsp\n");
    output.push_str("  sub rsp, 208\n");

    for program in programs {
        gen_asm(program, &mut output);
    }

    output.push_str("  mov rsp, rbp\n");
    output.push_str("  pop rbp\n");
    output.push_str("  ret\n");

    write_asm("build/tmp.s", output).unwrap();
}

fn write_asm(path: &str, asm: String) -> Result<(), std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::create(path)?;
    file.write_all(asm.as_bytes())?;
    Ok(())
}
