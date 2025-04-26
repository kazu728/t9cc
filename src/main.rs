use std::io::Write;
use t9cc::{asm_generator::gen_asm, parser::expr, token::tokenize};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    print!("{:?}", args);
    if args.len() != 2 {
        eprintln!("引数の個数が正しくありません");
        std::process::exit(1);
    }
    let input = args[1].as_str();
    let token = tokenize(input);
    let node = expr(&mut Some(Box::new(token)), input);

    let asm = gen_asm(node);

    write_asm("build/tmp.s", asm).unwrap();
}

fn write_asm(path: &str, asm: String) -> Result<(), std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::create(path)?;
    file.write_all(asm.as_bytes())?;
    Ok(())
}
