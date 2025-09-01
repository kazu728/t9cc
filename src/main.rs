use std::{fs, io};
use t9cc::asm_generator::gen_program_asm;
use t9cc::error::set_filename;
use t9cc::parser::program;
use t9cc::token::Tokenizer;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 2 {
        std::process::exit(1);
    }

    let filename = &args[1];
    let input = match read_file(filename) {
        Ok(contents) => contents,
        Err(e) => {
            eprintln!("ファイル {} を開けません: {}", filename, e);
            std::process::exit(1);
        }
    };

    set_filename(filename);

    let token = match Tokenizer::new(&input).tokenize() {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Tokenize error: {}", e);
            std::process::exit(1);
        }
    };
    let program_ast = match program(&mut Some(Box::new(token)), &input) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse Error: {}", e);
            std::process::exit(1);
        }
    };

    let mut output = String::new();
    gen_program_asm(&program_ast, &mut output);

    print!("{}", output);
}

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut contents = fs::read_to_string(path)?;

    if contents.is_empty() || !contents.ends_with('\n') {
        contents.push('\n');
    }

    Ok(contents)
}
