use std::fs;
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

    let token = Tokenizer::new(&input).tokenize();

    let program_ast = program(&mut Some(Box::new(token)), &input);

    let mut output = String::new();

    output.push_str(".intel_syntax noprefix\n");
    output.push_str(".global main\n\n");

    gen_program_asm(&program_ast, &mut output);

    print!("{}", output);
}

fn read_file(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut contents = fs::read_to_string(path)?;

    if contents.is_empty() || !contents.ends_with('\n') {
        contents.push('\n');
    }

    Ok(contents)
}
