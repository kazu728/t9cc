use std::io::Write;

#[derive(Clone, Debug, PartialEq)]
enum TokenKind {
    Reserved(char), // 記号
    Number(i32),    // 整数トークン
    Begin,          // 入力の始まりを表すトークン, 入力の終わりはNoneで表すため定義しない
}

#[derive(Clone, Debug, PartialEq)]
struct Token<'a> {
    kind: TokenKind,              // トークンの型
    next: Option<Box<Token<'a>>>, // 次の入力トークン
    input: &'a str,               // トークン文字列
}

enum ASTNodeKind {
    Add,
    Sub,
    Mul,
    Div,
    Num(i32),
}

struct ASTNode {
    kind: ASTNodeKind,
    lhs: Option<Box<ASTNode>>,
    rhs: Option<Box<ASTNode>>,
}

impl ASTNode {
    fn new(kind: ASTNodeKind) -> ASTNode {
        ASTNode {
            kind,
            lhs: None,
            rhs: None,
        }
    }

    fn new_num(n: i32) -> ASTNode {
        ASTNode {
            kind: ASTNodeKind::Num(n),
            lhs: None,
            rhs: None,
        }
    }
}

impl<'a> Token<'a> {
    fn new(kind: TokenKind, input: &str) -> Token {
        Token {
            kind,
            next: None,
            input,
        }
    }
}

fn tokenize(input: &str) -> Token {
    let mut head = Token::new(TokenKind::Begin, "");
    let mut current = &mut head;

    let mut chars = input.char_indices().peekable();

    while let Some((i, c)) = chars.peek().copied() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        let next_token = match c {
            '0'..='9' => {
                let start = i;
                while let Some((_, c)) = chars.peek() {
                    if c.is_ascii_digit() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|(i, _)| *i).unwrap_or(input.len());
                Some(Box::new(Token::new(
                    TokenKind::Number(input[start..end].parse().unwrap()),
                    &input[start..end],
                )))
            }
            '+' | '-' => {
                chars.next();
                Some(Box::new(Token::new(
                    TokenKind::Reserved(c),
                    &input[i..i + 1],
                )))
            }
            _ => unimplemented!(),
        };

        current.next = next_token;
        current = current.next.as_mut().unwrap();
    }

    *head.next.take().unwrap()
}
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

fn error_at(loc: &str, input: &str, message: &str) {
    let pos_bytes = loc.as_ptr() as usize - input.as_ptr() as usize;
    let pos_chars = input[..pos_bytes].chars().count();

    eprintln!("{}", input);
    eprintln!("{:>1$}", "^", pos_chars + 1);
    eprintln!("error: {}", message);
    std::process::exit(1);
}

/**
 * 次のトークンが数値の場合、トークンを1つ読み進めてその数値を返す
 * トークンが数値でない場合はエラーを出力して終了する
 */
fn expect_number(current_token: &mut Token, input: &str) -> i32 {
    // TODO: ファイル分割するあたりでResult返すように修正したい
    match current_token.next {
        Some(ref mut token) => match token.kind {
            TokenKind::Number(n) => {
                current_token.next = token.next.take();
                n
            }
            _ => {
                error_at(&current_token.input, input, "数値ではありません");
                std::process::exit(1);
            }
        },
        None => {
            error_at(&current_token.input, input, "数値ではありません");
            std::process::exit(1);
        }
    }
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

    #[test]
    fn test_tokenize() {
        let input = "5 + 20 - 4";
        let result = tokenize(input);

        fn new_token<'a>(
            kind: TokenKind,
            input: &'a str,
            next: Option<Box<Token<'a>>>,
        ) -> Token<'a> {
            Token { kind, next, input }
        }

        let token = new_token(
            TokenKind::Number(5),
            "5",
            Some(Box::new(new_token(
                TokenKind::Reserved('+'),
                "+",
                Some(Box::new(new_token(
                    TokenKind::Number(20),
                    "20",
                    Some(Box::new(new_token(
                        TokenKind::Reserved('-'),
                        "-",
                        Some(Box::new(Token::new(TokenKind::Number(4), "4"))),
                    ))),
                ))),
            ))),
        );

        assert_eq!(result, token);
    }
}
