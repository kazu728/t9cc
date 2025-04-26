#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenKind {
    Reserved(char), // 記号
    Number(i32),    // 整数トークン
    Begin,          // 入力の始まりを表すトークン, 入力の終わりはNoneで表すため定義しない
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind,              // トークンの型
    pub next: Option<Box<Token<'a>>>, // 次の入力トークン
    pub input: &'a str, // トークン文字列。エラー出力時等に生の文字列があった方が良いので保持
}

impl<'a> Token<'a> {
    pub fn init(kind: TokenKind, input: &str) -> Token {
        Token {
            kind,
            next: None,
            input,
        }
    }

    pub fn new_token(kind: TokenKind, input: &'a str, next: Option<Box<Token<'a>>>) -> Token<'a> {
        Token { kind, next, input }
    }

    /**
     * 現在のトークンが指定された記号かどうか確認する
     * トークンが指定された記号であればトークンを消費して次に進む
     */
    pub fn consume(token: &mut Option<Box<Token<'a>>>, c: char) -> bool {
        if let Some(tok) = token {
            if let TokenKind::Reserved(ch) = tok.kind {
                if ch == c {
                    *token = tok.next.take();
                    return true;
                }
            }
        }
        false
    }

    /**
     * 次のトークンが数値の場合、トークンを1つ読み進めてその数値を返す
     * トークンが数値でない場合はエラーを出力して終了する
     */
    pub fn expect_number(token: &mut Option<Box<Token<'a>>>, raw_input: &str) -> i32 {
        if let Some(tok) = token {
            if let TokenKind::Number(n) = tok.kind {
                *token = tok.next.take();
                return n;
            }
            error_at(&tok.input, raw_input, "数値ではありません");
        } else {
            error_at("", raw_input, "予期しないトークンの終端です");
        }
        unreachable!()
    }

    /**
     * 次のトークンが指定された値であることを確認する
     * トークンが指定された値であればトークンを消費して次に進む
     * トークンが指定された値でない場合はエラーを出力して終了する
     */
    // TOOO: expect_number といい感じに共通化したい
    pub fn expect(token: &mut Option<Box<Token<'a>>>, c: char, input: &str) {
        if let Some(tok) = token {
            if let TokenKind::Reserved(ch) = tok.kind {
                if ch == c {
                    *token = tok.next.take();
                    return;
                }
            }
            error_at(&tok.input, input, &format!("'{}' ではありません", c));
        } else {
            error_at("", input, "予期しないトークンの終端です");
        }
    }
}

pub fn tokenize(input: &str) -> Token {
    let mut head = Token::init(TokenKind::Begin, "");
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
                Some(Box::new(Token::init(
                    TokenKind::Number(input[start..end].parse().unwrap()),
                    &input[start..end],
                )))
            }
            '+' | '-' | '*' | '/' | '(' | ')' => {
                chars.next();
                Some(Box::new(Token::init(
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

fn error_at(loc: &str, input: &str, message: &str) {
    let pos_bytes = loc.as_ptr() as usize - input.as_ptr() as usize;
    let pos_chars = input[..pos_bytes].chars().count();

    eprintln!("{}", input);
    eprintln!("{:>1$}", "^", pos_chars + 1);

    panic!("{}", message);
}

/**
 * 次のトークンが数値の場合、トークンを1つ読み進めてその数値を返す
 * トークンが数値でない場合はエラーを出力して終了する
 */
pub fn expect_number(current_token: &mut Token, raw_input: &str) -> i32 {
    // TODO: ファイル分割するあたりでResult返すように修正したい
    match current_token.next {
        Some(ref mut token) => match token.kind {
            TokenKind::Number(n) => {
                current_token.next = token.next.take();
                n
            }
            _ => {
                error_at(&current_token.input, raw_input, "数値ではありません");
                std::process::exit(1);
            }
        },
        None => {
            error_at(&current_token.input, raw_input, "数値ではありません");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_consume() {
        let cases = vec![
            (
                Some(Box::new(Token::init(TokenKind::Reserved('+'), "+"))),
                true,
                None,
            ),
            (
                Some(Box::new(Token::init(TokenKind::Reserved('-'), "-"))),
                false,
                Some(Box::new(Token::init(TokenKind::Reserved('-'), "-"))),
            ),
            (
                Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                false,
                Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
            ),
        ];
        for (input, expected, next) in cases {
            let mut token = input;
            let result = Token::consume(&mut token, '+');

            assert_eq!(result, expected);
            assert_eq!(token, next);
        }
    }

    #[test]
    fn test_expect_number() {
        let mut token = Some(Box::new(Token::init(TokenKind::Number(1), "1")));

        let result = Token::expect_number(&mut token, "1");
        assert_eq!(result, 1);
        assert_eq!(token, None);
    }

    #[test]
    #[should_panic(expected = "数値ではありません")]
    fn test_expect_number_should_panic() {
        let mut token = Some(Box::new(Token::init(TokenKind::Reserved('+'), "+")));
        Token::expect_number(&mut token, "+");
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
                        Some(Box::new(Token::init(TokenKind::Number(4), "4"))),
                    ))),
                ))),
            ))),
        );

        assert_eq!(result, token);
    }
}
