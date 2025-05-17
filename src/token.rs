const INITIAL_IDENTIFIER_OFFSET: u32 = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind<'a> {
    Reserved(&'a str),             // 記号
    Number(i32),                   // 整数トークン
    Begin, // 入力の始まりを表すトークン, 入力の終わりはNoneで表すため定義しない
    Identifier(LocalVariable<'a>), // 識別子
}

type MaybeToken<'a> = Option<Box<Token<'a>>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,  // トークンの型
    pub next: MaybeToken<'a>, // 次の入力トークン
    pub input: &'a str,       // トークン文字列。エラー出力時等に生の文字列があった方が良いので保持
    pub len: usize,           // トークンの長さ
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVariable<'a> {
    next: Option<Box<LocalVariable<'a>>>,
    name: &'a str,
    offset: u32,
    len: usize,
}

impl<'a> LocalVariable<'a> {
    pub fn new(name: &'a str, offset: u32) -> LocalVariable<'a> {
        LocalVariable {
            next: None,
            name,
            offset,
            len: name.len(),
        }
    }
    pub fn get_offset(&self) -> u32 {
        self.offset
    }
}

impl<'a> Token<'a> {
    pub fn new(kind: TokenKind<'a>, input: &'a str, next: MaybeToken<'a>) -> Token<'a> {
        Token {
            kind,
            next,
            input,
            len: input.len(),
        }
    }

    pub fn init(kind: TokenKind<'a>, input: &'a str) -> Token<'a> {
        Token {
            kind,
            next: None,
            input,
            len: input.len(),
        }
    }

    pub fn new_maybe_token(kind: TokenKind<'a>, input: &'a str) -> MaybeToken<'a> {
        Some(Box::new(Token::init(kind, input)))
    }

    // pub fn new_token(
    //     kind: TokenKind<'a>,
    //     input: &'a str,
    //     next: MaybeToken<'a>,
    //     len: usize,
    // ) -> Token<'a> {
    //     Token {
    //         kind,
    //         next,
    //         input,
    //         len,
    //     }
    // }

    fn is_valid(&self, op: &str) -> bool {
        self.kind == TokenKind::Reserved(op) && op.len() == self.len && self.input == op
    }

    /**
     * 現在のトークンが指定された記号かどうか確認する
     * トークンが指定された記号であればトークンを消費して次に進む
     */
    pub fn consume(maybe_token: &mut MaybeToken<'a>, op: &str) -> bool {
        if let Some(token) = maybe_token {
            if token.is_valid(op) {
                *maybe_token = token.next.take();
                return true;
            }
        }
        false
    }

    /**
     * 次のトークンが数値の場合、トークンを1つ読み進めてその数値を返す
     * トークンが数値でない場合はエラーを出力して終了する
     */
    pub fn expect_number(token: &mut MaybeToken<'a>, raw_input: &str) -> i32 {
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
    pub fn expect(token: &mut MaybeToken<'a>, c: &str, input: &str) {
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

pub struct Tokenizer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    variable_environment: Vec<(&'a str, u32)>, // (変数名, オフセット)のペア
    next_offset: u32,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer {
            input,
            chars: input.char_indices().peekable(),
            variable_environment: Vec::new(),
            next_offset: INITIAL_IDENTIFIER_OFFSET,
        }
    }

    pub fn tokenize(&mut self) -> Token<'a> {
        let mut head = Token::init(TokenKind::Begin, "");
        let mut current = &mut head;

        while let Some((i, c)) = self.chars.peek().copied() {
            if c.is_whitespace() {
                self.chars.next();
                continue;
            }

            let next_token = match c {
                '0'..='9' => self.parse_number(i),
                'a'..='z' | 'A'..='Z' => self.parse_identifier(i),
                '+' | '-' | '*' | '/' | '(' | ')' => {
                    self.parse_single_char_op(i, &self.input[i..i + 1])
                }
                '=' => self.parse_multiple_char_op(i, "=", "=="),
                '>' => self.parse_multiple_char_op(i, ">", ">="),
                '<' => self.parse_multiple_char_op(i, "<", "<="),
                '!' => self.parse_multiple_char_op(i, "!", "!="),
                _ => unimplemented!("未実装のトークン: '{}'", c),
            };

            current.next = next_token;
            current = current.next.as_mut().unwrap();
        }

        *head.next.take().unwrap()
    }

    fn parse_number(&mut self, i: usize) -> MaybeToken<'a> {
        let start = i;
        while let Some((_, c)) = self.chars.peek() {
            if c.is_ascii_digit() {
                self.chars.next();
            } else {
                break;
            }
        }
        let end = self
            .chars
            .peek()
            .map(|(i, _)| *i)
            .unwrap_or(self.input.len());
        Token::new_maybe_token(
            TokenKind::Number(self.input[start..end].parse().unwrap()),
            &self.input[start..end],
        )
    }

    fn parse_identifier(&mut self, i: usize) -> MaybeToken<'a> {
        self.chars.next();
        let start = i;
        while let Some((_, c)) = self.chars.peek() {
            if c.is_ascii_alphanumeric() {
                self.chars.next();
            } else {
                break;
            }
        }
        let end = self
            .chars
            .peek()
            .map(|(i, _)| *i)
            .unwrap_or(self.input.len());

        let name = &self.input[start..end];

        let offset = if let Some(existing_offset) = self.find_variable(name) {
            existing_offset
        } else {
            let new_offset = self.next_offset;
            self.variable_environment.push((name, new_offset));
            self.next_offset += 8;
            new_offset
        };

        Token::new_maybe_token(
            TokenKind::Identifier(LocalVariable::new(name, offset)),
            name,
        )
    }

    fn find_variable(&self, name: &str) -> Option<u32> {
        self.variable_environment
            .iter()
            .find(|(var_name, _)| *var_name == name)
            .map(|(_, offset)| *offset)
    }

    fn parse_single_char_op(&mut self, i: usize, op: &'a str) -> MaybeToken<'a> {
        self.chars.next();

        Token::new_maybe_token(TokenKind::Reserved(op), &self.input[i..i + 1])
    }

    fn parse_multiple_char_op(
        &mut self,
        i: usize,
        single: &'a str,
        double: &'a str,
    ) -> MaybeToken<'a> {
        self.chars.next();
        if let Some((_, '=')) = self.chars.peek() {
            self.chars.next();
            Token::new_maybe_token(TokenKind::Reserved(double), &self.input[i..i + 2])
        } else {
            Token::new_maybe_token(TokenKind::Reserved(single), &self.input[i..i + 1])
        }
    }
}

fn error_at(loc: &str, input: &str, message: &str) {
    let pos_bytes = loc.as_ptr() as usize - input.as_ptr() as usize;
    let pos_chars = input[..pos_bytes].chars().count();

    eprintln!("{}", input);
    eprintln!("{:>1$}", "^", pos_chars + 1);

    panic!("{}", message);
}

#[cfg(test)]
mod tests {

    use super::*;


    #[test]
    fn test_consume() {
        let cases = vec![
            (
                Token::new_maybe_token(TokenKind::Reserved("+"), "+"),
                true,
                None,
            ),
            (
                Token::new_maybe_token(TokenKind::Reserved("-"), "-"),
                false,
                Token::new_maybe_token(TokenKind::Reserved("-"), "-"),
            ),
            (
                Token::new_maybe_token(TokenKind::Number(1), "1"),
                false,
                Token::new_maybe_token(TokenKind::Number(1), "1"),
            ),
        ];
        for (input, expected, next) in cases {
            let mut token = input;
            let result = Token::consume(&mut token, "+");

            assert_eq!(result, expected);
            assert_eq!(token, next);
        }
    }

    #[test]
    fn test_expect_number() {
        let mut token = Token::new_maybe_token(TokenKind::Number(1), "1");

        let result = Token::expect_number(&mut token, "1");
        assert_eq!(result, 1);
        assert_eq!(token, None);
    }

    #[test]
    #[should_panic(expected = "数値ではありません")]
    fn test_expect_number_should_panic() {
        let mut token = Token::new_maybe_token(TokenKind::Reserved("+"), "+");
        Token::expect_number(&mut token, "+");
    }

    #[test]
    fn test_tokenize() {
        struct TestCase<'a> {
            input: &'a str,
            expected: Token<'a>,
        }

        let test_cases = vec![
            TestCase {
                input: "5 + 20 - 4",
                expected: Token::new(
                    TokenKind::Number(5),
                    "5",
                    Some(Box::new(Token::new(
                        TokenKind::Reserved("+"),
                        "+",
                        Some(Box::new(Token::new(
                            TokenKind::Number(20),
                            "20",
                            Some(Box::new(Token::new(
                                TokenKind::Reserved("-"),
                                "-",
                                Some(Box::new(Token::init(TokenKind::Number(4), "4"))),
                            ))),
                        ))),
                    ))),
                ),
            },
            TestCase {
                input: "1 <= 2",
                expected: Token::new(
                    TokenKind::Number(1),
                    "1",
                    Some(Box::new(Token::new(
                        TokenKind::Reserved("<="),
                        "<=",
                        Some(Box::new(Token::new(TokenKind::Number(2), "2", None))),
                    ))),
                ),
            },
            TestCase {
                input: "var1 + var2",
                expected: Token::new(
                    TokenKind::Identifier(LocalVariable::new("var1", 8)),
                    "var1",
                    Some(Box::new(Token::new(
                        TokenKind::Reserved("+"),
                        "+",
                        Some(Box::new(Token::new(
                            TokenKind::Identifier(LocalVariable::new("var2", 16)),
                            "var2",
                            None,
                        ))),
                    ))),
                ),
            },
            // TestCase {
            //     input: "a = 1; a = 2",
            //     expected: new_token(
            //         TokenKind::Identifier(LocalVariable::new("a", 8)),
            //         "a",
            //         Some(Box::new(new_token(
            //             TokenKind::Reserved("="),
            //             "=",
            //             Some(Box::new(new_token(
            //                 TokenKind::Number(1),
            //                 "1",
            //                 Some(Box::new(new_token(
            //                     TokenKind::Reserved(";"),
            //                     ";",
            //                     Some(Box::new(Token::new(
            //                         TokenKind::Identifier(LocalVariable::new("a", 8)), // 同じオフセット
            //                         "a",
            //                         Some(Box::new(Token::new(
            //                             TokenKind::Reserved("="),
            //                             "=",
            //                             Some(Box::new(Token::new(TokenKind::Number(2), "2", None))),
            //                         ))),
            //                     ))),
            //                 ))),
            //             ))),
            //         ))),
            //     ),
            // },
        ];

        for test_case in test_cases {
            let mut tokenizer = Tokenizer::new(test_case.input);
            let result = tokenizer.tokenize();

            assert_eq!(result, test_case.expected);
        }
    }

    #[test]
    fn test_parse_number() {
        let mut tokenizer = Tokenizer::new("123");
        let result = tokenizer.parse_number(0);

        assert_eq!(
            result,
            Some(Box::new(Token::init(TokenKind::Number(123), "123")))
        );
    }

    #[test]
    fn test_parse_identifier() {
        let mut tokenizer = Tokenizer::new("var1 var2");
        let result = tokenizer.parse_identifier(0);

        let expected =
            Token::new_maybe_token(TokenKind::Identifier(LocalVariable::new("var1", 8)), "var1");

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_single_char_op() {
        let mut tokenizer = Tokenizer::new("+");
        let result = tokenizer.parse_single_char_op(0, "+");
        assert_eq!(
            result,
            Token::new_maybe_token(TokenKind::Reserved("+"), "+")
        );
    }

    #[test]
    fn test_parse_multiple_char_op() {
        let mut tokenizer = Tokenizer::new("==");
        let result = tokenizer.parse_multiple_char_op(0, "=", "==");

        assert_eq!(
            result,
            Token::new_maybe_token(TokenKind::Reserved("=="), "==")
        );
    }
}
