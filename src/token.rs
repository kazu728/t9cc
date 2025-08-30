use super::error::error_at;

const INITIAL_IDENTIFIER_OFFSET: u32 = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind<'a> {
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Equal,        // ==
    NotEqual,     // !=
    Less,         // <
    LessEqual,    // <=
    Greater,      // >
    GreaterEqual, // >=
    Assign,       // =
    Ampersand,    // &

    LParen,    // (
    RParen,    // )
    LBrace,    // {
    RBrace,    // }
    LBracket,  // [
    RBracket,  // ]
    Semicolon, // ;
    Comma,     // ,

    Number(i32),                   // 整数トークン
    CharLiteral(char),             // 文字リテラル
    StringLiteral(String),         // 文字列リテラル
    Identifier(LocalVariable<'a>), // 識別子

    Int,    // int型宣言
    Char,   // char型宣言
    Return, // return文
    If,     // if文
    Else,   // else文
    While,  // while文
    For,    // for文
    Sizeof, // sizeof演算子

    Begin, // 入力の始まりを表すトークン
}

type MaybeToken<'a> = Option<Box<Token<'a>>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,  // トークンの型
    pub next: MaybeToken<'a>, // 次の入力トークン、最後の入力トークン以外は Some(Box<Token>), 最後のトークンは None
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

    pub fn get_name(&self) -> &'a str {
        self.name
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

    pub fn new_maybe_token(
        kind: TokenKind<'a>,
        input: &'a str,
        next: MaybeToken<'a>,
    ) -> MaybeToken<'a> {
        Some(Box::new(Token::new(kind, input, next)))
    }

    /**
     * 現在のトークンが指定されたトークン種別かどうか確認する
     * トークンが指定された種別であればトークンを消費して次に進む
     */
    pub fn consume(maybe_token: &mut MaybeToken<'a>, kind: TokenKind<'a>) -> bool {
        if let Some(token) = maybe_token {
            if token.kind == kind {
                *maybe_token = token.next.take();
                return true;
            }
        }
        false
    }

    fn expect_token<T, F>(
        token: &mut MaybeToken<'a>,
        input: &str,
        check: F,
        error_msg: &str,
    ) -> Option<T>
    where
        F: FnOnce(&Token<'a>) -> Option<T>,
    {
        if let Some(tok) = token {
            if let Some(result) = check(tok) {
                *token = tok.next.take();
                return Some(result);
            }
            error_at(tok.input, input, error_msg);
        } else {
            error_at("", input, "予期しないトークンの終端です");
        }
        None
    }

    /**
     * 次のトークンが指定された種別であることを確認する
     * トークンが指定された種別であればトークンを消費して次に進む
     * トークンが指定された種別でない場合はエラーを出力して終了する
     */
    pub fn expect(token: &mut MaybeToken<'a>, kind: TokenKind<'a>, input: &str) {
        Self::expect_token(
            token,
            input,
            |tok| if tok.kind == kind { Some(()) } else { None },
            &format!("{:?} ではありません", kind),
        );
    }

    pub fn expect_identifier(token: &mut MaybeToken<'a>, input: &str) -> String {
        Self::expect_token(
            token,
            input,
            |tok| match &tok.kind {
                TokenKind::Identifier(_) => Some(tok.input.to_string()),
                _ => None,
            },
            "識別子が必要です",
        )
        .unwrap_or_else(|| unreachable!())
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
                'a'..='z' | 'A'..='Z' | '_' => self.parse_identifier(i),
                '\'' => self.parse_char_literal(i),
                '"' => self.parse_string_literal(i),
                '+' => self.parse_single_char_op(i, TokenKind::Plus),
                '-' => self.parse_single_char_op(i, TokenKind::Minus),
                '*' => self.parse_single_char_op(i, TokenKind::Star),
                '/' => self.parse_single_char_op(i, TokenKind::Slash),
                '(' => self.parse_single_char_op(i, TokenKind::LParen),
                ')' => self.parse_single_char_op(i, TokenKind::RParen),
                '{' => self.parse_single_char_op(i, TokenKind::LBrace),
                '}' => self.parse_single_char_op(i, TokenKind::RBrace),
                '[' => self.parse_single_char_op(i, TokenKind::LBracket),
                ']' => self.parse_single_char_op(i, TokenKind::RBracket),
                ';' => self.parse_single_char_op(i, TokenKind::Semicolon),
                ',' => self.parse_single_char_op(i, TokenKind::Comma),
                '=' => self.parse_multiple_char_op(i, TokenKind::Assign, TokenKind::Equal),
                '>' => self.parse_multiple_char_op(i, TokenKind::Greater, TokenKind::GreaterEqual),
                '<' => self.parse_multiple_char_op(i, TokenKind::Less, TokenKind::LessEqual),
                '&' => self.parse_single_char_op(i, TokenKind::Ampersand),
                '!' => {
                    self.chars.next();
                    if let Some((_, '=')) = self.chars.peek() {
                        self.chars.next();
                        Token::new_maybe_token(TokenKind::NotEqual, &self.input[i..i + 2], None)
                    } else {
                        unimplemented!("未実装のトークン: '!'");
                    }
                }
                _ => unimplemented!("未実装のトークン: '{}'", c),
            };

            current.next = next_token;
            current = current.next.as_mut().unwrap();
        }

        *head.next.take().unwrap()
    }

    fn is_alphanumeric(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
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
            None,
        )
    }

    fn parse_identifier(&mut self, i: usize) -> MaybeToken<'a> {
        self.chars.next();
        let start = i;
        while let Some((_, c)) = self.chars.peek() {
            if Self::is_alphanumeric(*c) {
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

        match name {
            "int" => Token::new_maybe_token(TokenKind::Int, name, None),
            "char" => Token::new_maybe_token(TokenKind::Char, name, None),
            "return" => Token::new_maybe_token(TokenKind::Return, name, None),
            "if" => Token::new_maybe_token(TokenKind::If, name, None),
            "else" => Token::new_maybe_token(TokenKind::Else, name, None),
            "while" => Token::new_maybe_token(TokenKind::While, name, None),
            "for" => Token::new_maybe_token(TokenKind::For, name, None),
            "sizeof" => Token::new_maybe_token(TokenKind::Sizeof, name, None),
            _ => {
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
                    None,
                )
            }
        }
    }

    fn find_variable(&self, name: &str) -> Option<u32> {
        self.variable_environment
            .iter()
            .find(|(var_name, _)| *var_name == name)
            .map(|(_, offset)| *offset)
    }

    fn parse_single_char_op(&mut self, i: usize, kind: TokenKind<'a>) -> MaybeToken<'a> {
        self.chars.next();
        Token::new_maybe_token(kind, &self.input[i..i + 1], None)
    }

    fn parse_multiple_char_op(
        &mut self,
        i: usize,
        single_kind: TokenKind<'a>,
        double_kind: TokenKind<'a>,
    ) -> MaybeToken<'a> {
        self.chars.next();
        if let Some((_, '=')) = self.chars.peek() {
            self.chars.next();
            Token::new_maybe_token(double_kind, &self.input[i..i + 2], None)
        } else {
            Token::new_maybe_token(single_kind, &self.input[i..i + 1], None)
        }
    }

    fn parse_char_literal(&mut self, i: usize) -> MaybeToken<'a> {
        self.chars.next();

        if let Some((char_pos, c)) = self.chars.next() {
            if let Some((end_pos, '\'')) = self.chars.next() {
                Token::new_maybe_token(TokenKind::CharLiteral(c), &self.input[i..end_pos + 1], None)
            } else {
                error_at(
                    &self.input[char_pos..],
                    self.input,
                    "文字リテラルの終端がありません",
                );
                None
            }
        } else {
            error_at(&self.input[i..], self.input, "不正な文字リテラル");
            None
        }
    }

    fn parse_string_literal(&mut self, i: usize) -> MaybeToken<'a> {
        self.chars.next();

        let mut string_content = String::new();

        while let Some((pos, c)) = self.chars.peek().copied() {
            if c == '"' {
                self.chars.next();
                return Token::new_maybe_token(
                    TokenKind::StringLiteral(string_content),
                    &self.input[i..pos + 1],
                    None,
                );
            }
            self.chars.next();
            string_content.push(c);
        }

        error_at(
            &self.input[i..],
            self.input,
            "文字列リテラルの終端がありません",
        );
        None
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_consume() {
        let cases = vec![
            (
                Token::new_maybe_token(TokenKind::Plus, "+", None),
                TokenKind::Plus,
                true,
                None,
            ),
            (
                Token::new_maybe_token(TokenKind::Minus, "-", None),
                TokenKind::Plus,
                false,
                Token::new_maybe_token(TokenKind::Minus, "-", None),
            ),
            (
                Token::new_maybe_token(TokenKind::Number(1), "1", None),
                TokenKind::Plus,
                false,
                Token::new_maybe_token(TokenKind::Number(1), "1", None),
            ),
        ];
        for (input, test_kind, expected, next) in cases {
            let mut token = input;
            let result = Token::consume(&mut token, test_kind);

            assert_eq!(result, expected);
            assert_eq!(token, next);
        }
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
                        TokenKind::Plus,
                        "+",
                        Some(Box::new(Token::new(
                            TokenKind::Number(20),
                            "20",
                            Some(Box::new(Token::new(
                                TokenKind::Minus,
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
                    Token::new_maybe_token(
                        TokenKind::LessEqual,
                        "<=",
                        Token::new_maybe_token(TokenKind::Number(2), "2", None),
                    ),
                ),
            },
            TestCase {
                input: "var1 + var2",
                expected: Token::new(
                    TokenKind::Identifier(LocalVariable::new("var1", 8)),
                    "var1",
                    Token::new_maybe_token(
                        TokenKind::Plus,
                        "+",
                        Token::new_maybe_token(
                            TokenKind::Identifier(LocalVariable::new("var2", 16)),
                            "var2",
                            None,
                        ),
                    ),
                ),
            },
            TestCase {
                input: "a = 1; a = 2",
                expected: Token::new(
                    TokenKind::Identifier(LocalVariable::new("a", 8)),
                    "a",
                    Token::new_maybe_token(
                        TokenKind::Assign,
                        "=",
                        Token::new_maybe_token(
                            TokenKind::Number(1),
                            "1",
                            Token::new_maybe_token(
                                TokenKind::Semicolon,
                                ";",
                                Token::new_maybe_token(
                                    TokenKind::Identifier(LocalVariable::new("a", 8)), // 同じオフセット
                                    "a",
                                    Token::new_maybe_token(
                                        TokenKind::Assign,
                                        "=",
                                        Token::new_maybe_token(TokenKind::Number(2), "2", None),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            },
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
        struct TestCase<'a> {
            input: &'a str,
            expected: Option<Box<Token<'a>>>,
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                input: "var1 var2",
                expected: Token::new_maybe_token(
                    TokenKind::Identifier(LocalVariable::new("var1", 8)),
                    "var1",
                    None,
                ),
            },
            TestCase {
                input: "return",
                expected: Token::new_maybe_token(TokenKind::Return, "return", None),
            },
            TestCase {
                input: "return_1",
                expected: Token::new_maybe_token(
                    TokenKind::Identifier(LocalVariable::new("return_1", 8)),
                    "return_1",
                    None,
                ),
            },
            TestCase {
                input: "if",
                expected: Token::new_maybe_token(TokenKind::If, "if", None),
            },
            TestCase {
                input: "while",
                expected: Token::new_maybe_token(TokenKind::While, "while", None),
            },
            TestCase {
                input: "for",
                expected: Token::new_maybe_token(TokenKind::For, "for", None),
            },
        ];

        for case in cases {
            let mut tokenizer = Tokenizer::new(case.input);
            let result = tokenizer.parse_identifier(0);

            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_parse_single_char_op() {
        let mut tokenizer = Tokenizer::new("+");
        let result = tokenizer.parse_single_char_op(0, TokenKind::Plus);
        assert_eq!(result, Token::new_maybe_token(TokenKind::Plus, "+", None));
    }

    #[test]
    fn test_parse_multiple_char_op() {
        let mut tokenizer = Tokenizer::new("==");
        let result = tokenizer.parse_multiple_char_op(0, TokenKind::Assign, TokenKind::Equal);

        assert_eq!(result, Token::new_maybe_token(TokenKind::Equal, "==", None));
    }

    #[test]
    fn test_parse_char_literal() {
        struct TestCase<'a> {
            input: &'a str,
            expected: Option<Box<Token<'a>>>,
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                input: "'a'",
                expected: Token::new_maybe_token(TokenKind::CharLiteral('a'), "'a'", None),
            },
            TestCase {
                input: "'1'",
                expected: Token::new_maybe_token(TokenKind::CharLiteral('1'), "'1'", None),
            },
            TestCase {
                input: "'Z'",
                expected: Token::new_maybe_token(TokenKind::CharLiteral('Z'), "'Z'", None),
            },
        ];

        for case in cases {
            let mut tokenizer = Tokenizer::new(case.input);
            let result = tokenizer.parse_char_literal(0);
            assert_eq!(result, case.expected);
        }
    }
}
