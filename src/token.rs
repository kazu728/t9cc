use std::fmt;
use std::iter::Peekable;

#[derive(Clone, Debug, PartialEq)]
pub enum TokenizeError {
    UnexpectedCharacter {
        pos: usize,
        ch: char,
    },
    UnterminatedCharLiteral {
        pos: usize,
    },
    UnterminatedStringLiteral {
        pos: usize,
    },
    UnterminatedBlockComment {
        pos: usize,
    },
    InvalidCharLiteral {
        pos: usize,
    },
    InvalidNumberLiteral {
        pos: usize,
    },
    UnexpectedEndOfInput,
    UnexpectedToken {
        pos: usize,
        expected: String,
        found: String,
    },
    ExpectedIdentifier {
        pos: usize,
    },
}

impl fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizeError::UnexpectedCharacter { pos: _, ch } => {
                write!(f, "予期しない文字: '{}'", ch)
            }
            TokenizeError::UnterminatedCharLiteral { pos: _ } => {
                write!(f, "文字リテラルの終端がありません")
            }
            TokenizeError::UnterminatedStringLiteral { pos: _ } => {
                write!(f, "文字列リテラルの終端がありません")
            }
            TokenizeError::UnterminatedBlockComment { pos: _ } => {
                write!(f, "コメントが閉じられていません")
            }
            TokenizeError::InvalidCharLiteral { pos: _ } => {
                write!(f, "不正な文字リテラル")
            }
            TokenizeError::InvalidNumberLiteral { pos: _ } => {
                write!(f, "不正な数値リテラル")
            }
            TokenizeError::UnexpectedEndOfInput => {
                write!(f, "予期しない入力の終端です")
            }
            TokenizeError::UnexpectedToken {
                pos: _,
                expected,
                found,
            } => {
                write!(
                    f,
                    "予期しないトークン: {} ではなく {} が必要です",
                    found, expected
                )
            }
            TokenizeError::ExpectedIdentifier { pos: _ } => {
                write!(f, "識別子が必要です")
            }
        }
    }
}

impl TokenizeError {
    pub fn pos(&self) -> Option<usize> {
        match self {
            TokenizeError::UnexpectedCharacter { pos, .. }
            | TokenizeError::UnterminatedCharLiteral { pos }
            | TokenizeError::UnterminatedStringLiteral { pos }
            | TokenizeError::UnterminatedBlockComment { pos }
            | TokenizeError::InvalidCharLiteral { pos }
            | TokenizeError::InvalidNumberLiteral { pos }
            | TokenizeError::UnexpectedToken { pos, .. }
            | TokenizeError::ExpectedIdentifier { pos } => Some(*pos),
            TokenizeError::UnexpectedEndOfInput => None,
        }
    }
}

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

    Number(i32),           // 整数トークン
    CharLiteral(char),     // 文字リテラル
    StringLiteral(String), // 文字列リテラル
    Identifier(&'a str),   // 識別子

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
}

impl<'a> Token<'a> {
    pub fn new(kind: TokenKind<'a>, input: &'a str, next: MaybeToken<'a>) -> Token<'a> {
        Token { kind, next, input }
    }

    pub fn init(kind: TokenKind<'a>, input: &'a str) -> Token<'a> {
        Token {
            kind,
            next: None,
            input,
        }
    }
}

pub struct Tokenizer<'a> {
    input: &'a str,
    chars: Peekable<std::str::CharIndices<'a>>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer {
            input,
            chars: input.char_indices().peekable(),
        }
    }

    pub fn tokenize(&mut self) -> Result<Token<'a>, TokenizeError> {
        let mut head = Token::init(TokenKind::Begin, "");
        let mut current = &mut head;

        while let Some((i, c)) = self.chars.peek().copied() {
            if c.is_whitespace() {
                self.chars.next();
                continue;
            }

            if c == '/' {
                if self.parse_comment(i)?.is_some() {
                    continue;
                }
            }

            let next_token = match c {
                '0'..='9' => self.parse_number(i)?,
                'a'..='z' | 'A'..='Z' | '_' => self.parse_identifier(i)?,
                '\'' => self.parse_char_literal(i)?,
                '"' => self.parse_string_literal(i)?,
                '+' => self.parse_single_char_op(i, TokenKind::Plus)?,
                '-' => self.parse_single_char_op(i, TokenKind::Minus)?,
                '*' => self.parse_single_char_op(i, TokenKind::Star)?,
                '/' => self.parse_single_char_op(i, TokenKind::Slash)?,
                '(' => self.parse_single_char_op(i, TokenKind::LParen)?,
                ')' => self.parse_single_char_op(i, TokenKind::RParen)?,
                '{' => self.parse_single_char_op(i, TokenKind::LBrace)?,
                '}' => self.parse_single_char_op(i, TokenKind::RBrace)?,
                '[' => self.parse_single_char_op(i, TokenKind::LBracket)?,
                ']' => self.parse_single_char_op(i, TokenKind::RBracket)?,
                ';' => self.parse_single_char_op(i, TokenKind::Semicolon)?,
                ',' => self.parse_single_char_op(i, TokenKind::Comma)?,
                '=' => self.parse_multiple_char_op(i, TokenKind::Assign, TokenKind::Equal)?,
                '>' => {
                    self.parse_multiple_char_op(i, TokenKind::Greater, TokenKind::GreaterEqual)?
                }
                '<' => self.parse_multiple_char_op(i, TokenKind::Less, TokenKind::LessEqual)?,
                '&' => self.parse_single_char_op(i, TokenKind::Ampersand)?,
                '!' => self.parse_not_operator(i)?,
                _ => return Err(TokenizeError::UnexpectedCharacter { pos: i, ch: c }),
            };

            current.next = Some(Box::new(next_token));
            current = current.next.as_mut().unwrap();
        }

        Ok(*head.next.take().unwrap())
    }

    fn is_alphanumeric(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    fn parse_number(&mut self, i: usize) -> Result<Token<'a>, TokenizeError> {
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

        let number_str = &self.input[start..end];
        let number = number_str
            .parse()
            .map_err(|_| TokenizeError::InvalidNumberLiteral { pos: start })?;
        Ok(Token::init(TokenKind::Number(number), number_str))
    }

    fn parse_identifier(&mut self, i: usize) -> Result<Token<'a>, TokenizeError> {
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

        let token_kind = match name {
            "int" => TokenKind::Int,
            "char" => TokenKind::Char,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "sizeof" => TokenKind::Sizeof,
            _ => TokenKind::Identifier(name),
        };
        Ok(Token::init(token_kind, name))
    }

    fn parse_single_char_op(
        &mut self,
        i: usize,
        kind: TokenKind<'a>,
    ) -> Result<Token<'a>, TokenizeError> {
        self.chars.next();
        Ok(Token::init(kind, &self.input[i..i + 1]))
    }

    fn parse_multiple_char_op(
        &mut self,
        i: usize,
        single_kind: TokenKind<'a>,
        double_kind: TokenKind<'a>,
    ) -> Result<Token<'a>, TokenizeError> {
        self.chars.next();
        if let Some((_, '=')) = self.chars.peek() {
            self.chars.next();
            Ok(Token::init(double_kind, &self.input[i..i + 2]))
        } else {
            Ok(Token::init(single_kind, &self.input[i..i + 1]))
        }
    }

    fn parse_not_operator(&mut self, i: usize) -> Result<Token<'a>, TokenizeError> {
        self.chars.next();

        if let Some((_, '=')) = self.chars.peek() {
            self.chars.next();
            Ok(Token::init(TokenKind::NotEqual, &self.input[i..i + 2]))
        } else {
            Err(TokenizeError::UnexpectedCharacter { pos: i, ch: '!' })
        }
    }

    fn parse_char_literal(&mut self, i: usize) -> Result<Token<'a>, TokenizeError> {
        self.chars.next();

        if let Some((char_pos, c)) = self.chars.next() {
            if let Some((end_pos, '\'')) = self.chars.next() {
                Ok(Token::init(
                    TokenKind::CharLiteral(c),
                    &self.input[i..end_pos + 1],
                ))
            } else {
                Err(TokenizeError::UnterminatedCharLiteral { pos: char_pos })
            }
        } else {
            Err(TokenizeError::InvalidCharLiteral { pos: i })
        }
    }

    fn parse_string_literal(&mut self, i: usize) -> Result<Token<'a>, TokenizeError> {
        self.chars.next();

        let mut string_content = String::new();

        while let Some((pos, c)) = self.chars.peek().copied() {
            if c == '"' {
                self.chars.next();
                return Ok(Token::init(
                    TokenKind::StringLiteral(string_content),
                    &self.input[i..pos + 1],
                ));
            }
            self.chars.next();
            string_content.push(c);
        }

        Err(TokenizeError::UnterminatedStringLiteral { pos: i })
    }

    fn parse_comment(&mut self, i: usize) -> Result<Option<()>, TokenizeError> {
        let mut chars_iter = self.chars.clone();
        chars_iter.next();

        match chars_iter.peek() {
            Some((_, '/')) => self.parse_line_comment(),
            Some((_, '*')) => self.parse_block_comment(i),
            _ => Ok(None),
        }
    }

    fn parse_line_comment(&mut self) -> Result<Option<()>, TokenizeError> {
        self.chars.next();
        self.chars.next();

        while let Some((_, ch)) = self.chars.peek() {
            if *ch == '\n' {
                break;
            }
            self.chars.next();
        }
        Ok(Some(()))
    }

    fn parse_block_comment(&mut self, start_pos: usize) -> Result<Option<()>, TokenizeError> {
        self.chars.next();
        self.chars.next();

        while let Some((_, ch)) = self.chars.next() {
            if ch == '*' {
                if matches!(self.chars.peek(), Some((_, '/'))) {
                    self.chars.next();
                    return Ok(Some(()));
                }
            }
        }

        Err(TokenizeError::UnterminatedBlockComment { pos: start_pos })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
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
                    Some(Box::new(Token::new(
                        TokenKind::LessEqual,
                        "<=",
                        Some(Box::new(Token::init(TokenKind::Number(2), "2"))),
                    ))),
                ),
            },
            TestCase {
                input: "var1 + var2",
                expected: Token::new(
                    TokenKind::Identifier("var1"),
                    "var1",
                    Some(Box::new(Token::new(
                        TokenKind::Plus,
                        "+",
                        Some(Box::new(Token::init(TokenKind::Identifier("var2"), "var2"))),
                    ))),
                ),
            },
            TestCase {
                input: "a = 1; a = 2",
                expected: Token::new(
                    TokenKind::Identifier("a"),
                    "a",
                    Some(Box::new(Token::new(
                        TokenKind::Assign,
                        "=",
                        Some(Box::new(Token::new(
                            TokenKind::Number(1),
                            "1",
                            Some(Box::new(Token::new(
                                TokenKind::Semicolon,
                                ";",
                                Some(Box::new(Token::new(
                                    TokenKind::Identifier("a"),
                                    "a",
                                    Some(Box::new(Token::new(
                                        TokenKind::Assign,
                                        "=",
                                        Some(Box::new(Token::init(TokenKind::Number(2), "2"))),
                                    ))),
                                ))),
                            ))),
                        ))),
                    ))),
                ),
            },
        ];

        for test_case in test_cases {
            let mut tokenizer = Tokenizer::new(test_case.input);
            let result = tokenizer.tokenize().expect("Tokenization should succeed");

            assert_eq!(result, test_case.expected);
        }
    }

    #[test]
    fn test_parse_number() {
        let mut tokenizer = Tokenizer::new("123");
        let result = tokenizer
            .parse_number(0)
            .expect("Number parsing should succeed");

        assert_eq!(result, Token::init(TokenKind::Number(123), "123"));
    }

    #[test]
    fn test_parse_identifier() {
        struct TestCase<'a> {
            input: &'a str,
            expected: Token<'a>,
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                input: "var1 var2",
                expected: Token::init(TokenKind::Identifier("var1"), "var1"),
            },
            TestCase {
                input: "return",
                expected: Token::init(TokenKind::Return, "return"),
            },
            TestCase {
                input: "return_1",
                expected: Token::init(TokenKind::Identifier("return_1"), "return_1"),
            },
            TestCase {
                input: "if",
                expected: Token::init(TokenKind::If, "if"),
            },
            TestCase {
                input: "while",
                expected: Token::init(TokenKind::While, "while"),
            },
            TestCase {
                input: "for",
                expected: Token::init(TokenKind::For, "for"),
            },
        ];

        for case in cases {
            let mut tokenizer = Tokenizer::new(case.input);
            let result = tokenizer
                .parse_identifier(0)
                .expect("Identifier parsing should succeed");

            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_parse_single_char_op() {
        let mut tokenizer = Tokenizer::new("+");
        let result = tokenizer
            .parse_single_char_op(0, TokenKind::Plus)
            .expect("Single char op parsing should succeed");
        assert_eq!(result, Token::init(TokenKind::Plus, "+"));
    }

    #[test]
    fn test_parse_not_operator() {
        let mut tokenizer = Tokenizer::new("!=");
        let result = tokenizer
            .parse_not_operator(0)
            .expect("Not equal parsing should succeed");
        assert_eq!(result, Token::init(TokenKind::NotEqual, "!="));

        let mut tokenizer = Tokenizer::new("!");
        let result = tokenizer.parse_not_operator(0);
        assert!(result.is_err());
        if let Err(TokenizeError::UnexpectedCharacter { pos, ch }) = result {
            assert_eq!(pos, 0);
            assert_eq!(ch, '!');
        } else {
            panic!("Expected UnexpectedCharacter error");
        }
    }

    #[test]
    fn test_parse_line_comment() {
        let mut tokenizer = Tokenizer::new("// this is a comment\n");
        let result = tokenizer
            .parse_line_comment()
            .expect("Line comment parsing should succeed");
        assert_eq!(result, Some(()));

        assert_eq!(tokenizer.chars.peek().map(|(_, c)| *c), Some('\n'));
    }

    #[test]
    fn test_parse_block_comment() {
        let mut tokenizer = Tokenizer::new("/* this is a block comment */");
        let result = tokenizer
            .parse_block_comment(0)
            .expect("Block comment parsing should succeed");
        assert_eq!(result, Some(()));

        let mut tokenizer = Tokenizer::new("/* unterminated");
        let result = tokenizer.parse_block_comment(0);
        assert!(result.is_err());
        if let Err(TokenizeError::UnterminatedBlockComment { pos }) = result {
            assert_eq!(pos, 0);
        } else {
            panic!("Expected UnterminatedBlockComment error");
        }
    }

    #[test]
    fn test_parse_multiple_char_op() {
        let mut tokenizer = Tokenizer::new("==");
        let result = tokenizer
            .parse_multiple_char_op(0, TokenKind::Assign, TokenKind::Equal)
            .expect("Multiple char op parsing should succeed");

        assert_eq!(result, Token::init(TokenKind::Equal, "=="));
    }

    #[test]
    fn test_parse_char_literal() {
        struct TestCase<'a> {
            input: &'a str,
            expected: Token<'a>,
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                input: "'a'",
                expected: Token::init(TokenKind::CharLiteral('a'), "'a'"),
            },
            TestCase {
                input: "'1'",
                expected: Token::init(TokenKind::CharLiteral('1'), "'1'"),
            },
            TestCase {
                input: "'Z'",
                expected: Token::init(TokenKind::CharLiteral('Z'), "'Z'"),
            },
        ];

        for case in cases {
            let mut tokenizer = Tokenizer::new(case.input);
            let result = tokenizer
                .parse_char_literal(0)
                .expect("Char literal parsing should succeed");
            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_tokenize_with_comments() {
        struct TestCase<'a> {
            input: &'a str,
            expected_token_count: usize,
            description: &'a str,
        }

        let test_cases = vec![
            TestCase {
                input: "return 42; // line comment\n",
                expected_token_count: 3, // return, 42, ;
                description: "line comment should be skipped",
            },
            TestCase {
                input: "return /* block comment */ 42;",
                expected_token_count: 3, // return, 42, ;
                description: "block comment should be skipped",
            },
            TestCase {
                input: "42 // comment",
                expected_token_count: 1, // 42
                description: "line comment at end without newline",
            },
        ];

        for test_case in test_cases {
            let mut tokenizer = Tokenizer::new(test_case.input);
            let token = tokenizer.tokenize().expect("Tokenization should succeed");
            let mut count = 0;
            let mut current = Some(Box::new(token));

            while let Some(t) = current {
                current = t.next;
                count += 1;
            }

            assert_eq!(
                count, test_case.expected_token_count,
                "Failed for test case: {}",
                test_case.description
            );
        }
    }

    #[test]
    fn test_line_comment_debug() {
        let input = "int main() { return 42; // comment\n}";
        let mut tokenizer = Tokenizer::new(input);
        let token = tokenizer.tokenize().expect("Tokenization should succeed");

        let mut current = Some(Box::new(token));
        let mut tokens = Vec::new();

        while let Some(t) = current {
            tokens.push(format!("{:?}", t.kind));
            current = t.next;
        }

        println!("Tokens: {:?}", tokens);
        // This should produce: [Int, Identifier("main"), LParen, RParen, LBrace, Return, Number(42), Semicolon, RBrace]
    }
}
