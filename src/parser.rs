use super::token::{Token, TokenKind};

#[derive(Debug, PartialEq, Eq)]
pub enum ASTNodeKind {
    Add,
    Sub,
    Mul,
    Div,
    Num(i32),

    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    LocalVariable(u32), // ローカル変数のベースポインタからのオフセット,
    Assign,

    Return,
    If,
    IfBody,
    Else,
    While,
    For,
    ForInit,
    ForUpdate,

    Block,
}

pub type MaybeASTNode = Option<Box<ASTNode>>;

#[derive(Debug, PartialEq, Eq)]
pub struct ASTNode {
    pub kind: ASTNodeKind,
    pub lhs: MaybeASTNode,
    pub rhs: MaybeASTNode,
}

impl ASTNode {
    pub fn new(kind: ASTNodeKind, lhs: MaybeASTNode, rhs: MaybeASTNode) -> ASTNode {
        ASTNode { kind, lhs, rhs }
    }

    pub fn new_boxed(
        kind: ASTNodeKind,
        lhs: Option<Box<ASTNode>>,
        rhs: Option<Box<ASTNode>>,
    ) -> Box<ASTNode> {
        Box::new(ASTNode::new(kind, lhs, rhs))
    }

    pub fn binary(kind: ASTNodeKind, lhs: Box<ASTNode>, rhs: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::new(kind, Some(lhs), Some(rhs)))
    }

    pub fn leaf(kind: ASTNodeKind) -> Box<ASTNode> {
        Box::new(ASTNode::new(kind, None, None))
    }

    pub fn unary(kind: ASTNodeKind, child: Box<ASTNode>) -> Box<ASTNode> {
        Box::new(ASTNode::new(kind, Some(child), None))
    }
}

// program    = stmt*

// stmt       = expr ";"
// | "{" stmt* "}"
// | "if" "(" expr ")" stmt ("else" stmt)?
// | "while" "(" expr ")" stmt
// | "for" "(" expr? ";" expr? ";" expr? ")" stmt
// | "return" expr ";"

// expr       = assign
// assign     = equality ("=" assign)?
// equality   = relational ("==" relational | "!=" relational)*
// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
// add        = mul ("+" mul | "-" mul)*
// mul        = unary ("*" unary | "/" unary)*
// unary      = ("+" | "-")? primary
// primary    = num | ident | "(" expr ")"

pub fn program(token: &mut Option<Box<Token>>, input: &str) -> Vec<Box<ASTNode>> {
    let mut statements: Vec<Box<ASTNode>> = vec![];

    loop {
        if token.is_none() {
            break;
        }

        statements.push(stmt(token, input));
    }

    statements
}

fn stmt(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::LBrace) {
        let mut stmts = vec![];
        while !Token::consume(token, TokenKind::RBrace) {
            stmts.push(stmt(token, input));
        }

        if stmts.is_empty() {
            return ASTNode::leaf(ASTNodeKind::Block);
        }

        let mut block = stmts.pop().unwrap();
        while let Some(stmt) = stmts.pop() {
            block = ASTNode::new_boxed(ASTNodeKind::Block, Some(stmt), Some(block));
        }

        return block;
    }

    if Token::consume(token, TokenKind::If) {
        Token::expect(token, TokenKind::LParen, input);
        let cond_node = expr(token, input);
        Token::expect(token, TokenKind::RParen, input);
        let then_node = stmt(token, input);
        let else_node = if Token::consume(token, TokenKind::Else) {
            Some(stmt(token, input))
        } else {
            None
        };
        // If { lhs: 条件式, rhs: IfBody { lhs: then節, rhs: else節 } }
        let if_body = ASTNode::new_boxed(ASTNodeKind::IfBody, Some(then_node), else_node);
        return ASTNode::new_boxed(ASTNodeKind::If, Some(cond_node), Some(if_body));
    }

    if Token::consume(token, TokenKind::While) {
        Token::expect(token, TokenKind::LParen, input);
        let cond_node = expr(token, input);
        Token::expect(token, TokenKind::RParen, input);
        let body_node = stmt(token, input);
        return ASTNode::new_boxed(ASTNodeKind::While, Some(cond_node), Some(body_node));
    }

    if Token::consume(token, TokenKind::For) {
        Token::expect(token, TokenKind::LParen, input);

        let init_node = if !Token::consume(token, TokenKind::Semicolon) {
            let init = expr(token, input);
            Token::expect(token, TokenKind::Semicolon, input);
            Some(init)
        } else {
            None
        };

        let cond_node = if !Token::consume(token, TokenKind::Semicolon) {
            let cond = expr(token, input);
            Token::expect(token, TokenKind::Semicolon, input);
            Some(cond)
        } else {
            None
        };

        let update_node = if !Token::consume(token, TokenKind::RParen) {
            let update = expr(token, input);
            Token::expect(token, TokenKind::RParen, input);
            Some(update)
        } else {
            None
        };

        return ASTNode::new_boxed(
            ASTNodeKind::For,
            Some(ASTNode::new_boxed(
                ASTNodeKind::ForInit,
                init_node,
                cond_node,
            )),
            Some(ASTNode::new_boxed(
                ASTNodeKind::ForUpdate,
                update_node,
                Some(stmt(token, input)),
            )),
        );
    }

    if Token::consume(token, TokenKind::Return) {
        let expr_node = expr(token, input);
        Token::expect(token, TokenKind::Semicolon, input);
        return ASTNode::unary(ASTNodeKind::Return, expr_node);
    }

    let node = expr(token, input);
    Token::expect(token, TokenKind::Semicolon, input);
    node
}

pub fn expr(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    assign(token, input)
}

fn assign(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    let mut node = equality(token, input);

    if Token::consume(token, TokenKind::Assign) {
        let rhs = assign(token, input);
        node = ASTNode::binary(ASTNodeKind::Assign, node, rhs);
    }

    node
}

fn equality(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    let mut node = relational(token, input);

    loop {
        if Token::consume(token, TokenKind::Equal) {
            let rhs = relational(token, input);
            node = ASTNode::binary(ASTNodeKind::Equal, node, rhs);
        } else if Token::consume(token, TokenKind::NotEqual) {
            let rhs = relational(token, input);
            node = ASTNode::binary(ASTNodeKind::NotEqual, node, rhs);
        } else {
            break;
        }
    }

    node
}

fn relational(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    let mut node = add(token, input);

    loop {
        if Token::consume(token, TokenKind::Less) {
            let rhs = add(token, input);
            node = ASTNode::binary(ASTNodeKind::Less, node, rhs);
        } else if Token::consume(token, TokenKind::LessEqual) {
            let rhs = add(token, input);
            node = ASTNode::binary(ASTNodeKind::LessEqual, node, rhs);
        } else if Token::consume(token, TokenKind::Greater) {
            let rhs = add(token, input);
            node = ASTNode::binary(ASTNodeKind::Greater, node, rhs);
        } else if Token::consume(token, TokenKind::GreaterEqual) {
            let rhs = add(token, input);
            node = ASTNode::binary(ASTNodeKind::GreaterEqual, node, rhs);
        } else {
            break;
        }
    }

    node
}

fn add(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    let mut node = mul(token, input);

    loop {
        if Token::consume(token, TokenKind::Plus) {
            let rhs = mul(token, input);
            node = ASTNode::binary(ASTNodeKind::Add, node, rhs);
        } else if Token::consume(token, TokenKind::Minus) {
            let rhs = mul(token, input);
            node = ASTNode::binary(ASTNodeKind::Sub, node, rhs);
        } else {
            break;
        }
    }

    node
}

//  mul = unary ("*" unary | "/" unary)*
fn mul(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    let mut node = unary(token, input);

    loop {
        if Token::consume(token, TokenKind::Star) {
            let rhs = unary(token, input);
            node = ASTNode::binary(ASTNodeKind::Mul, node, rhs);
        } else if Token::consume(token, TokenKind::Slash) {
            let rhs = unary(token, input);
            node = ASTNode::binary(ASTNodeKind::Div, node, rhs);
        } else {
            break;
        }
    }

    node
}

// unary   = ("+" | "-")? primary
fn unary(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::Plus) {
        return primary(token, input);
    } else if Token::consume(token, TokenKind::Minus) {
        let node = primary(token, input);
        return ASTNode::binary(ASTNodeKind::Sub, ASTNode::leaf(ASTNodeKind::Num(0)), node);
    }
    primary(token, input)
}

// primary = num | identifier |  "(" expr ")"
fn primary(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::LParen) {
        let node = expr(token, input);
        Token::expect(token, TokenKind::RParen, input);
        return node;
    }

    if let Some(t) = token.take() {
        match t.kind {
            TokenKind::Number(num) => {
                *token = t.next;
                return ASTNode::leaf(ASTNodeKind::Num(num));
            }
            TokenKind::Identifier(var) => {
                *token = t.next;
                return ASTNode::leaf(ASTNodeKind::LocalVariable(var.get_offset()));
            }
            _ => unreachable!("{:?}", t),
        }
    }

    unreachable!("unexpected token: {:?}", token)
}

#[cfg(test)]
mod tests {
    use super::super::token::{LocalVariable, TokenKind};
    use super::*;

    struct TestTokenStream<'a> {
        source: &'a str,
        tokens: Vec<(TokenKind<'a>, usize, usize)>, // 種類、開始位置、終了位置
    }

    impl<'a> TestTokenStream<'a> {
        fn new(source: &'a str) -> Self {
            Self {
                source,
                tokens: Vec::new(),
            }
        }

        fn add(&mut self, kind: TokenKind<'a>, start: usize, end: usize) -> &mut Self {
            self.tokens.push((kind, start, end));
            self
        }

        fn build(&self) -> Option<Box<Token<'a>>> {
            let mut result: Option<Box<Token<'a>>> = None;
            let mut current = &mut result;

            for (kind, start, end) in &self.tokens {
                let fragment = &self.source[*start..*end];
                let new_token = Some(Box::new(Token::init(kind.clone(), fragment)));
                *current = new_token;

                if let Some(ref mut token) = *current {
                    current = &mut token.next;
                }
            }

            result
        }
    }

    struct TestCase<'a> {
        name: &'a str,
        token: Option<Box<Token<'a>>>,
        raw_input: &'a str,
        expected: Box<ASTNode>,
    }

    #[test]
    fn test_expr() {
        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None)),
            },
            TestCase {
                name: "1 + 2 * 3 が正しくparseされること",
                token: TestTokenStream::new("1+2*3")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::Plus, 1, 2)
                    .add(TokenKind::Number(2), 2, 3)
                    .add(TokenKind::Star, 3, 4)
                    .add(TokenKind::Number(3), 4, 5)
                    .build(),
                raw_input: "1 + 2 * 3",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Mul,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                    ))),
                )),
            },
            TestCase {
                name: "1*2+(3+4) が正しくparseされること",
                token: TestTokenStream::new("1*2+(3+4)")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::Star, 1, 2)
                    .add(TokenKind::Number(2), 2, 3)
                    .add(TokenKind::Plus, 3, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::Number(3), 5, 6)
                    .add(TokenKind::Plus, 6, 7)
                    .add(TokenKind::Number(4), 7, 8)
                    .add(TokenKind::RParen, 8, 9)
                    .build(),
                raw_input: "1*2+(3+4)",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Mul,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Add,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(4), None, None))),
                    ))),
                )),
            },
            TestCase {
                name: "-1 * +2 が正しくparseされること",
                token: TestTokenStream::new("-1*+2")
                    .add(TokenKind::Minus, 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .add(TokenKind::Star, 2, 3)
                    .add(TokenKind::Plus, 3, 4)
                    .add(TokenKind::Number(2), 4, 5)
                    .build(),
                raw_input: "-1 * +2",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Sub,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                )),
            },
            TestCase {
                name: "1 <= 2 が正しくparseされること",
                token: TestTokenStream::new("1<=2")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::LessEqual, 1, 3)
                    .add(TokenKind::Number(2), 3, 4)
                    .build(),
                raw_input: "1 <= 2",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::LessEqual,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                )),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let result = expr(&mut token, case.raw_input);
            assert_eq!(result, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn test_unary() {
        let test_cases = vec![
            TestCase {
                name: "-1 が正しくparseされること",
                token: TestTokenStream::new("-1")
                    .add(TokenKind::Minus, 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "-1",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Sub,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                )),
            },
            TestCase {
                name: "+1 が正しくparseされること",
                token: TestTokenStream::new("+1")
                    .add(TokenKind::Plus, 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "+1",
                expected: Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None)),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let result = unary(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_primary() {
        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None)),
            },
            TestCase {
                name: "識別子が正しくparseされること",
                token: Some(Box::new(Token::init(
                    TokenKind::Identifier(LocalVariable::new("x", 0)),
                    "x",
                ))),
                raw_input: "x",
                expected: Box::new(ASTNode::new(ASTNodeKind::LocalVariable(0), None, None)),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let result = primary(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_mul() {
        let test_cases = vec![TestCase {
            name: "1 * 2 が正しくparseされること",
            token: TestTokenStream::new("1*2")
                .add(TokenKind::Number(1), 0, 1)
                .add(TokenKind::Star, 1, 2)
                .add(TokenKind::Number(2), 2, 3)
                .build(),
            raw_input: "1 * 2",
            expected: Box::new(ASTNode::new(
                ASTNodeKind::Mul,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            )),
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = mul(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }

    #[test]
    fn test_relational() {
        let test_cases = vec![TestCase {
            name: "1 < 2 が正しくparseされること",
            token: TestTokenStream::new("1<2")
                .add(TokenKind::Number(1), 0, 1)
                .add(TokenKind::Less, 1, 2)
                .add(TokenKind::Number(2), 2, 3)
                .build(),
            raw_input: "1 < 2",
            expected: Box::new(ASTNode::new(
                ASTNodeKind::Less,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            )),
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = relational(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }
    #[test]
    fn test_equality() {
        let test_cases = vec![TestCase {
            name: "1 == 2 が正しくparseされること",
            token: TestTokenStream::new("1==2")
                .add(TokenKind::Number(1), 0, 1)
                .add(TokenKind::Equal, 1, 3)
                .add(TokenKind::Number(2), 3, 4)
                .build(),
            raw_input: "1 == 2",
            expected: Box::new(ASTNode::new(
                ASTNodeKind::Equal,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            )),
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = equality(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }
    #[test]
    fn test_assign() {
        let test_cases = vec![TestCase {
            name: "x = 1 が正しくparseされること",
            token: TestTokenStream::new("x=1")
                .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 0, 1)
                .add(TokenKind::Assign, 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .build(),
            raw_input: "x = 1",
            expected: Box::new(ASTNode::new(
                ASTNodeKind::Assign,
                Some(Box::new(ASTNode::new(
                    ASTNodeKind::LocalVariable(0),
                    None,
                    None,
                ))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            )),
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = assign(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }
    #[test]
    fn test_stmt() {
        let test_cases = vec![
            TestCase {
                name: "x = 1; が正しくparseされること",
                token: TestTokenStream::new("x=1;")
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 0, 1)
                    .add(TokenKind::Assign, 1, 2)
                    .add(TokenKind::Number(1), 2, 3)
                    .add(TokenKind::Semicolon, 3, 4)
                    .build(),
                raw_input: "x = 1;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(0),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                )),
            },
            TestCase {
                name: "return文が正しくparseされること",
                token: TestTokenStream::new("return 42;")
                    .add(TokenKind::Return, 0, 6)
                    .add(TokenKind::Number(42), 7, 9)
                    .add(TokenKind::Semicolon, 9, 10)
                    .build(),
                raw_input: "return 42;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Return,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                    None,
                )),
            },
            TestCase {
                name: "return式が正しくparseされること",
                token: TestTokenStream::new("return x+1;")
                    .add(TokenKind::Return, 0, 6)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 7, 8)
                    .add(TokenKind::Plus, 8, 9)
                    .add(TokenKind::Number(1), 9, 10)
                    .add(TokenKind::Semicolon, 10, 11)
                    .build(),
                raw_input: "return x+1;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Return,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Add,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::LocalVariable(0),
                            None,
                            None,
                        ))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    ))),
                    None,
                )),
            },
            TestCase {
                name: "while文が正しくparseされること",
                token: TestTokenStream::new("while(x<10)x=x+1;")
                    .add(TokenKind::While, 0, 5)
                    .add(TokenKind::LParen, 5, 6)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 6, 7)
                    .add(TokenKind::Less, 7, 8)
                    .add(TokenKind::Number(10), 8, 10)
                    .add(TokenKind::RParen, 10, 11)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 11, 12)
                    .add(TokenKind::Assign, 12, 13)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 13, 14)
                    .add(TokenKind::Plus, 14, 15)
                    .add(TokenKind::Number(1), 15, 16)
                    .add(TokenKind::Semicolon, 16, 17)
                    .build(),
                raw_input: "while(x<10)x=x+1;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::While,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Less,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::LocalVariable(0),
                            None,
                            None,
                        ))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(10), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Assign,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::LocalVariable(0),
                            None,
                            None,
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Add,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(0),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                        ))),
                    ))),
                )),
            },
            TestCase {
                name: "if-else文が正しくparseされること",
                token: TestTokenStream::new("if(x>0)y=1;else y=2;")
                    .add(TokenKind::If, 0, 2)
                    .add(TokenKind::LParen, 2, 3)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 3, 4)
                    .add(TokenKind::Greater, 4, 5)
                    .add(TokenKind::Number(0), 5, 6)
                    .add(TokenKind::RParen, 6, 7)
                    .add(TokenKind::Identifier(LocalVariable::new("y", 8)), 7, 8)
                    .add(TokenKind::Assign, 8, 9)
                    .add(TokenKind::Number(1), 9, 10)
                    .add(TokenKind::Semicolon, 10, 11)
                    .add(TokenKind::Else, 11, 15)
                    .add(TokenKind::Identifier(LocalVariable::new("y", 8)), 16, 17)
                    .add(TokenKind::Assign, 17, 18)
                    .add(TokenKind::Number(2), 18, 19)
                    .add(TokenKind::Semicolon, 19, 20)
                    .build(),
                raw_input: "if(x>0)y=1;else y=2;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::If,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Greater,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::LocalVariable(0),
                            None,
                            None,
                        ))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::IfBody,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                        ))),
                    ))),
                )),
            },
            TestCase {
                name: "for文が正しくparseされること",
                token: TestTokenStream::new("for(i=0;i<10;i=i+1)x=x+1;")
                    .add(TokenKind::For, 0, 3)
                    .add(TokenKind::LParen, 3, 4)
                    .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 4, 5)
                    .add(TokenKind::Assign, 5, 6)
                    .add(TokenKind::Number(0), 6, 7)
                    .add(TokenKind::Semicolon, 7, 8)
                    .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 8, 9)
                    .add(TokenKind::Less, 9, 10)
                    .add(TokenKind::Number(10), 10, 12)
                    .add(TokenKind::Semicolon, 12, 13)
                    .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 13, 14)
                    .add(TokenKind::Assign, 14, 15)
                    .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 15, 16)
                    .add(TokenKind::Plus, 16, 17)
                    .add(TokenKind::Number(1), 17, 18)
                    .add(TokenKind::RParen, 18, 19)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 8)), 19, 20)
                    .add(TokenKind::Assign, 20, 21)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 8)), 21, 22)
                    .add(TokenKind::Plus, 22, 23)
                    .add(TokenKind::Number(1), 23, 24)
                    .add(TokenKind::Semicolon, 24, 25)
                    .build(),
                raw_input: "for(i=0;i<10;i=i+1)x=x+1;",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::For,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::ForInit,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(0),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Less,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(0),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(10), None, None))),
                        ))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::ForUpdate,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(0),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::Add,
                                Some(Box::new(ASTNode::new(
                                    ASTNodeKind::LocalVariable(0),
                                    None,
                                    None,
                                ))),
                                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                            ))),
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::Add,
                                Some(Box::new(ASTNode::new(
                                    ASTNodeKind::LocalVariable(8),
                                    None,
                                    None,
                                ))),
                                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                            ))),
                        ))),
                    ))),
                )),
            },
            TestCase {
                name: "空のブロック文が正しくparseされること",
                token: TestTokenStream::new("{}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::RBrace, 1, 2)
                    .build(),
                raw_input: "{}",
                expected: Box::new(ASTNode::new(ASTNodeKind::Block, None, None)),
            },
            TestCase {
                name: "単一文のブロックが正しくparseされること",
                token: TestTokenStream::new("{x=1;}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 1, 2)
                    .add(TokenKind::Assign, 2, 3)
                    .add(TokenKind::Number(1), 3, 4)
                    .add(TokenKind::Semicolon, 4, 5)
                    .add(TokenKind::RBrace, 5, 6)
                    .build(),
                raw_input: "{x=1;}",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(0),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                )),
            },
            TestCase {
                name: "複数文のブロックが正しくparseされること",
                token: TestTokenStream::new("{x=1;y=2;z=3;}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 1, 2)
                    .add(TokenKind::Assign, 2, 3)
                    .add(TokenKind::Number(1), 3, 4)
                    .add(TokenKind::Semicolon, 4, 5)
                    .add(TokenKind::Identifier(LocalVariable::new("y", 8)), 5, 6)
                    .add(TokenKind::Assign, 6, 7)
                    .add(TokenKind::Number(2), 7, 8)
                    .add(TokenKind::Semicolon, 8, 9)
                    .add(TokenKind::Identifier(LocalVariable::new("z", 16)), 9, 10)
                    .add(TokenKind::Assign, 10, 11)
                    .add(TokenKind::Number(3), 11, 12)
                    .add(TokenKind::Semicolon, 12, 13)
                    .add(TokenKind::RBrace, 13, 14)
                    .build(),
                raw_input: "{x=1;y=2;z=3;}",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::Block,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Assign,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::LocalVariable(0),
                            None,
                            None,
                        ))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Block,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(16),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                        ))),
                    ))),
                )),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let result = stmt(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }
    #[test]
    fn test_program() {
        struct ProgramTestCase<'a> {
            name: &'a str,
            token: Option<Box<Token<'a>>>,
            raw_input: &'a str,
            expected: Vec<Box<ASTNode>>,
        }

        let test_cases = vec![ProgramTestCase {
            name: "x = 1; y = 2; が正しくparseされること",
            token: TestTokenStream::new("x=1;y=2;")
                .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 0, 1)
                .add(TokenKind::Assign, 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .add(TokenKind::Semicolon, 3, 4)
                .add(TokenKind::Identifier(LocalVariable::new("y", 0)), 4, 5)
                .add(TokenKind::Assign, 5, 6)
                .add(TokenKind::Number(2), 6, 7)
                .add(TokenKind::Semicolon, 7, 8)
                .build(),
            raw_input: "x = 1; y = 2;",
            expected: vec![
                Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(0),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                )),
                Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(0),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                )),
            ],
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = program(&mut token, case.raw_input);
            assert_eq!(result, case.expected, "{}", case.name);
        }
    }
}
