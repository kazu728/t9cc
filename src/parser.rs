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
    pub fn new_meybe_node(kind: ASTNodeKind, lhs: MaybeASTNode, rhs: MaybeASTNode) -> MaybeASTNode {
        Some(Box::new(ASTNode { kind, lhs, rhs }))
    }
}

// program    = stmt*
// stmt       = expr ";"
// expr       = assign
// assign     = equality ("=" assign)?
// equality   = relational ("==" relational | "!=" relational)*
// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
// add        = mul ("+" mul | "-" mul)*
// mul        = unary ("*" unary | "/" unary)*
// unary      = ("+" | "-")? primary
// primary    = num | ident | "(" expr ")"

pub fn program(token: &mut Option<Box<Token>>, input: &str) -> Vec<MaybeASTNode> {
    let mut statements: Vec<MaybeASTNode> = vec![];

    loop {
        if token.is_none() {
            break;
        }

        statements.push(stmt(token, input));
    }

    statements
}

fn stmt(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let node = expr(token, input);
    Token::consume(token, ";");
    node
}

pub fn expr(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    assign(token, input)
}

fn assign(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let mut node = equality(token, input);

    if Token::consume(token, "=") {
        let rhs = assign(token, input);
        node = ASTNode::new_meybe_node(ASTNodeKind::Assign, node, rhs);
    }

    node
}

fn equality(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let mut node = relational(token, input);

    loop {
        if Token::consume(token, "==") {
            let rhs = relational(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Equal, node, rhs);
        } else if Token::consume(token, "!=") {
            let rhs = relational(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::NotEqual, node, rhs);
        } else {
            break;
        }
    }

    node
}

fn relational(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let mut node = add(token, input);

    loop {
        if Token::consume(token, "<") {
            let rhs = add(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Less, node, rhs);
        } else if Token::consume(token, "<=") {
            let rhs = add(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::LessEqual, node, rhs);
        } else if Token::consume(token, ">") {
            let rhs = add(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Greater, node, rhs);
        } else if Token::consume(token, ">=") {
            let rhs = add(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::GreaterEqual, node, rhs);
        } else {
            break;
        }
    }

    node
}

fn add(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let mut node = mul(token, input);

    loop {
        if Token::consume(token, "+") {
            let rhs = mul(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Add, node, rhs);
        } else if Token::consume(token, "-") {
            let rhs = mul(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Sub, node, rhs);
        } else {
            break;
        }
    }

    node
}

//  mul = unary ("*" unary | "/" unary)*
fn mul(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    let mut node = unary(token, input);

    loop {
        if Token::consume(token, "*") {
            let rhs = unary(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Mul, node, rhs);
        } else if Token::consume(token, "/") {
            let rhs = unary(token, input);
            node = ASTNode::new_meybe_node(ASTNodeKind::Div, node, rhs);
        } else {
            break;
        }
    }

    node
}

// unary   = ("+" | "-")? primary
fn unary(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    if Token::consume(token, "+") {
        return primary(token, input);
    } else if Token::consume(token, "-") {
        let node = primary(token, input);
        return Some(Box::new(ASTNode::new(
            ASTNodeKind::Sub,
            Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
            node,
        )));
    }
    primary(token, input)
}

// primary = num | identifier |  "(" expr ")"
fn primary(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    if Token::consume(token, "(") {
        let node = expr(token, input);
        Token::expect(token, ")", input);
        return node;
    }

    if let Some(t) = token {
        match t.kind {
            TokenKind::Number(num) => {
                *token = t.next.clone();
                return ASTNode::new_meybe_node(ASTNodeKind::Num(num), None, None);
            }
            TokenKind::Identifier(id) => {
                *token = t.next.clone();
                return ASTNode::new_meybe_node(
                    ASTNodeKind::LocalVariable(calculate_offset(id)),
                    None,
                    None,
                );
            }
            _ => unreachable!("{:?}", t),
        }
    }

    unreachable!("unexpected token: {:?}", token)
}

fn calculate_offset(input: &str) -> u32 {
    let mut offset = 0;
    for ch in input.chars() {
        if ch.is_alphabetic() {
            offset += (ch as u8 - b'a' + 1) as u32 * 8;
        }
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::super::token::TokenKind;
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
                let new_token = Some(Box::new(Token::init(*kind, fragment)));
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
        expected: MaybeASTNode,
    }

    #[test]
    fn test_expr() {
        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            },
            TestCase {
                name: "1 + 2 * 3 が正しくparseされること",
                token: TestTokenStream::new("1+2*3")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::Reserved("+"), 1, 2)
                    .add(TokenKind::Number(2), 2, 3)
                    .add(TokenKind::Reserved("*"), 3, 4)
                    .add(TokenKind::Number(3), 4, 5)
                    .build(),
                raw_input: "1 + 2 * 3",
                expected: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Mul,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                    ))),
                ))),
            },
            TestCase {
                name: "1*2+(3+4) が正しくparseされること",
                token: TestTokenStream::new("1*2+(3+4)")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::Reserved("*"), 1, 2)
                    .add(TokenKind::Number(2), 2, 3)
                    .add(TokenKind::Reserved("+"), 3, 4)
                    .add(TokenKind::Reserved("("), 4, 5)
                    .add(TokenKind::Number(3), 5, 6)
                    .add(TokenKind::Reserved("+"), 6, 7)
                    .add(TokenKind::Number(4), 7, 8)
                    .add(TokenKind::Reserved(")"), 8, 9)
                    .build(),
                raw_input: "1*2+(3+4)",
                expected: Some(Box::new(ASTNode::new(
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
                ))),
            },
            TestCase {
                name: "-1 * +2 が正しくparseされること",
                token: TestTokenStream::new("-1*+2")
                    .add(TokenKind::Reserved("-"), 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .add(TokenKind::Reserved("*"), 2, 3)
                    .add(TokenKind::Reserved("+"), 3, 4)
                    .add(TokenKind::Number(2), 4, 5)
                    .build(),
                raw_input: "-1 * +2",
                expected: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Sub,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
            },
            TestCase {
                name: "1 <= 2 が正しくparseされること",
                token: TestTokenStream::new("1<=2")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::Reserved("<="), 1, 3)
                    .add(TokenKind::Number(2), 3, 4)
                    .build(),
                raw_input: "1 <= 2",
                expected: Some(Box::new(ASTNode::new(
                    ASTNodeKind::LessEqual,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
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
                    .add(TokenKind::Reserved("-"), 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "-1",
                expected: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Sub,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                ))),
            },
            TestCase {
                name: "+1 が正しくparseされること",
                token: TestTokenStream::new("+1")
                    .add(TokenKind::Reserved("+"), 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "+1",
                expected: Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
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
                expected: Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            },
            TestCase {
                name: "識別子が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Identifier("x"), "x"))),
                raw_input: "x",
                expected: Some(Box::new(ASTNode::new(
                    ASTNodeKind::LocalVariable(calculate_offset("x")),
                    None,
                    None,
                ))),
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
                .add(TokenKind::Reserved("*"), 1, 2)
                .add(TokenKind::Number(2), 2, 3)
                .build(),
            raw_input: "1 * 2",
            expected: Some(Box::new(ASTNode::new(
                ASTNodeKind::Mul,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            ))),
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
                .add(TokenKind::Reserved("<"), 1, 2)
                .add(TokenKind::Number(2), 2, 3)
                .build(),
            raw_input: "1 < 2",
            expected: Some(Box::new(ASTNode::new(
                ASTNodeKind::Less,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            ))),
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
                .add(TokenKind::Reserved("=="), 1, 3)
                .add(TokenKind::Number(2), 3, 4)
                .build(),
            raw_input: "1 == 2",
            expected: Some(Box::new(ASTNode::new(
                ASTNodeKind::Equal,
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
            ))),
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
                .add(TokenKind::Identifier("x"), 0, 1)
                .add(TokenKind::Reserved("="), 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .build(),
            raw_input: "x = 1",
            expected: Some(Box::new(ASTNode::new(
                ASTNodeKind::Assign,
                Some(Box::new(ASTNode::new(
                    ASTNodeKind::LocalVariable(calculate_offset("x")),
                    None,
                    None,
                ))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            ))),
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = assign(&mut token, case.raw_input);
            assert_eq!(result, case.expected);
        }
    }
    #[test]
    fn test_stmt() {
        let test_cases = vec![TestCase {
            name: "x = 1; が正しくparseされること",
            token: TestTokenStream::new("x=1;")
                .add(TokenKind::Identifier("x"), 0, 1)
                .add(TokenKind::Reserved("="), 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .add(TokenKind::Reserved(";"), 3, 4)
                .build(),
            raw_input: "x = 1;",
            expected: Some(Box::new(ASTNode::new(
                ASTNodeKind::Assign,
                Some(Box::new(ASTNode::new(
                    ASTNodeKind::LocalVariable(calculate_offset("x")),
                    None,
                    None,
                ))),
                Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            ))),
        }];

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
            expected: Vec<MaybeASTNode>,
        }

        let test_cases = vec![ProgramTestCase {
            name: "x = 1; y = 2; が正しくparseされること",
            token: TestTokenStream::new("x=1;y=2;")
                .add(TokenKind::Identifier("x"), 0, 1)
                .add(TokenKind::Reserved("="), 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .add(TokenKind::Reserved(";"), 3, 4)
                .add(TokenKind::Identifier("y"), 4, 5)
                .add(TokenKind::Reserved("="), 5, 6)
                .add(TokenKind::Number(2), 6, 7)
                .add(TokenKind::Reserved(";"), 7, 8)
                .build(),
            raw_input: "x = 1; y = 2;",
            expected: vec![
                Some(Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(calculate_offset("x")),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                ))),
                Some(Box::new(ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(calculate_offset("y")),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
            ],
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = program(&mut token, case.raw_input);
            assert_eq!(result, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn test_calculate_offset() {
        let test_cases = vec![
            ("a", 8),
            ("b", 16),
            ("c", 24),
            ("x", 192),
            ("y", 200),
            ("z", 208),
        ];

        for (input, expected) in test_cases {
            let result = calculate_offset(input);
            assert_eq!(result, expected);
        }
    }
}
