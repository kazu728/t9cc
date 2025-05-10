use super::token::Token;

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
// expr       = equality
// equality   = relational ("==" relational | "!=" relational)*
// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
// add        = mul ("+" mul | "-" mul)*
// mul        = unary ("*" unary | "/" unary)*
// unary      = ("+" | "-")? primary
// primary    = num | "(" expr ")"

// expr = mul ("+" mul | "-" mul)*

// TODO: Some(Box::new(ASTNode::new これの共通化はして良い
pub fn expr(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    equality(token, input)
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

// primary = num | "(" expr ")"
fn primary(token: &mut Option<Box<Token>>, input: &str) -> MaybeASTNode {
    if Token::consume(token, "(") {
        let node = expr(token, input);
        Token::expect(token, ")", input);
        return node;
    }

    let num = Token::expect_number(token, input);
    ASTNode::new_meybe_node(ASTNodeKind::Num(num), None, None)
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

    #[test]
    fn test_expr() {
        struct TestCase<'a> {
            token: Option<Box<Token<'a>>>,
            raw_input: &'a str,
            expected: MaybeASTNode,
        }

        let test_cases = vec![
            TestCase {
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            },
            // 1 + 2 * 3 が正しくparseされること
            TestCase {
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
            // 1*2+(3+4) が正しくparseされること
            TestCase {
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
            // -1 * +2 が正しくparseされること
            TestCase {
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
            // 1 <= 2 が正しくparseされること
            TestCase {
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
            assert_eq!(result, case.expected);
        }
    }
}
