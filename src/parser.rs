use super::ast::{ASTNode, Function, Program};
use super::token::{Token, TokenKind};
use super::types::Type;
use std::collections::BTreeMap;
use std::fmt;
use std::iter;

#[derive(Debug, Clone)]
pub enum ParseError {
    MissingTypeDeclaration,
    InvalidArraySize,
    ArraySizeNotSpecified,
    UndefinedVariable(String),
    UnexpectedToken(String),
    UnexpectedEndOfInput,
    ArrayOrPointerRequired,
    BothPointerUnsupported,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingTypeDeclaration => write!(f, "型宣言が必要です"),
            ParseError::InvalidArraySize => write!(f, "配列のサイズは数値である必要があります"),
            ParseError::ArraySizeNotSpecified => write!(f, "配列のサイズが指定されていません"),
            ParseError::UndefinedVariable(name) => write!(f, "未定義の変数: {}", name),
            ParseError::UnexpectedToken(token) => write!(f, "予期しないトークン: {}", token),
            ParseError::UnexpectedEndOfInput => write!(f, "予期しない入力の終端です"),
            ParseError::ArrayOrPointerRequired => {
                write!(f, "配列またはポインタである必要があります")
            }
            ParseError::BothPointerUnsupported => write!(f, "両方がポインタの場合は未サポート"),
        }
    }
}

type MaybeToken<'a> = Option<Box<Token<'a>>>;

pub trait TokenExt<'a> {
    fn consume(&mut self, kind: TokenKind<'a>) -> bool;
    fn expect(&mut self, kind: TokenKind<'a>, input: &str) -> Result<(), ParseError>;
    fn expect_identifier(&mut self, input: &str) -> Result<String, ParseError>;
}

impl<'a> TokenExt<'a> for MaybeToken<'a> {
    /**
     * 現在のトークンが指定されたトークン種別かどうか確認する
     * トークンが指定された種別であればトークンを消費して次に進む
     */
    fn consume(&mut self, kind: TokenKind<'a>) -> bool {
        if let Some(token) = self {
            if token.kind == kind {
                *self = token.next.take();
                return true;
            }
        }
        false
    }

    /**
     * 次のトークンが指定された種別であることを確認する
     * トークンが指定された種別であればトークンを消費して次に進む
     * トークンが指定された種別でない場合はエラーを返す
     */
    fn expect(&mut self, kind: TokenKind<'a>, _input: &str) -> Result<(), ParseError> {
        if let Some(tok) = self {
            if tok.kind == kind {
                *self = tok.next.take();
                return Ok(());
            }
            return Err(ParseError::UnexpectedToken(format!(
                "{:?} が必要ですが、{:?} が見つかりました",
                kind, tok.kind
            )));
        } else {
            return Err(ParseError::UnexpectedEndOfInput);
        }
    }

    fn expect_identifier(&mut self, _input: &str) -> Result<String, ParseError> {
        if let Some(tok) = self {
            if let TokenKind::Identifier(name) = &tok.kind {
                let result = name.to_string();
                *self = tok.next.take();
                return Ok(result);
            }
            return Err(ParseError::UnexpectedToken("識別子が必要です".to_string()));
        } else {
            return Err(ParseError::UnexpectedEndOfInput);
        }
    }
}

struct FunctionScope {
    variables: BTreeMap<String, (u32, Type)>,
    next_offset: u32,
}

impl FunctionScope {
    fn new() -> Self {
        FunctionScope {
            variables: BTreeMap::new(),
            next_offset: 8,
        }
    }

    fn add_variable(&mut self, name: String, var_type: Type) -> u32 {
        if let Some((offset, _)) = self.variables.get(&name) {
            *offset
        } else {
            let offset = self.next_offset;
            let type_size = var_type.sizeof() as u32;
            let aligned_size = (type_size + 7) & !7;
            self.variables.insert(name, (offset, var_type));
            self.next_offset += aligned_size;
            offset
        }
    }

    fn get_variable(&self, name: &str) -> Option<&(u32, Type)> {
        self.variables.get(name)
    }
}

fn consume_type(token: &mut Option<Box<Token>>) -> Result<Type, ParseError> {
    match token.as_ref().map(|t| &t.kind) {
        Some(TokenKind::Int) => {
            token.consume(TokenKind::Int);
            Ok(Type::new_int())
        }
        Some(TokenKind::Char) => {
            token.consume(TokenKind::Char);
            Ok(Type::new_char())
        }
        _ => Err(ParseError::MissingTypeDeclaration),
    }
}

fn parse_array_size(token: &mut Option<Box<Token>>, input: &str) -> Result<usize, ParseError> {
    if let Some(size_token) = token.take() {
        if let TokenKind::Number(array_size) = size_token.kind {
            *token = size_token.next;
            token.expect(TokenKind::RBracket, input)?;
            Ok(array_size as usize)
        } else {
            Err(ParseError::InvalidArraySize)
        }
    } else {
        Err(ParseError::ArraySizeNotSpecified)
    }
}

// program    = (function_def | global_var_def)*
// function_def = ident "(" params? ")" "{" stmt* "}"
// params     = ident ("," ident)*

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
// primary    = num | ident ("(" (expr ("," expr)*)? ")")? | "(" expr ")"

pub fn program(token: &mut Option<Box<Token>>, input: &str) -> Result<Program, ParseError> {
    let mut program = Program::new();

    while token.is_some() {
        if is_function_definition(token) {
            let new_function = function_def(
                token,
                input,
                &mut program.global_vars,
                &mut program.string_literals,
            )?;
            program.functions.push(new_function);
        } else {
            global_var_def(token, input, &mut program.global_vars)?;
        }
    }

    Ok(program)
}

// トークンを先読みして関数定義か変数定義かを判別する
fn is_function_definition(token: &Option<Box<Token>>) -> bool {
    if let Some(t) = token {
        if matches!(t.kind, TokenKind::Int | TokenKind::Char) {
            let mut current = &t.next;

            while let Some(t) = current {
                if matches!(t.kind, TokenKind::Star) {
                    current = &t.next;
                } else {
                    break;
                }
            }

            if let Some(t) = current {
                if matches!(t.kind, TokenKind::Identifier(_)) {
                    if let Some(next) = &t.next {
                        return matches!(next.kind, TokenKind::LParen);
                    }
                }
            }
        }
    }
    false
}

fn function_def(
    token: &mut Option<Box<Token>>,
    input: &str,
    _global_vars: &mut BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<Function, ParseError> {
    let _return_type = consume_type(token)?;
    let function_name = token.expect_identifier(input)?;

    token.expect(TokenKind::LParen, input)?;

    let mut scope = FunctionScope::new();
    let params = parse_parameters(token, input, &mut scope)?;

    token.expect(TokenKind::LBrace, input)?;

    let mut body_stmts = Vec::new();
    while !token.consume(TokenKind::RBrace) {
        body_stmts.push(stmt(
            token,
            input,
            &mut scope,
            _global_vars,
            string_literals,
        )?);
    }

    let body = ASTNode::Block {
        statements: body_stmts,
    };

    Ok(Function::new(function_name, params, body))
}

fn global_var_def(
    token: &mut Option<Box<Token>>,
    input: &str,
    global_vars: &mut BTreeMap<String, Type>,
) -> Result<(), ParseError> {
    let base_type = consume_type(token)?;

    let ptr_count = iter::from_fn(|| token.consume(TokenKind::Star).then_some(())).count();

    let var_name = token.expect_identifier(input)?;

    let var_type = if token.consume(TokenKind::LBracket) {
        let array_size = if let Some(t) = token.take() {
            if let TokenKind::Number(size) = t.kind {
                *token = t.next;
                size as usize
            } else {
                return Err(ParseError::InvalidArraySize);
            }
        } else {
            return Err(ParseError::ArraySizeNotSpecified);
        };
        token.expect(TokenKind::RBracket, input)?;

        let ptr_type = (0..ptr_count).fold(base_type.clone(), |acc, _| Type::new_ptr(acc));
        Type::new_array(ptr_type, array_size)
    } else {
        (0..ptr_count).fold(base_type, |acc, _| Type::new_ptr(acc))
    };

    global_vars.insert(var_name, var_type);
    token.expect(TokenKind::Semicolon, input)?;
    Ok(())
}

fn stmt(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    match token.as_ref().map(|t| &t.kind) {
        Some(TokenKind::LBrace) => {
            token.consume(TokenKind::LBrace);
            let mut stmts = vec![];
            while !token.consume(TokenKind::RBrace) {
                stmts.push(stmt(token, input, scope, global_vars, string_literals)?);
            }
            Ok(ASTNode::Block { statements: stmts })
        }
        Some(TokenKind::Int) => {
            token.consume(TokenKind::Int);
            parse_variable_declaration(token, input, scope, Type::new_int())
        }
        Some(TokenKind::Char) => {
            token.consume(TokenKind::Char);
            parse_variable_declaration(token, input, scope, Type::new_char())
        }
        Some(TokenKind::If) => {
            token.consume(TokenKind::If);
            token.expect(TokenKind::LParen, input)?;

            let condition = expr(token, input, scope, global_vars, string_literals)?;
            token.expect(TokenKind::RParen, input)?;
            Ok(ASTNode::If {
                condition: Box::new(condition),
                then_stmt: Box::new(stmt(token, input, scope, global_vars, string_literals)?),
                else_stmt: if token.consume(TokenKind::Else) {
                    Some(Box::new(stmt(
                        token,
                        input,
                        scope,
                        global_vars,
                        string_literals,
                    )?))
                } else {
                    None
                },
            })
        }
        Some(TokenKind::While) => {
            token.consume(TokenKind::While);
            token.expect(TokenKind::LParen, input)?;
            let condition = expr(token, input, scope, global_vars, string_literals)?;
            token.expect(TokenKind::RParen, input)?;
            Ok(ASTNode::While {
                condition: Box::new(condition),
                body: Box::new(stmt(token, input, scope, global_vars, string_literals)?),
            })
        }
        Some(TokenKind::For) => {
            token.consume(TokenKind::For);
            token.expect(TokenKind::LParen, input)?;

            let init = match token.as_ref().map(|t| &t.kind) {
                Some(TokenKind::Semicolon) => {
                    token.consume(TokenKind::Semicolon);
                    None
                }
                _ => {
                    let init_expr = expr(token, input, scope, global_vars, string_literals)?;
                    token.expect(TokenKind::Semicolon, input)?;
                    Some(Box::new(init_expr))
                }
            };

            let condition = match token.as_ref().map(|t| &t.kind) {
                Some(TokenKind::Semicolon) => {
                    token.consume(TokenKind::Semicolon);
                    None
                }
                _ => {
                    let cond_expr = expr(token, input, scope, global_vars, string_literals)?;
                    token.expect(TokenKind::Semicolon, input)?;
                    Some(Box::new(cond_expr))
                }
            };

            let update = match token.as_ref().map(|t| &t.kind) {
                Some(TokenKind::RParen) => {
                    token.consume(TokenKind::RParen);
                    None
                }
                _ => {
                    let update_expr = expr(token, input, scope, global_vars, string_literals)?;
                    token.expect(TokenKind::RParen, input)?;
                    Some(Box::new(update_expr))
                }
            };

            let body = Box::new(stmt(token, input, scope, global_vars, string_literals)?);

            Ok(ASTNode::For {
                init,
                condition,
                update,
                body,
            })
        }
        Some(TokenKind::Return) => {
            token.consume(TokenKind::Return);
            let expr_node = expr(token, input, scope, global_vars, string_literals)?;
            token.expect(TokenKind::Semicolon, input)?;
            Ok(ASTNode::Return {
                expr: Some(Box::new(expr_node)),
            })
        }
        _ => {
            let node = expr(token, input, scope, global_vars, string_literals)?;
            token.expect(TokenKind::Semicolon, input)?;
            Ok(ASTNode::ExpressionStatement {
                expr: Box::new(node),
            })
        }
    }
}

fn parse_variable_declaration(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    base_type: Type,
) -> Result<ASTNode, ParseError> {
    let ptr_count = iter::from_fn(|| token.consume(TokenKind::Star).then_some(())).count();

    let var_name = token.expect_identifier(input)?;

    let var_type = if token.consume(TokenKind::LBracket) {
        let array_size = parse_array_size(token, input)?;
        let ptr_type = (0..ptr_count).fold(base_type, |acc, _| Type::new_ptr(acc));
        Type::new_array(ptr_type, array_size)
    } else {
        (0..ptr_count).fold(base_type, |acc, _| Type::new_ptr(acc))
    };

    scope.add_variable(var_name, var_type);
    token.expect(TokenKind::Semicolon, input)?;
    Ok(ASTNode::Block { statements: vec![] })
}

fn expr(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    assign(token, input, scope, global_vars, string_literals)
}

fn assign(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut node = equality(token, input, scope, global_vars, string_literals)?;

    if token.consume(TokenKind::Assign) {
        let rhs = assign(token, input, scope, global_vars, string_literals)?;

        let rhs_type = rhs.get_node_type().cloned().unwrap_or(Type::new_int());
        node = ASTNode::Assign {
            lhs: Box::new(node),
            rhs: Box::new(rhs),
            node_type: rhs_type,
        };
    }

    Ok(node)
}

fn equality(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut node = relational(token, input, scope, global_vars, string_literals)?;

    loop {
        match token.as_ref().map(|t| &t.kind) {
            Some(TokenKind::Equal) => {
                token.consume(TokenKind::Equal);
                let rhs = relational(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::Equal {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            Some(TokenKind::NotEqual) => {
                token.consume(TokenKind::NotEqual);
                let rhs = relational(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::NotEqual {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            _ => break,
        }
    }

    Ok(node)
}

fn relational(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut node = add(token, input, scope, global_vars, string_literals)?;

    loop {
        match token.as_ref().map(|t| &t.kind) {
            Some(TokenKind::Less) => {
                token.consume(TokenKind::Less);
                let rhs = add(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::Less {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            Some(TokenKind::LessEqual) => {
                token.consume(TokenKind::LessEqual);
                let rhs = add(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::LessEqual {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            Some(TokenKind::Greater) => {
                token.consume(TokenKind::Greater);
                let rhs = add(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::Greater {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            Some(TokenKind::GreaterEqual) => {
                token.consume(TokenKind::GreaterEqual);
                let rhs = add(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::GreaterEqual {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            }
            _ => break,
        }
    }

    Ok(node)
}

fn add(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut node = mul(token, input, scope, global_vars, string_literals)?;

    loop {
        match token.as_ref().map(|t| &t.kind) {
            Some(TokenKind::Plus) => {
                token.consume(TokenKind::Plus);
                let mut rhs = mul(token, input, scope, global_vars, string_literals)?;

                node.decay_array_type();
                rhs.decay_array_type();

                let lhs_type = node.get_node_type().cloned();
                let rhs_type = rhs.get_node_type().cloned();

                let lhs_is_ptr = lhs_type.as_ref().map_or(false, |t| t.is_pointer());
                let rhs_is_ptr = rhs_type.as_ref().map_or(false, |t| t.is_pointer());

                match (lhs_is_ptr, rhs_is_ptr) {
                    (true, false) => {
                        // p + n
                        node = ASTNode::PtrAdd {
                            ptr: Box::new(node),
                            offset: Box::new(rhs),
                            node_type: lhs_type.unwrap_or(Type::new_int()),
                        };
                    }
                    (false, true) => {
                        // n + p -> p + n
                        node = ASTNode::PtrAdd {
                            ptr: Box::new(rhs),
                            offset: Box::new(node),
                            node_type: rhs_type.unwrap_or(Type::new_int()),
                        };
                    }
                    _ => {
                        // n + n
                        node = ASTNode::Add {
                            lhs: Box::new(node),
                            rhs: Box::new(rhs),
                            node_type: Type::new_int(),
                        };
                    }
                }
            }
            Some(TokenKind::Minus) => {
                token.consume(TokenKind::Minus);
                let mut rhs = mul(token, input, scope, global_vars, string_literals)?;

                node.decay_array_type();
                rhs.decay_array_type();

                let lhs_type = node.get_node_type().cloned();
                let lhs_is_ptr = lhs_type.as_ref().map_or(false, |t| t.is_pointer());

                if lhs_is_ptr {
                    node = ASTNode::PtrSub {
                        ptr: Box::new(node),
                        offset: Box::new(rhs),
                        node_type: lhs_type.unwrap_or(Type::new_int()),
                    };
                } else {
                    node = ASTNode::Sub {
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                        node_type: Type::new_int(),
                    };
                }
            }
            _ => break,
        }
    }

    Ok(node)
}

fn mul(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut node = unary(token, input, scope, global_vars, string_literals)?;

    loop {
        match token.as_ref().map(|t| &t.kind) {
            Some(TokenKind::Star) => {
                token.consume(TokenKind::Star);
                let rhs = unary(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::Mul {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                    node_type: Type::new_int(),
                };
            }
            Some(TokenKind::Slash) => {
                token.consume(TokenKind::Slash);
                let rhs = unary(token, input, scope, global_vars, string_literals)?;
                node = ASTNode::Div {
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                    node_type: Type::new_int(),
                };
            }
            _ => break,
        }
    }

    Ok(node)
}

fn unary(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    match token.as_ref().map(|t| &t.kind) {
        Some(TokenKind::Plus) => {
            token.consume(TokenKind::Plus);
            postfix(token, input, scope, global_vars, string_literals)
        }
        Some(TokenKind::Minus) => {
            token.consume(TokenKind::Minus);
            let node = postfix(token, input, scope, global_vars, string_literals)?;
            let zero_node = ASTNode::Num {
                value: 0,
                node_type: Type::new_int(),
            };
            Ok(ASTNode::Sub {
                lhs: Box::new(zero_node),
                rhs: Box::new(node),
                node_type: Type::new_int(),
            })
        }
        Some(TokenKind::Star) => {
            token.consume(TokenKind::Star);
            let node = unary(token, input, scope, global_vars, string_literals)?;

            let deref_type = if let Some(node_type) = node.get_node_type() {
                if let Some(ref ptr_to) = node_type.ptr_to {
                    (**ptr_to).clone()
                } else {
                    Type::new_int()
                }
            } else {
                Type::new_int()
            };

            Ok(ASTNode::Deref {
                operand: Box::new(node),
                node_type: deref_type,
            })
        }
        Some(TokenKind::Ampersand) => {
            token.consume(TokenKind::Ampersand);
            let node = unary(token, input, scope, global_vars, string_literals)?;
            let addr_type = if let Some(node_type) = node.get_node_type() {
                Type::new_ptr(node_type.clone())
            } else {
                Type::new_ptr(Type::new_int())
            };

            Ok(ASTNode::Addr {
                operand: Box::new(node),
                node_type: addr_type,
            })
        }
        Some(TokenKind::Sizeof) => {
            token.consume(TokenKind::Sizeof);
            let node = unary(token, input, scope, global_vars, string_literals)?;
            let size = node.get_node_type().map_or(4, |t| t.sizeof());

            Ok(ASTNode::Num {
                value: size,
                node_type: Type::new_int(),
            })
        }
        _ => postfix(token, input, scope, global_vars, string_literals),
    }
}

fn postfix(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    let mut base = primary(token, input, scope, global_vars, string_literals)?;

    while token.consume(TokenKind::LBracket) {
        let mut index = expr(token, input, scope, global_vars, string_literals)?;
        token.expect(TokenKind::RBracket, input)?;

        base.decay_array_type();
        index.decay_array_type();

        let base_type = base.get_node_type().cloned();
        let index_type = index.get_node_type().cloned();

        let base_is_ptr = base_type.as_ref().map_or(false, |t| t.is_pointer());
        let index_is_ptr = index_type.as_ref().map_or(false, |t| t.is_pointer());

        let (ptr_node, int_node, ptr_type) = match (base_is_ptr, index_is_ptr) {
            (true, false) => (base, index, base_type.unwrap_or(Type::new_int())),
            (false, true) => (index, base, index_type.unwrap_or(Type::new_int())),
            (false, false) => return Err(ParseError::ArrayOrPointerRequired),
            (true, true) => return Err(ParseError::BothPointerUnsupported),
        };

        let add_node = ASTNode::PtrAdd {
            ptr: Box::new(ptr_node),
            offset: Box::new(int_node),
            node_type: ptr_type.clone(),
        };

        let deref_result_type = if let Some(ref ptr_to) = ptr_type.ptr_to {
            (**ptr_to).clone()
        } else {
            Type::new_int()
        };

        base = ASTNode::Deref {
            operand: Box::new(add_node),
            node_type: deref_result_type,
        };
    }

    Ok(base)
}

fn primary(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
    global_vars: &BTreeMap<String, Type>,
    string_literals: &mut Vec<String>,
) -> Result<ASTNode, ParseError> {
    if token.consume(TokenKind::LParen) {
        let node = expr(token, input, scope, global_vars, string_literals)?;
        token.expect(TokenKind::RParen, input)?;
        return Ok(node);
    }

    if let Some(t) = token.take() {
        match t.kind {
            TokenKind::Number(num) => {
                *token = t.next;
                return Ok(ASTNode::Num {
                    value: num,
                    node_type: Type::new_int(),
                });
            }
            TokenKind::CharLiteral(c) => {
                *token = t.next;
                return Ok(ASTNode::Num {
                    value: c as i32,
                    node_type: Type::new_char(),
                });
            }
            TokenKind::StringLiteral(string_content) => {
                *token = t.next;

                let string_id = if let Some(existing_id) =
                    string_literals.iter().position(|s| s == &string_content)
                {
                    existing_id
                } else {
                    let new_id = string_literals.len();
                    string_literals.push(string_content);
                    new_id
                };
                return Ok(ASTNode::StringLiteral {
                    id: string_id,
                    node_type: Type::new_ptr(Type::new_char()),
                });
            }
            TokenKind::Identifier(name) => {
                let var_name = name.to_string();
                *token = t.next;

                if token.consume(TokenKind::LParen) {
                    let mut args = Vec::new();
                    if !token.consume(TokenKind::RParen) {
                        loop {
                            let arg = expr(token, input, scope, global_vars, string_literals)?;
                            args.push(arg);
                            if !token.consume(TokenKind::Comma) {
                                break;
                            }
                        }
                        token.expect(TokenKind::RParen, input)?;
                    }

                    return Ok(ASTNode::FunctionCall {
                        name: var_name,
                        args,
                        node_type: Type::new_int(), // Default to int return type
                    });
                }

                match scope.get_variable(&var_name) {
                    Some((offset, var_type)) => {
                        return Ok(ASTNode::LocalVariable {
                            offset: *offset,
                            node_type: var_type.clone(),
                        });
                    }
                    None => match global_vars.get(&var_name) {
                        Some(var_type) => {
                            return Ok(ASTNode::GlobalVariable {
                                name: var_name,
                                node_type: var_type.clone(),
                            });
                        }

                        None => return Err(ParseError::UndefinedVariable(var_name)),
                    },
                }
            }
            _ => return Err(ParseError::UnexpectedToken(format!("{:?}", t))),
        }
    }

    Err(ParseError::UnexpectedEndOfInput)
}

fn parse_parameters(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
) -> Result<Vec<String>, ParseError> {
    let mut params = Vec::new();

    if !token.consume(TokenKind::RParen) {
        loop {
            let param_type = consume_type(token)?;

            let param_name = token.expect_identifier(input)?;

            scope.add_variable(param_name.clone(), param_type);
            params.push(param_name);

            if !token.consume(TokenKind::Comma) {
                break;
            }
        }
        token.expect(TokenKind::RParen, input)?;
    }

    Ok(params)
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
        let mut scope = FunctionScope::new();
        scope.add_variable("x".to_string(), Type::new_int());

        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
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
                expected: Box::new(ASTNode::Add {
                    lhs: Box::new(ASTNode::Mul {
                        lhs: Box::new(ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 2,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Add {
                        lhs: Box::new(ASTNode::Num {
                            value: 3,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 4,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                }),
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
                expected: Box::new(ASTNode::Mul {
                    lhs: Box::new(ASTNode::Sub {
                        lhs: Box::new(ASTNode::Num {
                            value: 0,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "1 <= 2 が正しくparseされること",
                token: TestTokenStream::new("1<=2")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::LessEqual, 1, 3)
                    .add(TokenKind::Number(2), 3, 4)
                    .build(),
                raw_input: "1 <= 2",
                expected: Box::new(ASTNode::LessEqual {
                    lhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                }),
            },
            TestCase {
                name: "変数が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Identifier("x"), "x"))),
                raw_input: "x",
                expected: Box::new(ASTNode::LocalVariable {
                    offset: 8,
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "x = 1 が正しくparseされること",
                token: TestTokenStream::new("x=1")
                    .add(TokenKind::Identifier("x"), 0, 1)
                    .add(TokenKind::Assign, 1, 2)
                    .add(TokenKind::Number(1), 2, 3)
                    .build(),
                raw_input: "x = 1",
                expected: Box::new(ASTNode::Assign {
                    lhs: Box::new(ASTNode::LocalVariable {
                        offset: 8,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "関数呼び出しを含む式が正しくparseされること",
                token: TestTokenStream::new("func()+1")
                    .add(TokenKind::Identifier("func"), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::RParen, 5, 6)
                    .add(TokenKind::Plus, 6, 7)
                    .add(TokenKind::Number(1), 7, 8)
                    .build(),
                raw_input: "func() + 1",
                expected: Box::new(ASTNode::Add {
                    lhs: Box::new(ASTNode::FunctionCall {
                        name: "func".to_string(),
                        args: vec![],
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "引数を持つ関数呼び出しが正しくparseされること",
                token: TestTokenStream::new("func(1, 2)")
                    .add(TokenKind::Identifier("func"), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::Number(1), 5, 6)
                    .add(TokenKind::Comma, 6, 7)
                    .add(TokenKind::Number(2), 7, 8)
                    .add(TokenKind::RParen, 8, 9)
                    .build(),
                raw_input: "func(1, 2)",
                expected: Box::new(ASTNode::FunctionCall {
                    name: "func".to_string(),
                    args: vec![
                        ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        },
                        ASTNode::Num {
                            value: 2,
                            node_type: Type::new_int(),
                        },
                    ],
                    node_type: Type::new_int(),
                }),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = expr(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected, "{}", case.name);
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
                expected: Box::new(ASTNode::Sub {
                    lhs: Box::new(ASTNode::Num {
                        value: 0,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "+1 が正しくparseされること",
                token: TestTokenStream::new("+1")
                    .add(TokenKind::Plus, 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "+1",
                expected: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
            },
        ];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = unary(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
        }
    }

    #[test]
    fn test_primary() {
        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "識別子が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Identifier("x"), "x"))),
                raw_input: "x",
                expected: Box::new(ASTNode::LocalVariable {
                    offset: 8,
                    node_type: Type::new_int(),
                }),
            },
            TestCase {
                name: "関数呼び出しが正しくparseされること",
                token: TestTokenStream::new("func()")
                    .add(TokenKind::Identifier("func"), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::RParen, 5, 6)
                    .build(),
                raw_input: "func()",
                expected: Box::new(ASTNode::FunctionCall {
                    name: "func".to_string(),
                    args: vec![],
                    node_type: Type::new_int(),
                }),
            },
        ];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            scope.add_variable("x".to_string(), Type::new_int());
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = primary(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
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
            expected: Box::new(ASTNode::Mul {
                lhs: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
                rhs: Box::new(ASTNode::Num {
                    value: 2,
                    node_type: Type::new_int(),
                }),
                node_type: Type::new_int(),
            }),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = mul(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
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
            expected: Box::new(ASTNode::Less {
                lhs: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
                rhs: Box::new(ASTNode::Num {
                    value: 2,
                    node_type: Type::new_int(),
                }),
            }),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = relational(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
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
            expected: Box::new(ASTNode::Equal {
                lhs: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
                rhs: Box::new(ASTNode::Num {
                    value: 2,
                    node_type: Type::new_int(),
                }),
            }),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = equality(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
        }
    }

    #[test]
    fn test_assign() {
        let test_cases = vec![TestCase {
            name: "x = 1 が正しくparseされること",
            token: TestTokenStream::new("x=1")
                .add(TokenKind::Identifier("x"), 0, 1)
                .add(TokenKind::Assign, 1, 2)
                .add(TokenKind::Number(1), 2, 3)
                .build(),
            raw_input: "x = 1",
            expected: Box::new(ASTNode::Assign {
                lhs: Box::new(ASTNode::LocalVariable {
                    offset: 8,
                    node_type: Type::new_int(),
                }),
                rhs: Box::new(ASTNode::Num {
                    value: 1,
                    node_type: Type::new_int(),
                }),
                node_type: Type::new_int(),
            }),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            scope.add_variable("x".to_string(), Type::new_int());
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = assign(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected);
        }
    }

    #[test]
    fn test_stmt() {
        let test_cases = vec![
            TestCase {
                name: "x = 1; が正しくparseされること",
                token: TestTokenStream::new("x=1;")
                    .add(TokenKind::Identifier("x"), 0, 1)
                    .add(TokenKind::Assign, 1, 2)
                    .add(TokenKind::Number(1), 2, 3)
                    .add(TokenKind::Semicolon, 3, 4)
                    .build(),
                raw_input: "x = 1;",
                expected: Box::new(ASTNode::ExpressionStatement {
                    expr: Box::new(ASTNode::Assign {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 16,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    }),
                }),
            },
            TestCase {
                name: "return文が正しくparseされること",
                token: TestTokenStream::new("return 42;")
                    .add(TokenKind::Return, 0, 6)
                    .add(TokenKind::Number(42), 7, 9)
                    .add(TokenKind::Semicolon, 9, 10)
                    .build(),
                raw_input: "return 42;",
                expected: Box::new(ASTNode::Return {
                    expr: Some(Box::new(ASTNode::Num {
                        value: 42,
                        node_type: Type::new_int(),
                    })),
                }),
            },
            TestCase {
                name: "return式が正しくparseされること",
                token: TestTokenStream::new("return x+1;")
                    .add(TokenKind::Return, 0, 6)
                    .add(TokenKind::Identifier("x"), 7, 8)
                    .add(TokenKind::Plus, 8, 9)
                    .add(TokenKind::Number(1), 9, 10)
                    .add(TokenKind::Semicolon, 10, 11)
                    .build(),
                raw_input: "return x+1;",
                expected: Box::new(ASTNode::Return {
                    expr: Some(Box::new(ASTNode::Add {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 16,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    })),
                }),
            },
            TestCase {
                name: "while文が正しくparseされること",
                token: TestTokenStream::new("while(x<10)x=x+1;")
                    .add(TokenKind::While, 0, 5)
                    .add(TokenKind::LParen, 5, 6)
                    .add(TokenKind::Identifier("x"), 6, 7)
                    .add(TokenKind::Less, 7, 8)
                    .add(TokenKind::Number(10), 8, 10)
                    .add(TokenKind::RParen, 10, 11)
                    .add(TokenKind::Identifier("x"), 11, 12)
                    .add(TokenKind::Assign, 12, 13)
                    .add(TokenKind::Identifier("x"), 13, 14)
                    .add(TokenKind::Plus, 14, 15)
                    .add(TokenKind::Number(1), 15, 16)
                    .add(TokenKind::Semicolon, 16, 17)
                    .build(),
                raw_input: "while(x<10)x=x+1;",
                expected: Box::new(ASTNode::While {
                    condition: Box::new(ASTNode::Less {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 16,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 10,
                            node_type: Type::new_int(),
                        }),
                    }),
                    body: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Assign {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 16,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Add {
                                lhs: Box::new(ASTNode::LocalVariable {
                                    offset: 16,
                                    node_type: Type::new_int(),
                                }),
                                rhs: Box::new(ASTNode::Num {
                                    value: 1,
                                    node_type: Type::new_int(),
                                }),
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                    }),
                }),
            },
            TestCase {
                name: "if-else文が正しくparseされること",
                token: TestTokenStream::new("if(x>0)y=1;else y=2;")
                    .add(TokenKind::If, 0, 2)
                    .add(TokenKind::LParen, 2, 3)
                    .add(TokenKind::Identifier("x"), 3, 4)
                    .add(TokenKind::Greater, 4, 5)
                    .add(TokenKind::Number(0), 5, 6)
                    .add(TokenKind::RParen, 6, 7)
                    .add(TokenKind::Identifier("y"), 7, 8)
                    .add(TokenKind::Assign, 8, 9)
                    .add(TokenKind::Number(1), 9, 10)
                    .add(TokenKind::Semicolon, 10, 11)
                    .add(TokenKind::Else, 11, 15)
                    .add(TokenKind::Identifier("y"), 16, 17)
                    .add(TokenKind::Assign, 17, 18)
                    .add(TokenKind::Number(2), 18, 19)
                    .add(TokenKind::Semicolon, 19, 20)
                    .build(),
                raw_input: "if(x>0)y=1;else y=2;",
                expected: Box::new(ASTNode::If {
                    condition: Box::new(ASTNode::Greater {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 16,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 0,
                            node_type: Type::new_int(),
                        }),
                    }),
                    then_stmt: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Assign {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 24,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Num {
                                value: 1,
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                    }),
                    else_stmt: Some(Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Assign {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 24,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Num {
                                value: 2,
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                    })),
                }),
            },
            TestCase {
                name: "for文が正しくparseされること",
                token: TestTokenStream::new("for(i=0;i<10;i=i+1)x=x+1;")
                    .add(TokenKind::For, 0, 3)
                    .add(TokenKind::LParen, 3, 4)
                    .add(TokenKind::Identifier("i"), 4, 5)
                    .add(TokenKind::Assign, 5, 6)
                    .add(TokenKind::Number(0), 6, 7)
                    .add(TokenKind::Semicolon, 7, 8)
                    .add(TokenKind::Identifier("i"), 8, 9)
                    .add(TokenKind::Less, 9, 10)
                    .add(TokenKind::Number(10), 10, 12)
                    .add(TokenKind::Semicolon, 12, 13)
                    .add(TokenKind::Identifier("i"), 13, 14)
                    .add(TokenKind::Assign, 14, 15)
                    .add(TokenKind::Identifier("i"), 15, 16)
                    .add(TokenKind::Plus, 16, 17)
                    .add(TokenKind::Number(1), 17, 18)
                    .add(TokenKind::RParen, 18, 19)
                    .add(TokenKind::Identifier("x"), 19, 20)
                    .add(TokenKind::Assign, 20, 21)
                    .add(TokenKind::Identifier("x"), 21, 22)
                    .add(TokenKind::Plus, 22, 23)
                    .add(TokenKind::Number(1), 23, 24)
                    .add(TokenKind::Semicolon, 24, 25)
                    .build(),
                raw_input: "for(i=0;i<10;i=i+1)x=x+1;",
                expected: Box::new(ASTNode::For {
                    init: Some(Box::new(ASTNode::Assign {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 8,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 0,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    })),
                    condition: Some(Box::new(ASTNode::Less {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 8,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 10,
                            node_type: Type::new_int(),
                        }),
                    })),
                    update: Some(Box::new(ASTNode::Assign {
                        lhs: Box::new(ASTNode::LocalVariable {
                            offset: 8,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Add {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 8,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Num {
                                value: 1,
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    })),
                    body: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Assign {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 16,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Add {
                                lhs: Box::new(ASTNode::LocalVariable {
                                    offset: 16,
                                    node_type: Type::new_int(),
                                }),
                                rhs: Box::new(ASTNode::Num {
                                    value: 1,
                                    node_type: Type::new_int(),
                                }),
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                    }),
                }),
            },
            TestCase {
                name: "空のブロック文が正しくparseされること",
                token: TestTokenStream::new("{}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::RBrace, 1, 2)
                    .build(),
                raw_input: "{}",
                expected: Box::new(ASTNode::Block { statements: vec![] }),
            },
            TestCase {
                name: "単一文のブロックが正しくparseされること",
                token: TestTokenStream::new("{x=1;}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::Identifier("x"), 1, 2)
                    .add(TokenKind::Assign, 2, 3)
                    .add(TokenKind::Number(1), 3, 4)
                    .add(TokenKind::Semicolon, 4, 5)
                    .add(TokenKind::RBrace, 5, 6)
                    .build(),
                raw_input: "{x=1;}",
                expected: Box::new(ASTNode::Block {
                    statements: vec![ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Assign {
                            lhs: Box::new(ASTNode::LocalVariable {
                                offset: 16,
                                node_type: Type::new_int(),
                            }),
                            rhs: Box::new(ASTNode::Num {
                                value: 1,
                                node_type: Type::new_int(),
                            }),
                            node_type: Type::new_int(),
                        }),
                    }],
                }),
            },
            TestCase {
                name: "複数文のブロックが正しくparseされること",
                token: TestTokenStream::new("{x=1;y=2;z=3;}")
                    .add(TokenKind::LBrace, 0, 1)
                    .add(TokenKind::Identifier("x"), 1, 2)
                    .add(TokenKind::Assign, 2, 3)
                    .add(TokenKind::Number(1), 3, 4)
                    .add(TokenKind::Semicolon, 4, 5)
                    .add(TokenKind::Identifier("y"), 5, 6)
                    .add(TokenKind::Assign, 6, 7)
                    .add(TokenKind::Number(2), 7, 8)
                    .add(TokenKind::Semicolon, 8, 9)
                    .add(TokenKind::Identifier("z"), 9, 10)
                    .add(TokenKind::Assign, 10, 11)
                    .add(TokenKind::Number(3), 11, 12)
                    .add(TokenKind::Semicolon, 12, 13)
                    .add(TokenKind::RBrace, 13, 14)
                    .build(),
                raw_input: "{x=1;y=2;z=3;}",
                expected: Box::new(ASTNode::Block {
                    statements: vec![
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Assign {
                                lhs: Box::new(ASTNode::LocalVariable {
                                    offset: 16,
                                    node_type: Type::new_int(),
                                }),
                                rhs: Box::new(ASTNode::Num {
                                    value: 1,
                                    node_type: Type::new_int(),
                                }),
                                node_type: Type::new_int(),
                            }),
                        },
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Assign {
                                lhs: Box::new(ASTNode::LocalVariable {
                                    offset: 24,
                                    node_type: Type::new_int(),
                                }),
                                rhs: Box::new(ASTNode::Num {
                                    value: 2,
                                    node_type: Type::new_int(),
                                }),
                                node_type: Type::new_int(),
                            }),
                        },
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Assign {
                                lhs: Box::new(ASTNode::LocalVariable {
                                    offset: 32,
                                    node_type: Type::new_int(),
                                }),
                                rhs: Box::new(ASTNode::Num {
                                    value: 3,
                                    node_type: Type::new_int(),
                                }),
                                node_type: Type::new_int(),
                            }),
                        },
                    ],
                }),
            },
        ];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            scope.add_variable("i".to_string(), Type::new_int()); // オフセット 8
            scope.add_variable("x".to_string(), Type::new_int()); // オフセット 16
            scope.add_variable("y".to_string(), Type::new_int()); // オフセット 24
            scope.add_variable("z".to_string(), Type::new_int()); // オフセット 32
            let mut token = case.token;
            let global_vars = BTreeMap::new();
            let mut string_literals = Vec::new();
            let result = stmt(
                &mut token,
                case.raw_input,
                &mut scope,
                &global_vars,
                &mut string_literals,
            )
            .unwrap();
            assert_eq!(result, *case.expected, "{}", case.name);
        }
    }
    #[test]
    fn test_program() {
        use crate::ast::{Function, Program};

        struct ProgramTestCase<'a> {
            name: &'a str,
            token: Option<Box<Token<'a>>>,
            raw_input: &'a str,
            expected: Program,
        }

        let test_cases = vec![ProgramTestCase {
            name: "int x; int y; x=1; y=2; が正しくparseされること",
            token: TestTokenStream::new("int main() { int x; int y; x=1; y=2; }")
                .add(TokenKind::Int, 0, 3)
                .add(TokenKind::Identifier("main"), 4, 8)
                .add(TokenKind::LParen, 8, 9)
                .add(TokenKind::RParen, 9, 10)
                .add(TokenKind::LBrace, 11, 12)
                .add(TokenKind::Int, 13, 16)
                .add(TokenKind::Identifier("x"), 17, 18)
                .add(TokenKind::Semicolon, 18, 19)
                .add(TokenKind::Int, 20, 23)
                .add(TokenKind::Identifier("y"), 24, 25)
                .add(TokenKind::Semicolon, 25, 26)
                .add(TokenKind::Identifier("x"), 27, 28)
                .add(TokenKind::Assign, 28, 29)
                .add(TokenKind::Number(1), 29, 30)
                .add(TokenKind::Semicolon, 30, 31)
                .add(TokenKind::Identifier("y"), 32, 33)
                .add(TokenKind::Assign, 33, 34)
                .add(TokenKind::Number(2), 34, 35)
                .add(TokenKind::Semicolon, 35, 36)
                .add(TokenKind::RBrace, 37, 38)
                .build(),
            raw_input: "int main() { int x; int y; x=1; y=2; }",
            expected: Program {
                functions: vec![Function {
                    name: "main".to_string(),
                    params: vec![],
                    body: ASTNode::Block {
                        statements: vec![
                            ASTNode::Block { statements: vec![] }, // int x;
                            ASTNode::Block { statements: vec![] }, // int y;
                            ASTNode::ExpressionStatement {
                                expr: Box::new(ASTNode::Assign {
                                    lhs: Box::new(ASTNode::LocalVariable {
                                        offset: 8,
                                        node_type: Type::new_int(),
                                    }),
                                    rhs: Box::new(ASTNode::Num {
                                        value: 1,
                                        node_type: Type::new_int(),
                                    }),
                                    node_type: Type::new_int(),
                                }),
                            },
                            ASTNode::ExpressionStatement {
                                expr: Box::new(ASTNode::Assign {
                                    lhs: Box::new(ASTNode::LocalVariable {
                                        offset: 16,
                                        node_type: Type::new_int(),
                                    }),
                                    rhs: Box::new(ASTNode::Num {
                                        value: 2,
                                        node_type: Type::new_int(),
                                    }),
                                    node_type: Type::new_int(),
                                }),
                            },
                        ],
                    },
                }],
                global_vars: BTreeMap::new(),
                string_literals: vec![],
            },
        }];

        for case in test_cases {
            let mut token = case.token;
            let result = program(&mut token, case.raw_input).unwrap();
            assert_eq!(result, case.expected, "{}", case.name);
        }
    }
}
