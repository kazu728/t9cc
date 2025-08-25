use super::token::{Token, TokenKind};
use std::collections::HashMap;
use std::iter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Int,
    Ptr,
    Array,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub kind: TypeKind,
    pub ptr_to: Option<Box<Type>>,
    pub array_size: Option<usize>,
}

impl Type {
    pub fn new_int() -> Self {
        Type {
            kind: TypeKind::Int,
            ptr_to: None,
            array_size: None,
        }
    }

    pub fn new_ptr(base_type: Type) -> Self {
        Type {
            kind: TypeKind::Ptr,
            ptr_to: Some(Box::new(base_type)),
            array_size: None,
        }
    }

    pub fn new_array(element_type: Type, size: usize) -> Self {
        Type {
            kind: TypeKind::Array,
            ptr_to: Some(Box::new(element_type)),
            array_size: Some(size),
        }
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self.kind, TypeKind::Ptr)
    }

    pub fn is_array(&self) -> bool {
        matches!(self.kind, TypeKind::Array)
    }

    pub fn decay_array_to_pointer(&self) -> Type {
        match self.kind {
            TypeKind::Array => {
                if let Some(element_type) = &self.ptr_to {
                    Type::new_ptr((**element_type).clone())
                } else {
                    panic!("Invalid array type: no element type");
                }
            }
            _ => self.clone(),
        }
    }

    pub fn sizeof(&self) -> i32 {
        match self.kind {
            TypeKind::Int => 4,
            TypeKind::Ptr => 8,
            TypeKind::Array => {
                if let (Some(element_type), Some(array_size)) = (&self.ptr_to, self.array_size) {
                    element_type.sizeof() * array_size as i32
                } else {
                    panic!("Invalid array type");
                }
            }
        }
    }
}

struct FunctionScope {
    variables: HashMap<String, (u32, Type)>,
    next_offset: u32,
}

impl FunctionScope {
    fn new() -> Self {
        FunctionScope {
            variables: HashMap::new(),
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

#[derive(Debug, PartialEq, Eq, Clone)]
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
    Addr,
    Deref,

    PtrAdd,
    PtrSub,

    Return,
    If,
    IfBody,
    Else,
    While,
    For,
    ForInit,
    ForUpdate,

    Block,
    FunctionCall,
    FunctionDef,
    FunctionName(String),
    Parameter(String),
    VarDecl, // 変数宣言
}

pub type MaybeASTNode = Option<Box<ASTNode>>;

#[derive(Debug, PartialEq, Eq)]
pub struct ASTNode {
    pub kind: ASTNodeKind,
    pub lhs: MaybeASTNode,
    pub rhs: MaybeASTNode,
    pub node_type: Option<Type>,
}

impl ASTNode {
    pub fn new(kind: ASTNodeKind, lhs: MaybeASTNode, rhs: MaybeASTNode) -> ASTNode {
        ASTNode {
            kind,
            lhs,
            rhs,
            node_type: None,
        }
    }

    pub fn new_with_type(
        kind: ASTNodeKind,
        lhs: MaybeASTNode,
        rhs: MaybeASTNode,
        node_type: Option<Type>,
    ) -> ASTNode {
        ASTNode {
            kind,
            lhs,
            rhs,
            node_type,
        }
    }

    pub fn new_num_with_type(num: i32) -> Box<ASTNode> {
        Box::new(ASTNode::new_with_type(
            ASTNodeKind::Num(num),
            None,
            None,
            Some(Type::new_int()),
        ))
    }

    pub fn new_local_var_with_type(offset: u32, var_type: Type) -> Box<ASTNode> {
        Box::new(ASTNode::new_with_type(
            ASTNodeKind::LocalVariable(offset),
            None,
            None,
            Some(var_type),
        ))
    }

    pub fn new_binary_with_type(
        kind: ASTNodeKind,
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        result_type: Type,
    ) -> Box<ASTNode> {
        Box::new(ASTNode::new_with_type(
            kind,
            Some(lhs),
            Some(rhs),
            Some(result_type),
        ))
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

    pub fn decay_array_type(&mut self) {
        if let Some(ref node_type) = self.node_type {
            if node_type.is_array() {
                let new_type = node_type.decay_array_to_pointer();
                self.node_type = Some(new_type);
            }
        }
    }
}

// program    = function_def*
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

pub fn program(token: &mut Option<Box<Token>>, input: &str) -> Vec<Box<ASTNode>> {
    let mut functions: Vec<Box<ASTNode>> = vec![];

    while token.is_some() {
        functions.push(function_def(token, input));
    }

    functions
}

fn function_def(token: &mut Option<Box<Token>>, input: &str) -> Box<ASTNode> {
    Token::expect(token, TokenKind::Int, input);
    let function_name = Token::expect_identifier(token, input);

    Token::expect(token, TokenKind::LParen, input);

    let mut scope = FunctionScope::new();
    let params = parse_parameters(token, input, &mut scope);

    Token::expect(token, TokenKind::LBrace, input);

    let mut body_stmts = Vec::new();
    while !Token::consume(token, TokenKind::RBrace) {
        body_stmts.push(stmt(token, input, &mut scope));
    }

    let body = build_block_ast(body_stmts);
    let params_node = build_params_ast(params);

    ASTNode::new_boxed(
        ASTNodeKind::FunctionDef,
        Some(ASTNode::new_boxed(
            ASTNodeKind::FunctionName(function_name),
            params_node,
            None,
        )),
        Some(body),
    )
}

fn stmt(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::LBrace) {
        let mut stmts = vec![];
        while !Token::consume(token, TokenKind::RBrace) {
            stmts.push(stmt(token, input, scope));
        }

        return build_block_ast(stmts);
    }

    if Token::consume(token, TokenKind::Int) {
        fn create_base_type(ptr_count: usize) -> Type {
            (0..ptr_count).fold(Type::new_int(), |acc, _| Type::new_ptr(acc))
        }

        fn parse_array_size(token: &mut Option<Box<Token>>, input: &str) -> Option<usize> {
            if let Some(size_token) = token.take() {
                if let TokenKind::Number(array_size) = size_token.kind {
                    *token = size_token.next;
                    Token::expect(token, TokenKind::RBracket, input);
                    Some(array_size as usize)
                } else {
                    panic!("配列のサイズは数値である必要があります");
                }
            } else {
                panic!("配列のサイズが指定されていません");
            }
        }

        let ptr_count =
            iter::from_fn(|| Token::consume(token, TokenKind::Star).then_some(())).count();

        let var_name = Token::expect_identifier(token, input);

        let var_type = if Token::consume(token, TokenKind::LBracket) {
            let array_size = parse_array_size(token, input).unwrap();
            let base_type = create_base_type(ptr_count);
            Type::new_array(base_type, array_size)
        } else {
            create_base_type(ptr_count)
        };

        scope.add_variable(var_name, var_type);
        Token::expect(token, TokenKind::Semicolon, input);
        return ASTNode::leaf(ASTNodeKind::VarDecl);
    }

    if Token::consume(token, TokenKind::If) {
        Token::expect(token, TokenKind::LParen, input);
        let cond_node = expr(token, input, scope);
        Token::expect(token, TokenKind::RParen, input);
        let then_node = stmt(token, input, scope);
        let else_node = if Token::consume(token, TokenKind::Else) {
            Some(stmt(token, input, scope))
        } else {
            None
        };
        // If { lhs: 条件式, rhs: IfBody { lhs: then節, rhs: else節 } }
        let if_body = ASTNode::new_boxed(ASTNodeKind::IfBody, Some(then_node), else_node);
        return ASTNode::new_boxed(ASTNodeKind::If, Some(cond_node), Some(if_body));
    }

    if Token::consume(token, TokenKind::While) {
        Token::expect(token, TokenKind::LParen, input);
        let cond_node = expr(token, input, scope);
        Token::expect(token, TokenKind::RParen, input);
        let body_node = stmt(token, input, scope);
        return ASTNode::new_boxed(ASTNodeKind::While, Some(cond_node), Some(body_node));
    }

    if Token::consume(token, TokenKind::For) {
        Token::expect(token, TokenKind::LParen, input);

        let init_node = if !Token::consume(token, TokenKind::Semicolon) {
            let init = expr(token, input, scope);
            Token::expect(token, TokenKind::Semicolon, input);
            Some(init)
        } else {
            None
        };

        let cond_node = if !Token::consume(token, TokenKind::Semicolon) {
            let cond = expr(token, input, scope);
            Token::expect(token, TokenKind::Semicolon, input);
            Some(cond)
        } else {
            None
        };

        let update_node = if !Token::consume(token, TokenKind::RParen) {
            let update = expr(token, input, scope);
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
                Some(stmt(token, input, scope)),
            )),
        );
    }

    if Token::consume(token, TokenKind::Return) {
        let expr_node = expr(token, input, scope);
        Token::expect(token, TokenKind::Semicolon, input);
        return ASTNode::unary(ASTNodeKind::Return, expr_node);
    }

    let node = expr(token, input, scope);
    Token::expect(token, TokenKind::Semicolon, input);
    node
}

fn expr(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    assign(token, input, scope)
}

fn assign(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    let mut node = equality(token, input, scope);

    if Token::consume(token, TokenKind::Assign) {
        let mut rhs = assign(token, input, scope);

        rhs.decay_array_type();

        node = ASTNode::binary(ASTNodeKind::Assign, node, rhs);
    }

    node
}

fn equality(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
) -> Box<ASTNode> {
    let mut node = relational(token, input, scope);

    loop {
        if Token::consume(token, TokenKind::Equal) {
            let rhs = relational(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::Equal, lhs_node, rhs_node);
        } else if Token::consume(token, TokenKind::NotEqual) {
            let rhs = relational(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::NotEqual, lhs_node, rhs_node);
        } else {
            break;
        }
    }

    node
}

fn convert_arrays_to_pointers(
    lhs: Box<ASTNode>,
    rhs: Box<ASTNode>,
) -> (Box<ASTNode>, Box<ASTNode>) {
    let mut lhs_node = lhs;
    lhs_node.decay_array_type();

    let mut rhs_node = rhs;
    rhs_node.decay_array_type();

    (lhs_node, rhs_node)
}

fn relational(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
) -> Box<ASTNode> {
    let mut node = add(token, input, scope);

    loop {
        if Token::consume(token, TokenKind::Less) {
            let rhs = add(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::Less, lhs_node, rhs_node);
        } else if Token::consume(token, TokenKind::LessEqual) {
            let rhs = add(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::LessEqual, lhs_node, rhs_node);
        } else if Token::consume(token, TokenKind::Greater) {
            let rhs = add(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::Greater, lhs_node, rhs_node);
        } else if Token::consume(token, TokenKind::GreaterEqual) {
            let rhs = add(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = ASTNode::binary(ASTNodeKind::GreaterEqual, lhs_node, rhs_node);
        } else {
            break;
        }
    }

    node
}

fn add(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    let mut node = mul(token, input, scope);

    loop {
        if Token::consume(token, TokenKind::Plus) {
            let rhs = mul(token, input, scope);

            let mut lhs_node = node;
            lhs_node.decay_array_type();

            let mut rhs_node = rhs;
            rhs_node.decay_array_type();

            let lhs_is_ptr = lhs_node
                .node_type
                .as_ref()
                .map_or(false, |t| t.is_pointer());
            let rhs_is_ptr = rhs_node
                .node_type
                .as_ref()
                .map_or(false, |t| t.is_pointer());

            let (op_kind, final_lhs, final_rhs) = match (lhs_is_ptr, rhs_is_ptr) {
                (true, false) => (ASTNodeKind::PtrAdd, lhs_node, rhs_node), // p + n
                (false, true) => (ASTNodeKind::PtrAdd, rhs_node, lhs_node), // n + p -> p + n
                _ => (ASTNodeKind::Add, lhs_node, rhs_node),                // n + n
            };

            let result_type = if matches!(op_kind, ASTNodeKind::PtrAdd) {
                final_lhs.node_type.clone()
            } else {
                // TODO: デフォルトintを直す
                Some(Type::new_int())
            };

            node = Box::new(ASTNode::new_with_type(
                op_kind,
                Some(final_lhs),
                Some(final_rhs),
                result_type,
            ));
        } else if Token::consume(token, TokenKind::Minus) {
            let rhs = mul(token, input, scope);

            let mut lhs_node = node;
            lhs_node.decay_array_type();

            let mut rhs_node = rhs;
            rhs_node.decay_array_type();

            let lhs_is_ptr = lhs_node
                .node_type
                .as_ref()
                .map_or(false, |t| t.is_pointer());

            let (op_kind, result_type) = if lhs_is_ptr {
                (ASTNodeKind::PtrSub, lhs_node.node_type.clone())
            } else {
                (ASTNodeKind::Sub, Some(Type::new_int()))
            };

            node = Box::new(ASTNode::new_with_type(
                op_kind,
                Some(lhs_node),
                Some(rhs_node),
                result_type,
            ));
        } else {
            break;
        }
    }

    node
}

fn mul(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    let mut node = unary(token, input, scope);

    loop {
        if Token::consume(token, TokenKind::Star) {
            let rhs = unary(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = Box::new(ASTNode::new_with_type(
                ASTNodeKind::Mul,
                Some(lhs_node),
                Some(rhs_node),
                Some(Type::new_int()),
            ));
        } else if Token::consume(token, TokenKind::Slash) {
            let rhs = unary(token, input, scope);
            let (lhs_node, rhs_node) = convert_arrays_to_pointers(node, rhs);
            node = Box::new(ASTNode::new_with_type(
                ASTNodeKind::Div,
                Some(lhs_node),
                Some(rhs_node),
                Some(Type::new_int()),
            ));
        } else {
            break;
        }
    }

    node
}

fn unary(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::Plus) {
        return primary(token, input, scope);
    } else if Token::consume(token, TokenKind::Minus) {
        let node = primary(token, input, scope);
        let mut zero_node = ASTNode::leaf(ASTNodeKind::Num(0));
        zero_node.node_type = Some(Type::new_int());
        return ASTNode::binary(ASTNodeKind::Sub, zero_node, node);
    } else if Token::consume(token, TokenKind::Star) {
        let mut node = unary(token, input, scope);
        node.decay_array_type();

        let mut deref_node = ASTNode::unary(ASTNodeKind::Deref, node);

        if let Some(ref node_type) = deref_node.lhs.as_ref().unwrap().node_type {
            if let Some(ref ptr_to) = node_type.ptr_to {
                deref_node.node_type = Some((**ptr_to).clone());
            }
        }
        return deref_node;
    } else if Token::consume(token, TokenKind::Ampersand) {
        let node = unary(token, input, scope);
        let mut addr_node = ASTNode::unary(ASTNodeKind::Addr, node);

        if let Some(ref node_type) = addr_node.lhs.as_ref().unwrap().node_type {
            addr_node.node_type = Some(Type::new_ptr(node_type.clone()));
        }
        return addr_node;
    } else if Token::consume(token, TokenKind::Sizeof) {
        let node = unary(token, input, scope);
        let size = node.node_type.as_ref().map_or(4, |t| t.sizeof());

        return Box::new(ASTNode::new_with_type(
            ASTNodeKind::Num(size),
            None,
            None,
            Some(Type::new_int()),
        ));
    }
    primary(token, input, scope)
}

fn primary(token: &mut Option<Box<Token>>, input: &str, scope: &mut FunctionScope) -> Box<ASTNode> {
    if Token::consume(token, TokenKind::LParen) {
        let node = expr(token, input, scope);
        Token::expect(token, TokenKind::RParen, input);
        return node;
    }

    if let Some(t) = token.take() {
        match t.kind {
            TokenKind::Number(num) => {
                *token = t.next;
                return Box::new(ASTNode::new_with_type(
                    ASTNodeKind::Num(num),
                    None,
                    None,
                    Some(Type::new_int()),
                ));
            }
            TokenKind::Identifier(var) => {
                let var_name = var.get_name().to_string();
                *token = t.next;

                if Token::consume(token, TokenKind::LParen) {
                    let mut args = Vec::new();
                    if !Token::consume(token, TokenKind::RParen) {
                        loop {
                            args.push(expr(token, input, scope));
                            if !Token::consume(token, TokenKind::Comma) {
                                break;
                            }
                        }
                        Token::expect(token, TokenKind::RParen, input);
                    }

                    let arg_list = args.into_iter().rev().fold(None, |acc, arg| {
                        Some(ASTNode::new_boxed(ASTNodeKind::Block, Some(arg), acc))
                    });

                    return ASTNode::new_boxed(
                        ASTNodeKind::FunctionCall,
                        Some(ASTNode::leaf(ASTNodeKind::FunctionName(var_name))),
                        arg_list,
                    );
                }

                match scope.get_variable(&var_name) {
                    Some((offset, var_type)) => {
                        let var_type = var_type.clone();

                        let mut variable_node = Box::new(ASTNode::new_with_type(
                            ASTNodeKind::LocalVariable(*offset),
                            None,
                            None,
                            Some(var_type.clone()),
                        ));

                        if Token::consume(token, TokenKind::LBracket) {
                            let index_expr = expr(token, input, scope);
                            Token::expect(token, TokenKind::RBracket, input);

                            let (add_op_kind, add_result_type) = match &var_type.kind {
                                TypeKind::Ptr => (ASTNodeKind::PtrAdd, Some(var_type.clone())),
                                TypeKind::Array => match &var_type.ptr_to {
                                    Some(element_type) => (
                                        ASTNodeKind::PtrAdd,
                                        Some(Type::new_ptr((**element_type).clone())),
                                    ),
                                    None => panic!("Invalid array type"),
                                },
                                _ => panic!("配列またはポインタである必要があります"),
                            };

                            let add_node = Box::new(ASTNode::new_with_type(
                                add_op_kind,
                                Some(variable_node),
                                Some(index_expr),
                                add_result_type.clone(),
                            ));

                            let deref_result_type =
                                add_result_type.and_then(|ptr_type| ptr_type.ptr_to.map(|t| *t));

                            variable_node = Box::new(ASTNode::new_with_type(
                                ASTNodeKind::Deref,
                                Some(add_node),
                                None,
                                deref_result_type,
                            ));
                        }

                        return variable_node;
                    }
                    None => panic!("未定義の変数: {}", var_name),
                }
            }
            _ => unreachable!("{:?}", t),
        }
    }

    unreachable!("unexpected token: {:?}", token)
}

fn parse_parameters(
    token: &mut Option<Box<Token>>,
    input: &str,
    scope: &mut FunctionScope,
) -> Vec<String> {
    let mut params = Vec::new();

    if !Token::consume(token, TokenKind::RParen) {
        loop {
            Token::expect(token, TokenKind::Int, input);
            let param_name = Token::expect_identifier(token, input);
            scope.add_variable(param_name.clone(), Type::new_int());
            params.push(param_name);

            if !Token::consume(token, TokenKind::Comma) {
                break;
            }
        }
        Token::expect(token, TokenKind::RParen, input);
    }

    params
}

fn build_block_ast(stmts: Vec<Box<ASTNode>>) -> Box<ASTNode> {
    if stmts.is_empty() {
        ASTNode::leaf(ASTNodeKind::Block)
    } else {
        let mut body = stmts.into_iter().collect::<Vec<_>>();
        let mut result = body.pop().unwrap();
        while let Some(stmt) = body.pop() {
            result = ASTNode::new_boxed(ASTNodeKind::Block, Some(stmt), Some(result));
        }
        result
    }
}

fn build_params_ast(params: Vec<String>) -> Option<Box<ASTNode>> {
    if params.is_empty() {
        None
    } else {
        let mut param_list = None;
        for param in params.into_iter().rev() {
            param_list = Some(ASTNode::new_boxed(
                ASTNodeKind::Parameter(param),
                None,
                param_list,
            ));
        }
        param_list
    }
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
        let mut scope = FunctionScope::new();
        scope.add_variable("x".to_string(), Type::new_int());

        let test_cases = vec![
            TestCase {
                name: "数値が正しくparseされること",
                token: Some(Box::new(Token::init(TokenKind::Number(1), "1"))),
                raw_input: "1",
                expected: ASTNode::new_num_with_type(1),
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
                expected: ASTNode::new_binary_with_type(
                    ASTNodeKind::Add,
                    ASTNode::new_binary_with_type(
                        ASTNodeKind::Mul,
                        ASTNode::new_num_with_type(1),
                        ASTNode::new_num_with_type(2),
                        Type::new_int(),
                    ),
                    ASTNode::new_binary_with_type(
                        ASTNodeKind::Add,
                        ASTNode::new_num_with_type(3),
                        ASTNode::new_num_with_type(4),
                        Type::new_int(),
                    ),
                    Type::new_int(),
                ),
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
                expected: ASTNode::new_binary_with_type(
                    ASTNodeKind::Mul,
                    {
                        let sub_node = ASTNode::new(
                            ASTNodeKind::Sub,
                            Some(ASTNode::new_num_with_type(0)),
                            Some(ASTNode::new_num_with_type(1)),
                        );
                        Box::new(sub_node)
                    },
                    ASTNode::new_num_with_type(2),
                    Type::new_int(),
                ),
            },
            TestCase {
                name: "1 <= 2 が正しくparseされること",
                token: TestTokenStream::new("1<=2")
                    .add(TokenKind::Number(1), 0, 1)
                    .add(TokenKind::LessEqual, 1, 3)
                    .add(TokenKind::Number(2), 3, 4)
                    .build(),
                raw_input: "1 <= 2",
                expected: {
                    let node = ASTNode::new(
                        ASTNodeKind::LessEqual,
                        Some(ASTNode::new_num_with_type(1)),
                        Some(ASTNode::new_num_with_type(2)),
                    );
                    Box::new(node)
                },
            },
            TestCase {
                name: "変数が正しくparseされること",
                token: Some(Box::new(Token::init(
                    TokenKind::Identifier(LocalVariable::new("x", 0)),
                    "x",
                ))),
                raw_input: "x",
                expected: ASTNode::new_local_var_with_type(8, Type::new_int()),
            },
            TestCase {
                name: "x = 1 が正しくparseされること",
                token: TestTokenStream::new("x=1")
                    .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 0, 1)
                    .add(TokenKind::Assign, 1, 2)
                    .add(TokenKind::Number(1), 2, 3)
                    .build(),
                raw_input: "x = 1",
                expected: {
                    let assign_node = ASTNode::new(
                        ASTNodeKind::Assign,
                        Some(ASTNode::new_local_var_with_type(8, Type::new_int())),
                        Some(ASTNode::new_num_with_type(1)),
                    );
                    Box::new(assign_node)
                },
            },
            TestCase {
                name: "関数呼び出しを含む式が正しくparseされること",
                token: TestTokenStream::new("func()+1")
                    .add(TokenKind::Identifier(LocalVariable::new("func", 0)), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::RParen, 5, 6)
                    .add(TokenKind::Plus, 6, 7)
                    .add(TokenKind::Number(1), 7, 8)
                    .build(),
                raw_input: "func() + 1",
                expected: ASTNode::new_binary_with_type(
                    ASTNodeKind::Add,
                    {
                        let func_call = ASTNode::new(
                            ASTNodeKind::FunctionCall,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::FunctionName("func".to_string()),
                                None,
                                None,
                            ))),
                            None,
                        );
                        Box::new(func_call)
                    },
                    ASTNode::new_num_with_type(1),
                    Type::new_int(),
                ),
            },
            TestCase {
                name: "引数を持つ関数呼び出しが正しくparseされること",
                token: TestTokenStream::new("func(1, 2)")
                    .add(TokenKind::Identifier(LocalVariable::new("func", 0)), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::Number(1), 5, 6)
                    .add(TokenKind::Comma, 6, 7)
                    .add(TokenKind::Number(2), 7, 8)
                    .add(TokenKind::RParen, 8, 9)
                    .build(),
                raw_input: "func(1, 2)",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::FunctionCall,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::FunctionName("func".to_string()),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Block,
                        Some(ASTNode::new_num_with_type(1)),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Block,
                            Some(ASTNode::new_num_with_type(2)),
                            None,
                        ))),
                    ))),
                )),
            },
        ];

        for case in test_cases {
            let mut token = case.token;
            let result = expr(&mut token, case.raw_input, &mut scope);
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
                    Some(ASTNode::new_num_with_type(0)),
                    Some(ASTNode::new_num_with_type(1)),
                )),
            },
            TestCase {
                name: "+1 が正しくparseされること",
                token: TestTokenStream::new("+1")
                    .add(TokenKind::Plus, 0, 1)
                    .add(TokenKind::Number(1), 1, 2)
                    .build(),
                raw_input: "+1",
                expected: ASTNode::new_num_with_type(1),
            },
        ];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let result = unary(&mut token, case.raw_input, &mut scope);
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
                expected: ASTNode::new_num_with_type(1),
            },
            TestCase {
                name: "識別子が正しくparseされること",
                token: Some(Box::new(Token::init(
                    TokenKind::Identifier(LocalVariable::new("x", 0)),
                    "x",
                ))),
                raw_input: "x",
                expected: ASTNode::new_local_var_with_type(8, Type::new_int()),
            },
            TestCase {
                name: "関数呼び出しが正しくparseされること",
                token: TestTokenStream::new("func()")
                    .add(TokenKind::Identifier(LocalVariable::new("func", 0)), 0, 4)
                    .add(TokenKind::LParen, 4, 5)
                    .add(TokenKind::RParen, 5, 6)
                    .build(),
                raw_input: "func()",
                expected: Box::new(ASTNode::new(
                    ASTNodeKind::FunctionCall,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::FunctionName("func".to_string()),
                        None,
                        None,
                    ))),
                    None,
                )),
            },
        ];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            scope.add_variable("x".to_string(), Type::new_int());
            let mut token = case.token;
            let result = primary(&mut token, case.raw_input, &mut scope);
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
            expected: ASTNode::new_binary_with_type(
                ASTNodeKind::Mul,
                ASTNode::new_num_with_type(1),
                ASTNode::new_num_with_type(2),
                Type::new_int(),
            ),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let result = mul(&mut token, case.raw_input, &mut scope);
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
                Some(ASTNode::new_num_with_type(1)),
                Some(ASTNode::new_num_with_type(2)),
            )),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let result = relational(&mut token, case.raw_input, &mut scope);
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
                Some(ASTNode::new_num_with_type(1)),
                Some(ASTNode::new_num_with_type(2)),
            )),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            let mut token = case.token;
            let result = equality(&mut token, case.raw_input, &mut scope);
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
                Some(ASTNode::new_local_var_with_type(8, Type::new_int())),
                Some(ASTNode::new_num_with_type(1)),
            )),
        }];

        for case in test_cases {
            let mut scope = FunctionScope::new();
            scope.add_variable("x".to_string(), Type::new_int());
            let mut token = case.token;
            let result = assign(&mut token, case.raw_input, &mut scope);
            assert_eq!(result, case.expected);
        }
    }

    // #[test]
    // fn test_stmt() {
    //     let test_cases = vec![
    //         TestCase {
    //             name: "x = 1; が正しくparseされること",
    //             token: TestTokenStream::new("x=1;")
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 0, 1)
    //                 .add(TokenKind::Assign, 1, 2)
    //                 .add(TokenKind::Number(1), 2, 3)
    //                 .add(TokenKind::Semicolon, 3, 4)
    //                 .build(),
    //             raw_input: "x = 1;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::Assign,
    //                 Some(ASTNode::new_local_var_with_type(16, Type::new_int())),
    //                 Some(ASTNode::new_num_with_type(1)),
    //             )),
    //         },
    //         TestCase {
    //             name: "return文が正しくparseされること",
    //             token: TestTokenStream::new("return 42;")
    //                 .add(TokenKind::Return, 0, 6)
    //                 .add(TokenKind::Number(42), 7, 9)
    //                 .add(TokenKind::Semicolon, 9, 10)
    //                 .build(),
    //             raw_input: "return 42;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::Return,
    //                 Some(ASTNode::new_num_with_type(42)),
    //                 None,
    //             )),
    //         },
    //         TestCase {
    //             name: "return式が正しくparseされること",
    //             token: TestTokenStream::new("return x+1;")
    //                 .add(TokenKind::Return, 0, 6)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 7, 8)
    //                 .add(TokenKind::Plus, 8, 9)
    //                 .add(TokenKind::Number(1), 9, 10)
    //                 .add(TokenKind::Semicolon, 10, 11)
    //                 .build(),
    //             raw_input: "return x+1;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::Return,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Add,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                 ))),
    //                 None,
    //             )),
    //         },
    //         TestCase {
    //             name: "while文が正しくparseされること",
    //             token: TestTokenStream::new("while(x<10)x=x+1;")
    //                 .add(TokenKind::While, 0, 5)
    //                 .add(TokenKind::LParen, 5, 6)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 6, 7)
    //                 .add(TokenKind::Less, 7, 8)
    //                 .add(TokenKind::Number(10), 8, 10)
    //                 .add(TokenKind::RParen, 10, 11)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 11, 12)
    //                 .add(TokenKind::Assign, 12, 13)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 13, 14)
    //                 .add(TokenKind::Plus, 14, 15)
    //                 .add(TokenKind::Number(1), 15, 16)
    //                 .add(TokenKind::Semicolon, 16, 17)
    //                 .build(),
    //             raw_input: "while(x<10)x=x+1;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::While,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Less,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(10), None, None))),
    //                 ))),
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Assign,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Add,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(8),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                     ))),
    //                 ))),
    //             )),
    //         },
    //         TestCase {
    //             name: "if-else文が正しくparseされること",
    //             token: TestTokenStream::new("if(x>0)y=1;else y=2;")
    //                 .add(TokenKind::If, 0, 2)
    //                 .add(TokenKind::LParen, 2, 3)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 3, 4)
    //                 .add(TokenKind::Greater, 4, 5)
    //                 .add(TokenKind::Number(0), 5, 6)
    //                 .add(TokenKind::RParen, 6, 7)
    //                 .add(TokenKind::Identifier(LocalVariable::new("y", 0)), 7, 8)
    //                 .add(TokenKind::Assign, 8, 9)
    //                 .add(TokenKind::Number(1), 9, 10)
    //                 .add(TokenKind::Semicolon, 10, 11)
    //                 .add(TokenKind::Else, 11, 15)
    //                 .add(TokenKind::Identifier(LocalVariable::new("y", 0)), 16, 17)
    //                 .add(TokenKind::Assign, 17, 18)
    //                 .add(TokenKind::Number(2), 18, 19)
    //                 .add(TokenKind::Semicolon, 19, 20)
    //                 .build(),
    //             raw_input: "if(x>0)y=1;else y=2;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::If,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Greater,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
    //                 ))),
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::IfBody,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(16),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                     ))),
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(16),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
    //                     ))),
    //                 ))),
    //             )),
    //         },
    //         TestCase {
    //             name: "for文が正しくparseされること",
    //             token: TestTokenStream::new("for(i=0;i<10;i=i+1)x=x+1;")
    //                 .add(TokenKind::For, 0, 3)
    //                 .add(TokenKind::LParen, 3, 4)
    //                 .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 4, 5)
    //                 .add(TokenKind::Assign, 5, 6)
    //                 .add(TokenKind::Number(0), 6, 7)
    //                 .add(TokenKind::Semicolon, 7, 8)
    //                 .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 8, 9)
    //                 .add(TokenKind::Less, 9, 10)
    //                 .add(TokenKind::Number(10), 10, 12)
    //                 .add(TokenKind::Semicolon, 12, 13)
    //                 .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 13, 14)
    //                 .add(TokenKind::Assign, 14, 15)
    //                 .add(TokenKind::Identifier(LocalVariable::new("i", 0)), 15, 16)
    //                 .add(TokenKind::Plus, 16, 17)
    //                 .add(TokenKind::Number(1), 17, 18)
    //                 .add(TokenKind::RParen, 18, 19)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 19, 20)
    //                 .add(TokenKind::Assign, 20, 21)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 21, 22)
    //                 .add(TokenKind::Plus, 22, 23)
    //                 .add(TokenKind::Number(1), 23, 24)
    //                 .add(TokenKind::Semicolon, 24, 25)
    //                 .build(),
    //             raw_input: "for(i=0;i<10;i=i+1)x=x+1;",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::For,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::ForInit,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(8),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
    //                     ))),
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Less,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(8),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(10), None, None))),
    //                     ))),
    //                 ))),
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::ForUpdate,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(8),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::Add,
    //                             Some(Box::new(ASTNode::new(
    //                                 ASTNodeKind::LocalVariable(8),
    //                                 None,
    //                                 None,
    //                             ))),
    //                             Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                         ))),
    //                     ))),
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(16),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::Add,
    //                             Some(Box::new(ASTNode::new(
    //                                 ASTNodeKind::LocalVariable(16),
    //                                 None,
    //                                 None,
    //                             ))),
    //                             Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                         ))),
    //                     ))),
    //                 ))),
    //             )),
    //         },
    //         TestCase {
    //             name: "空のブロック文が正しくparseされること",
    //             token: TestTokenStream::new("{}")
    //                 .add(TokenKind::LBrace, 0, 1)
    //                 .add(TokenKind::RBrace, 1, 2)
    //                 .build(),
    //             raw_input: "{}",
    //             expected: Box::new(ASTNode::new(ASTNodeKind::Block, None, None)),
    //         },
    //         TestCase {
    //             name: "単一文のブロックが正しくparseされること",
    //             token: TestTokenStream::new("{x=1;}")
    //                 .add(TokenKind::LBrace, 0, 1)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 1, 2)
    //                 .add(TokenKind::Assign, 2, 3)
    //                 .add(TokenKind::Number(1), 3, 4)
    //                 .add(TokenKind::Semicolon, 4, 5)
    //                 .add(TokenKind::RBrace, 5, 6)
    //                 .build(),
    //             raw_input: "{x=1;}",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::Assign,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::LocalVariable(8),
    //                     None,
    //                     None,
    //                 ))),
    //                 Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //             )),
    //         },
    //         TestCase {
    //             name: "複数文のブロックが正しくparseされること",
    //             token: TestTokenStream::new("{x=1;y=2;z=3;}")
    //                 .add(TokenKind::LBrace, 0, 1)
    //                 .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 1, 2)
    //                 .add(TokenKind::Assign, 2, 3)
    //                 .add(TokenKind::Number(1), 3, 4)
    //                 .add(TokenKind::Semicolon, 4, 5)
    //                 .add(TokenKind::Identifier(LocalVariable::new("y", 0)), 5, 6)
    //                 .add(TokenKind::Assign, 6, 7)
    //                 .add(TokenKind::Number(2), 7, 8)
    //                 .add(TokenKind::Semicolon, 8, 9)
    //                 .add(TokenKind::Identifier(LocalVariable::new("z", 0)), 9, 10)
    //                 .add(TokenKind::Assign, 10, 11)
    //                 .add(TokenKind::Number(3), 11, 12)
    //                 .add(TokenKind::Semicolon, 12, 13)
    //                 .add(TokenKind::RBrace, 13, 14)
    //                 .build(),
    //             raw_input: "{x=1;y=2;z=3;}",
    //             expected: Box::new(ASTNode::new(
    //                 ASTNodeKind::Block,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Assign,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                 ))),
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Block,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(16),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
    //                     ))),
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::Assign,
    //                         Some(Box::new(ASTNode::new(
    //                             ASTNodeKind::LocalVariable(24),
    //                             None,
    //                             None,
    //                         ))),
    //                         Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
    //                     ))),
    //                 ))),
    //             )),
    //         },
    //     ];

    //     for case in test_cases {
    //         let mut scope = FunctionScope::new();
    //         scope.add_variable("i".to_string(), Type::new_int()); // オフセット 8
    //         scope.add_variable("x".to_string(), Type::new_int()); // オフセット 16
    //         scope.add_variable("y".to_string(), Type::new_int()); // オフセット 24
    //         scope.add_variable("z".to_string(), Type::new_int()); // オフセット 32
    //         let mut token = case.token;
    //         let result = stmt(&mut token, case.raw_input, &mut scope);
    //         assert_eq!(result, case.expected);
    //     }
    // }
    // #[test]
    // fn test_program() {
    //     struct ProgramTestCase<'a> {
    //         name: &'a str,
    //         token: Option<Box<Token<'a>>>,
    //         raw_input: &'a str,
    //         expected: Vec<Box<ASTNode>>,
    //     }

    //     let test_cases = vec![ProgramTestCase {
    //         name: "x = 1; y = 2; が正しくparseされること",
    //         token: TestTokenStream::new("main() { x=1; y=2; }")
    //             .add(TokenKind::Identifier(LocalVariable::new("main", 0)), 0, 4)
    //             .add(TokenKind::LParen, 4, 5)
    //             .add(TokenKind::RParen, 5, 6)
    //             .add(TokenKind::LBrace, 7, 8)
    //             .add(TokenKind::Identifier(LocalVariable::new("x", 0)), 9, 10)
    //             .add(TokenKind::Assign, 10, 11)
    //             .add(TokenKind::Number(1), 11, 12)
    //             .add(TokenKind::Semicolon, 12, 13)
    //             .add(TokenKind::Identifier(LocalVariable::new("y", 0)), 14, 15)
    //             .add(TokenKind::Assign, 15, 16)
    //             .add(TokenKind::Number(2), 16, 17)
    //             .add(TokenKind::Semicolon, 17, 18)
    //             .add(TokenKind::RBrace, 19, 20)
    //             .build(),
    //         raw_input: "main() { x=1; y=2; }",
    //         expected: vec![Box::new(ASTNode::new(
    //             ASTNodeKind::FunctionDef,
    //             Some(Box::new(ASTNode::new(
    //                 ASTNodeKind::FunctionName("main".to_string()),
    //                 None,
    //                 None,
    //             ))),
    //             Some(Box::new(ASTNode::new(
    //                 ASTNodeKind::Block,
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Assign,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(8),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
    //                 ))),
    //                 Some(Box::new(ASTNode::new(
    //                     ASTNodeKind::Assign,
    //                     Some(Box::new(ASTNode::new(
    //                         ASTNodeKind::LocalVariable(16),
    //                         None,
    //                         None,
    //                     ))),
    //                     Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
    //                 ))),
    //             ))),
    //         ))],
    // //     }];

    //     for case in test_cases {
    //         let mut token = case.token;
    //         let result = program(&mut token, case.raw_input);
    //         assert_eq!(result, case.expected, "{}", case.name);
    //     }
    // }
}
