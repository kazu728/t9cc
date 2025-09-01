use crate::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum ASTNode {
    Num {
        value: i32,
        node_type: Type,
    },
    LocalVariable {
        offset: u32,
        node_type: Type,
    },
    GlobalVariable {
        name: String,
        node_type: Type,
    },
    StringLiteral {
        id: usize,
        node_type: Type,
    },
    Add {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        node_type: Type,
    },
    Sub {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        node_type: Type,
    },
    Mul {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        node_type: Type,
    },
    Div {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        node_type: Type,
    },
    Equal {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    NotEqual {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    Less {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    LessEqual {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    Greater {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    GreaterEqual {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    Assign {
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
        node_type: Type,
    },
    PtrAdd {
        ptr: Box<ASTNode>,
        offset: Box<ASTNode>,
        node_type: Type,
    },
    PtrSub {
        ptr: Box<ASTNode>,
        offset: Box<ASTNode>,
        node_type: Type,
    },
    Deref {
        operand: Box<ASTNode>,
        node_type: Type,
    },
    Addr {
        operand: Box<ASTNode>,
        node_type: Type,
    },
    FunctionCall {
        name: String,
        args: Vec<ASTNode>,
        node_type: Type,
    },
    Return {
        expr: Option<Box<ASTNode>>,
    },
    If {
        condition: Box<ASTNode>,
        then_stmt: Box<ASTNode>,
        else_stmt: Option<Box<ASTNode>>,
    },
    While {
        condition: Box<ASTNode>,
        body: Box<ASTNode>,
    },
    For {
        init: Option<Box<ASTNode>>,
        condition: Option<Box<ASTNode>>,
        update: Option<Box<ASTNode>>,
        body: Box<ASTNode>,
    },
    Block {
        statements: Vec<ASTNode>,
    },
    ExpressionStatement {
        expr: Box<ASTNode>,
    },
}

impl ASTNode {
    pub fn get_node_type(&self) -> Option<&Type> {
        match self {
            Self::Num { node_type, .. }
            | Self::LocalVariable { node_type, .. }
            | Self::GlobalVariable { node_type, .. }
            | Self::StringLiteral { node_type, .. }
            | Self::Add { node_type, .. }
            | Self::Sub { node_type, .. }
            | Self::Mul { node_type, .. }
            | Self::Div { node_type, .. }
            | Self::Assign { node_type, .. }
            | Self::PtrAdd { node_type, .. }
            | Self::PtrSub { node_type, .. }
            | Self::Deref { node_type, .. }
            | Self::Addr { node_type, .. }
            | Self::FunctionCall { node_type, .. } => Some(node_type),
            _ => None,
        }
    }

    pub fn get_node_type_mut(&mut self) -> Option<&mut Type> {
        match self {
            Self::Num { node_type, .. }
            | Self::LocalVariable { node_type, .. }
            | Self::GlobalVariable { node_type, .. }
            | Self::StringLiteral { node_type, .. }
            | Self::Add { node_type, .. }
            | Self::Sub { node_type, .. }
            | Self::Mul { node_type, .. }
            | Self::Div { node_type, .. }
            | Self::Assign { node_type, .. }
            | Self::PtrAdd { node_type, .. }
            | Self::PtrSub { node_type, .. }
            | Self::Deref { node_type, .. }
            | Self::Addr { node_type, .. }
            | Self::FunctionCall { node_type, .. } => Some(node_type),
            _ => None,
        }
    }

    pub fn decay_array_type(&mut self) {
        if let Some(node_type) = self.get_node_type_mut() {
            if node_type.is_array() {
                *node_type = node_type.decay_array_to_pointer();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub body: ASTNode,
    pub params: Vec<String>,
}

impl Function {
    pub fn new(name: String, params: Vec<String>, body: ASTNode) -> Self {
        Self { name, params, body }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Program {
    pub functions: Vec<Function>,
    pub global_vars: std::collections::BTreeMap<String, Type>,
    pub string_literals: Vec<String>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }
}
