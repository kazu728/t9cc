use crate::ast::{ASTNode, Function, Program};
use crate::types::TypeKind;
use std::sync::atomic::{AtomicUsize, Ordering};

const REGISTERS: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];
const MAX_REGISTER_ARGS: usize = 6;
const STACK_SIZE: usize = 208;

static LABEL_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn get_label_number() -> usize {
    LABEL_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub fn gen_program_asm(program: &Program, output: &mut String) {
    output.push_str(".intel_syntax noprefix\n");
    output.push_str(".global main\n\n");

    if !program.string_literals.is_empty() {
        output.push_str(".section .rodata\n");
        for (i, string) in program.string_literals.iter().enumerate() {
            output.push_str(&format!(".LC{}:\n", i));
            output.push_str(&format!("  .string \"{}\"\n", string));
        }
        output.push('\n');
    }

    if !program.global_vars.is_empty() {
        output.push_str(".bss\n");
        for (name, var_type) in &program.global_vars {
            output.push_str(&format!("{}:\n", name));
            output.push_str(&format!("  .zero {}\n", var_type.sizeof()));
        }
        output.push('\n');
    }

    output.push_str(".text\n");

    for function in &program.functions {
        gen_function_asm(function, output);
    }
}

fn gen_function_asm(function: &Function, output: &mut String) {
    output.push_str(&format!("{}:\n", function.name));

    output.push_str("  push rbp\n");
    output.push_str("  mov rbp, rsp\n");
    output.push_str(&format!("  sub rsp, {}\n", STACK_SIZE));

    for (i, _param) in function.params.iter().enumerate() {
        if i < MAX_REGISTER_ARGS {
            let offset = (i + 1) * 8;
            output.push_str(&format!("  mov [rbp-{}], {}\n", offset, REGISTERS[i]));
        }
    }

    gen_statement_asm(&function.body, output);

    output.push_str("  mov rax, 0\n");
    output.push_str("  mov rsp, rbp\n");
    output.push_str("  pop rbp\n");
    output.push_str("  ret\n\n");
}

fn gen_statement_asm(stmt: &ASTNode, output: &mut String) {
    match stmt {
        ASTNode::Block { statements } => {
            for statement in statements {
                gen_statement_asm(statement, output);
            }
        }
        ASTNode::ExpressionStatement { expr } => {
            gen_stack_instruction_asm(expr, output);
            output.push_str("  pop rax\n");
        }
        ASTNode::Return { expr } => {
            if let Some(expr) = expr {
                gen_stack_instruction_asm(expr, output);
                output.push_str("  pop rax\n");
            } else {
                output.push_str("  mov rax, 0\n");
            }
            output.push_str("  mov rsp, rbp\n");
            output.push_str("  pop rbp\n");
            output.push_str("  ret\n");
        }
        ASTNode::If {
            condition,
            then_stmt,
            else_stmt,
        } => {
            let label_num = get_label_number();
            gen_stack_instruction_asm(condition, output);
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, 0\n");

            if let Some(else_stmt) = else_stmt {
                output.push_str(&format!("  je .Lelse{}\n", label_num));
                gen_statement_asm(then_stmt, output);
                output.push_str(&format!("  jmp .Lend{}\n", label_num));
                output.push_str(&format!(".Lelse{}:\n", label_num));
                gen_statement_asm(else_stmt, output);
                output.push_str(&format!(".Lend{}:\n", label_num));
            } else {
                output.push_str(&format!("  je .Lend{}\n", label_num));
                gen_statement_asm(then_stmt, output);
                output.push_str(&format!(".Lend{}:\n", label_num));
            }
        }
        ASTNode::While { condition, body } => {
            let label_num = get_label_number();
            output.push_str(&format!(".Lbegin{}:\n", label_num));
            gen_stack_instruction_asm(condition, output);
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, 0\n");
            output.push_str(&format!("  je .Lend{}\n", label_num));
            gen_statement_asm(body, output);
            output.push_str(&format!("  jmp .Lbegin{}\n", label_num));
            output.push_str(&format!(".Lend{}:\n", label_num));
        }
        ASTNode::For {
            init,
            condition,
            update,
            body,
        } => {
            let label_num = get_label_number();

            if let Some(init) = init {
                gen_stack_instruction_asm(init, output);
                output.push_str("  pop rax\n");
            }

            output.push_str(&format!(".Lbegin{}:\n", label_num));

            if let Some(condition) = condition {
                gen_stack_instruction_asm(condition, output);
                output.push_str("  pop rax\n");
                output.push_str("  cmp rax, 0\n");
                output.push_str(&format!("  je .Lend{}\n", label_num));
            }

            gen_statement_asm(body, output);

            if let Some(update) = update {
                gen_stack_instruction_asm(update, output);
                output.push_str("  pop rax\n");
            }

            output.push_str(&format!("  jmp .Lbegin{}\n", label_num));
            output.push_str(&format!(".Lend{}:\n", label_num));
        }
        _ => {
            gen_stack_instruction_asm(stmt, output);
            output.push_str("  pop rax\n");
        }
    }
}

fn gen_stack_instruction_asm(ast_node: &ASTNode, output: &mut String) {
    match ast_node {
        ASTNode::Num { value, .. } => {
            output.push_str(&format!("  push {}\n", value));
        }
        ASTNode::LocalVariable { node_type, .. } => {
            gen_lvalue_asm(ast_node, output);
            output.push_str("  pop rax\n");

            match node_type.kind {
                TypeKind::Char => output.push_str("  movsx rax, BYTE PTR [rax]\n"),
                _ => output.push_str("  mov rax, [rax]\n"),
            }
            output.push_str("  push rax\n");
        }
        ASTNode::GlobalVariable { node_type, .. } => {
            gen_lvalue_asm(ast_node, output);
            output.push_str("  pop rax\n");

            match node_type.kind {
                TypeKind::Char => output.push_str("  movsx rax, BYTE PTR [rax]\n"),
                _ => output.push_str("  mov rax, [rax]\n"),
            }
            output.push_str("  push rax\n");
        }
        ASTNode::StringLiteral { id, .. } => {
            output.push_str(&format!("  lea rax, [rip + .LC{}]\n", id));
            output.push_str("  push rax\n");
        }
        ASTNode::Add { lhs, rhs, .. } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  add rax, rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Sub { lhs, rhs, .. } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  sub rax, rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Mul { lhs, rhs, .. } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  imul rax, rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Div { lhs, rhs, .. } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cqo\n");
            output.push_str("  idiv rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Equal { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  sete al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::NotEqual { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  setne al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Less { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  setl al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::LessEqual { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  setle al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Greater { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  setg al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::GreaterEqual { lhs, rhs } => {
            gen_stack_instruction_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, rdi\n");
            output.push_str("  setge al\n");
            output.push_str("  movzb rax, al\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Assign {
            lhs,
            rhs,
            node_type,
        } => {
            gen_lvalue_asm(lhs, output);
            gen_stack_instruction_asm(rhs, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");

            match node_type.kind {
                TypeKind::Char => output.push_str("  mov [rax], dil\n"),
                _ => output.push_str("  mov [rax], rdi\n"),
            }
            output.push_str("  push rdi\n");
        }
        ASTNode::PtrAdd { ptr, offset, .. } => {
            gen_stack_instruction_asm(ptr, output);
            gen_stack_instruction_asm(offset, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");

            if let Some(ptr_type) = ptr.get_node_type() {
                let size = ptr_type.get_pointed_type().sizeof();
                output.push_str(&format!("  imul rdi, {}\n", size));
            }

            output.push_str("  add rax, rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::PtrSub { ptr, offset, .. } => {
            gen_stack_instruction_asm(ptr, output);
            gen_stack_instruction_asm(offset, output);
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");

            if let Some(ptr_type) = ptr.get_node_type() {
                let size = ptr_type.get_pointed_type().sizeof();
                output.push_str(&format!("  imul rdi, {}\n", size));
            }

            output.push_str("  sub rax, rdi\n");
            output.push_str("  push rax\n");
        }
        ASTNode::Deref { operand, node_type } => {
            gen_stack_instruction_asm(operand, output);
            output.push_str("  pop rax\n");

            match node_type.kind {
                TypeKind::Char => output.push_str("  movsx rax, BYTE PTR [rax]\n"),
                _ => output.push_str("  mov rax, [rax]\n"),
            }
            output.push_str("  push rax\n");
        }
        ASTNode::Addr { operand, .. } => {
            gen_lvalue_asm(operand, output);
        }
        ASTNode::FunctionCall { name, args, .. } => {
            for arg in args.iter().rev() {
                gen_stack_instruction_asm(arg, output);
            }

            for i in 0..args.len().min(MAX_REGISTER_ARGS) {
                output.push_str(&format!("  pop {}\n", REGISTERS[i]));
            }

            let align_rsp = (16 - (args.len() % 2) * 8) % 16;
            if align_rsp != 0 {
                output.push_str(&format!("  sub rsp, {}\n", align_rsp));
            }

            output.push_str(&format!("  call {}\n", name));

            if align_rsp != 0 {
                output.push_str(&format!("  add rsp, {}\n", align_rsp));
            }

            output.push_str("  push rax\n");
        }
        _ => {
            panic!(
                "Unhandled AST node in gen_stack_instruction_asm: {:?}",
                ast_node
            );
        }
    }
}

fn gen_lvalue_asm(ast_node: &ASTNode, output: &mut String) {
    match ast_node {
        ASTNode::LocalVariable { offset, .. } => {
            output.push_str(&format!("  lea rax, [rbp-{}]\n", offset));
            output.push_str("  push rax\n");
        }
        ASTNode::GlobalVariable { name, .. } => {
            output.push_str(&format!("  lea rax, [rip + {}]\n", name));
            output.push_str("  push rax\n");
        }
        ASTNode::Deref { operand, .. } => {
            gen_stack_instruction_asm(operand, output);
        }
        _ => {
            panic!("Cannot generate lvalue for this node type");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Type;
    use std::sync::atomic::Ordering;

    fn reset_label_counter() {
        LABEL_COUNTER.store(0, Ordering::Relaxed);
    }

    #[test]
    fn test_gen_stack_instruction_asm() {
        struct TestCase {
            name: &'static str,
            node: ASTNode,
            expected: String,
        }

        let test_cases = vec![
            TestCase {
                name: "数値",
                node: ASTNode::Num {
                    value: 42,
                    node_type: Type::new_int(),
                },
                expected: r#"  push 42
"#
                .to_string(),
            },
            TestCase {
                name: "加算",
                node: ASTNode::Add {
                    lhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  push 1
  push 2
  pop rdi
  pop rax
  add rax, rdi
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "減算",
                node: ASTNode::Sub {
                    lhs: Box::new(ASTNode::Num {
                        value: 5,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 3,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  push 5
  push 3
  pop rdi
  pop rax
  sub rax, rdi
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "乗算",
                node: ASTNode::Mul {
                    lhs: Box::new(ASTNode::Num {
                        value: 4,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 6,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  push 4
  push 6
  pop rdi
  pop rax
  imul rax, rdi
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "除算",
                node: ASTNode::Div {
                    lhs: Box::new(ASTNode::Num {
                        value: 8,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  push 8
  push 2
  pop rdi
  pop rax
  cqo
  idiv rdi
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "複合式 (1+2)*(3-1)",
                node: ASTNode::Mul {
                    lhs: Box::new(ASTNode::Add {
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
                    rhs: Box::new(ASTNode::Sub {
                        lhs: Box::new(ASTNode::Num {
                            value: 3,
                            node_type: Type::new_int(),
                        }),
                        rhs: Box::new(ASTNode::Num {
                            value: 1,
                            node_type: Type::new_int(),
                        }),
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  push 1
  push 2
  pop rdi
  pop rax
  add rax, rdi
  push rax
  push 3
  push 1
  pop rdi
  pop rax
  sub rax, rdi
  push rax
  pop rdi
  pop rax
  imul rax, rdi
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "等価演算子",
                node: ASTNode::Equal {
                    lhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                },
                expected: r#"  push 1
  push 2
  pop rdi
  pop rax
  cmp rax, rdi
  sete al
  movzb rax, al
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "不等価演算子",
                node: ASTNode::NotEqual {
                    lhs: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 2,
                        node_type: Type::new_int(),
                    }),
                },
                expected: r#"  push 1
  push 2
  pop rdi
  pop rax
  cmp rax, rdi
  setne al
  movzb rax, al
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "ローカル変数",
                node: ASTNode::LocalVariable {
                    offset: 8,
                    node_type: Type::new_int(),
                },
                expected: r#"  lea rax, [rbp-8]
  push rax
  pop rax
  mov rax, [rax]
  push rax
"#
                .to_string(),
            },
            TestCase {
                name: "ローカル変数の代入",
                node: ASTNode::Assign {
                    lhs: Box::new(ASTNode::LocalVariable {
                        offset: 8,
                        node_type: Type::new_int(),
                    }),
                    rhs: Box::new(ASTNode::Num {
                        value: 42,
                        node_type: Type::new_int(),
                    }),
                    node_type: Type::new_int(),
                },
                expected: r#"  lea rax, [rbp-8]
  push rax
  push 42
  pop rdi
  pop rax
  mov [rax], rdi
  push rdi
"#
                .to_string(),
            },
        ];

        for case in test_cases {
            reset_label_counter();
            let mut output = String::new();
            gen_stack_instruction_asm(&case.node, &mut output);
            assert_eq!(
                output, case.expected,
                "テストケース '{}' が失敗しました",
                case.name
            );
        }
    }

    #[test]
    fn test_block_statements() {
        struct TestCase {
            name: &'static str,
            node: ASTNode,
            expected: String,
        }

        let test_cases = vec![
            TestCase {
                name: "空のブロック",
                node: ASTNode::Block { statements: vec![] },
                expected: r#""#.to_string(),
            },
            TestCase {
                name: "単一文のブロック",
                node: ASTNode::Block {
                    statements: vec![ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Num {
                            value: 42,
                            node_type: Type::new_int(),
                        }),
                    }],
                },
                expected: r#"  push 42
  pop rax
"#
                .to_string(),
            },
            TestCase {
                name: "複数文のブロック {1; 2; 3;}",
                node: ASTNode::Block {
                    statements: vec![
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Num {
                                value: 1,
                                node_type: Type::new_int(),
                            }),
                        },
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Num {
                                value: 2,
                                node_type: Type::new_int(),
                            }),
                        },
                        ASTNode::ExpressionStatement {
                            expr: Box::new(ASTNode::Num {
                                value: 3,
                                node_type: Type::new_int(),
                            }),
                        },
                    ],
                },
                expected: r#"  push 1
  pop rax
  push 2
  pop rax
  push 3
  pop rax
"#
                .to_string(),
            },
            TestCase {
                name: "while文",
                node: ASTNode::While {
                    condition: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    body: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Num {
                            value: 2,
                            node_type: Type::new_int(),
                        }),
                    }),
                },
                expected: r#".Lbegin0:
  push 1
  pop rax
  cmp rax, 0
  je .Lend0
  push 2
  pop rax
  jmp .Lbegin0
.Lend0:
"#
                .to_string(),
            },
            TestCase {
                name: "if文（else節なし）",
                node: ASTNode::If {
                    condition: Box::new(ASTNode::Num {
                        value: 1,
                        node_type: Type::new_int(),
                    }),
                    then_stmt: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Num {
                            value: 42,
                            node_type: Type::new_int(),
                        }),
                    }),
                    else_stmt: None,
                },
                expected: r#"  push 1
  pop rax
  cmp rax, 0
  je .Lend0
  push 42
  pop rax
.Lend0:
"#
                .to_string(),
            },
            TestCase {
                name: "return文",
                node: ASTNode::Return {
                    expr: Some(Box::new(ASTNode::Num {
                        value: 42,
                        node_type: Type::new_int(),
                    })),
                },
                expected: r#"  push 42
  pop rax
  mov rsp, rbp
  pop rbp
  ret
"#
                .to_string(),
            },
            TestCase {
                name: "if文（else節あり）",
                node: ASTNode::If {
                    condition: Box::new(ASTNode::Num {
                        value: 0,
                        node_type: Type::new_int(),
                    }),
                    then_stmt: Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Num {
                            value: 42,
                            node_type: Type::new_int(),
                        }),
                    }),
                    else_stmt: Some(Box::new(ASTNode::ExpressionStatement {
                        expr: Box::new(ASTNode::Num {
                            value: 24,
                            node_type: Type::new_int(),
                        }),
                    })),
                },
                expected: r#"  push 0
  pop rax
  cmp rax, 0
  je .Lelse0
  push 42
  pop rax
  jmp .Lend0
.Lelse0:
  push 24
  pop rax
.Lend0:
"#
                .to_string(),
            },
        ];

        for case in test_cases {
            reset_label_counter();
            let mut output = String::new();
            gen_statement_asm(&case.node, &mut output);
            assert_eq!(
                output, case.expected,
                "テストケース '{}' が失敗しました",
                case.name
            );
        }
    }

    #[test]
    fn test_gen_string_literal() {
        reset_label_counter();
        let mut output = String::new();

        let string_node = ASTNode::StringLiteral {
            id: 0,
            node_type: Type::new_ptr(Type::new_char()),
        };

        gen_stack_instruction_asm(&string_node, &mut output);

        let expected = r#"  lea rax, [rip + .LC0]
  push rax
"#;
        assert_eq!(output, expected);
    }

    #[test]
    fn test_gen_function_call() {
        reset_label_counter();
        let mut output = String::new();

        let args = vec![
            ASTNode::Num {
                value: 10,
                node_type: Type::new_int(),
            },
            ASTNode::Num {
                value: 20,
                node_type: Type::new_int(),
            },
        ];

        let func_call = ASTNode::FunctionCall {
            name: "test_func".to_string(),
            args,
            node_type: Type::new_int(),
        };

        gen_stack_instruction_asm(&func_call, &mut output);

        let expected = r#"  push 20
  push 10
  pop rdi
  pop rsi
  call test_func
  push rax
"#;
        assert_eq!(output, expected);
    }

    #[test]
    fn test_gen_pointer_operations() {
        reset_label_counter();
        let mut output = String::new();

        let ptr = Box::new(ASTNode::LocalVariable {
            offset: 8,
            node_type: Type::new_ptr(Type::new_int()),
        });

        let offset = Box::new(ASTNode::Num {
            value: 2,
            node_type: Type::new_int(),
        });

        let ptr_add = ASTNode::PtrAdd {
            ptr,
            offset,
            node_type: Type::new_ptr(Type::new_int()),
        };

        gen_stack_instruction_asm(&ptr_add, &mut output);

        let expected = r#"  lea rax, [rbp-8]
  push rax
  pop rax
  mov rax, [rax]
  push rax
  push 2
  pop rdi
  pop rax
  imul rdi, 4
  add rax, rdi
  push rax
"#;
        assert_eq!(output, expected);
    }

    #[test]
    fn test_gen_program() {
        reset_label_counter();
        let mut output = String::new();

        let body = ASTNode::Return {
            expr: Some(Box::new(ASTNode::Num {
                value: 42,
                node_type: Type::new_int(),
            })),
        };

        let main_func = Function {
            name: "main".to_string(),
            body,
            params: vec![],
        };

        let mut global_vars = std::collections::BTreeMap::new();
        global_vars.insert("global_x".to_string(), Type::new_int());

        let string_literals = vec!["hello".to_string()];

        let program = Program {
            functions: vec![main_func],
            global_vars,
            string_literals,
        };

        gen_program_asm(&program, &mut output);

        let expected = r#".intel_syntax noprefix
.global main

.section .rodata
.LC0:
  .string "hello"

.bss
global_x:
  .zero 4

.text
main:
  push rbp
  mov rbp, rsp
  sub rsp, 208
  push 42
  pop rax
  mov rsp, rbp
  pop rbp
  ret
  mov rax, 0
  mov rsp, rbp
  pop rbp
  ret

"#;
        assert_eq!(output, expected);
    }
}
