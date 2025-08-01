use crate::parser::{ASTNode, ASTNodeKind};

static mut LABEL_COUNTER: usize = 0;

fn get_label_number() -> usize {
    unsafe {
        let current = LABEL_COUNTER;
        LABEL_COUNTER += 1;
        current
    }
}

pub fn gen_asm(ast_node: &ASTNode, output: &mut String) -> String {
    gen_stack_insruction_asm(ast_node, output);
    output.push_str("  pop rax\n");
    return output.to_string();
}

fn gen_stack_insruction_asm(ast_node: &ASTNode, output: &mut String) {
    match ast_node.kind {
        ASTNodeKind::Num(n) => {
            output.push_str(format!("  push {}\n", n).as_str());
        }
        ASTNodeKind::LocalVariable(_) => {
            gen_local_varibale(ast_node, output);
            output.push_str("  pop rax\n");
            output.push_str("  mov rax, [rax]\n");
            output.push_str("  push rax\n");
        }
        ASTNodeKind::Assign => {
            if let Some(lhs) = &ast_node.lhs {
                gen_local_varibale(lhs, output);
            }
            if let Some(rhs) = &ast_node.rhs {
                gen_stack_insruction_asm(rhs, output);
            }

            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            output.push_str("  mov [rax], rdi\n");
            output.push_str("  push rdi\n");
        }
        ASTNodeKind::While => {
            let label_num = get_label_number();
            let begin_label = format!(".Lbegin{}", label_num);
            let end_label = format!(".Lend{}", label_num);

            output.push_str(&format!("{}:\n", begin_label));

            if let Some(cond) = &ast_node.lhs {
                gen_stack_insruction_asm(cond, output);
            }
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, 0\n");
            output.push_str(&format!("  je {}\n", end_label));

            if let Some(body) = &ast_node.rhs {
                gen_stack_insruction_asm(body, output);
                output.push_str("  pop rax\n");
            }

            output.push_str(&format!("  jmp {}\n", begin_label));
            output.push_str(&format!("{}:\n", end_label));
            // while文も値を返すため0をpush
            output.push_str("  push 0\n");
        }
        ASTNodeKind::For => {
            let label_num = get_label_number();
            let begin_label = format!(".Lbegin{}", label_num);
            let end_label = format!(".Lend{}", label_num);

            let (for_init, for_update) = match (&ast_node.lhs, &ast_node.rhs) {
                (Some(init_node), Some(update_node)) => (init_node, update_node),
                _ => panic!("Invalid for statement structure"),
            };

            if let Some(init) = &for_init.lhs {
                gen_stack_insruction_asm(init, output);
                output.push_str("  pop rax\n");
            }

            output.push_str(&format!("{}:\n", begin_label));

            if let Some(cond) = &for_init.rhs {
                gen_stack_insruction_asm(cond, output);
                output.push_str("  pop rax\n");
                output.push_str("  cmp rax, 0\n");
                output.push_str(&format!("  je {}\n", end_label));
            }

            if let Some(body) = &for_update.rhs {
                gen_stack_insruction_asm(body, output);
                output.push_str("  pop rax\n");
            }

            if let Some(update) = &for_update.lhs {
                gen_stack_insruction_asm(update, output);
                output.push_str("  pop rax\n");
            }

            output.push_str(&format!("  jmp {}\n", begin_label));
            output.push_str(&format!("{}:\n", end_label));
            // for文も値を返すため0をpush
            output.push_str("  push 0\n");
        }
        ASTNodeKind::Return => {
            if let Some(expr) = &ast_node.lhs {
                gen_stack_insruction_asm(expr, output);
                output.push_str("  pop rax\n");
                output.push_str("  mov rsp, rbp\n");
                output.push_str("  pop rbp\n");
                output.push_str("  ret\n");
            }
        }
        ASTNodeKind::Block => {
            if let Some(lhs) = &ast_node.lhs {
                gen_stack_insruction_asm(lhs, output);
                if ast_node.rhs.is_some() {
                    // rhsがある場合は、lhsの結果を破棄
                    output.push_str("  pop rax\n");
                }
            }
            if let Some(rhs) = &ast_node.rhs {
                gen_stack_insruction_asm(rhs, output);
            }
            // 空のブロックの場合は0を返す
            if ast_node.lhs.is_none() && ast_node.rhs.is_none() {
                output.push_str("  push 0\n");
            }
        }
        ASTNodeKind::If => {
            let label_num = get_label_number();
            let else_label = format!(".Lelse{}", label_num);
            let end_label = format!(".Lend{}", label_num);

            let if_body = match &ast_node.rhs {
                Some(body) => body,
                None => panic!("Invalid if statement structure"),
            };

            if let Some(cond) = &ast_node.lhs {
                gen_stack_insruction_asm(cond, output);
            }
            output.push_str("  pop rax\n");
            output.push_str("  cmp rax, 0\n");
            output.push_str(&format!("  je {}\n", else_label));

            if let Some(then_stmt) = &if_body.lhs {
                gen_stack_insruction_asm(then_stmt, output);
            }
            output.push_str(&format!("  jmp {}\n", end_label));
            output.push_str(&format!("{}:\n", else_label));

            if let Some(else_stmt) = &if_body.rhs {
                gen_stack_insruction_asm(else_stmt, output);
            } else {
                output.push_str("  push 0\n"); // else節がない場合は0をpush
            }
            output.push_str(&format!("{}:\n", end_label));
        }

        ASTNodeKind::Add
        | ASTNodeKind::Sub
        | ASTNodeKind::Mul
        | ASTNodeKind::Div
        | ASTNodeKind::Equal
        | ASTNodeKind::NotEqual
        | ASTNodeKind::Less
        | ASTNodeKind::LessEqual
        | ASTNodeKind::Greater
        | ASTNodeKind::GreaterEqual => {
            if let Some(lhs) = &ast_node.lhs {
                gen_stack_insruction_asm(lhs, output);
            }
            if let Some(rhs) = &ast_node.rhs {
                gen_stack_insruction_asm(rhs, output);
            }
            output.push_str("  pop rdi\n");
            output.push_str("  pop rax\n");
            match ast_node.kind {
                ASTNodeKind::Add => output.push_str("  add rax, rdi\n"),
                ASTNodeKind::Sub => output.push_str("  sub rax, rdi\n"),
                ASTNodeKind::Mul => output.push_str("  imul rax, rdi\n"),
                ASTNodeKind::Equal => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  sete al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::NotEqual => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  setne al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::Less => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  setl al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::LessEqual => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  setle al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::Greater => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  setg al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::GreaterEqual => {
                    output.push_str("  cmp rax, rdi\n");
                    output.push_str("  setge al\n");
                    output.push_str("  movzb rax, al\n");
                }
                ASTNodeKind::Div => {
                    output.push_str("  cqo\n");
                    output.push_str("  idiv rdi\n");
                }
                _ => unreachable!(),
            }
            output.push_str("  push rax\n");
        }
        _ => unreachable!("Unhandled node kind: {:?}", ast_node.kind),
    }
}

fn gen_local_varibale(ast_node: &ASTNode, output: &mut String) {
    match ast_node.kind {
        ASTNodeKind::LocalVariable(offset) => {
            output.push_str("  mov rax, rbp\n");
            output.push_str(format!("  sub rax, {}\n", offset).as_str());
            output.push_str("  push rax\n");
        }
        _ => {
            panic!("左辺値ではありません")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{ASTNode, ASTNodeKind};

    fn reset_label_counter() {
        unsafe {
            LABEL_COUNTER = 0;
        }
    }

    #[test]
    fn test_gen_stack_insruction_asm() {
        struct TestCase {
            name: &'static str,
            node: ASTNode,
            expected: String,
        }

        let test_cases = vec![
            TestCase {
                name: "数値",
                node: ASTNode::new(ASTNodeKind::Num(42), None, None),
                expected: "  push 42
"
                .to_string(),
            },
            TestCase {
                name: "加算",
                node: ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1
  push 2
  pop rdi
  pop rax
  add rax, rdi
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "減算",
                node: ASTNode::new(
                    ASTNodeKind::Sub,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(5), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                ),
                expected: "  push 5
  push 3
  pop rdi
  pop rax
  sub rax, rdi
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "乗算",
                node: ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(4), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(6), None, None))),
                ),
                expected: "  push 4
  push 6
  pop rdi
  pop rax
  imul rax, rdi
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "除算",
                node: ASTNode::new(
                    ASTNodeKind::Div,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(8), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 8
  push 2
  pop rdi
  pop rax
  cqo
  idiv rdi
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "複合式 (1+2)*(3-1)",
                node: ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Add,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Sub,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    ))),
                ),
                expected: "  push 1
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
"
                .to_string(),
            },
            TestCase {
                name: "等価演算子",
                node: ASTNode::new(
                    ASTNodeKind::Equal,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1
  push 2
  pop rdi
  pop rax
  cmp rax, rdi
  sete al
  movzb rax, al
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "不等価演算子",
                node: ASTNode::new(
                    ASTNodeKind::NotEqual,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1
  push 2
  pop rdi
  pop rax
  cmp rax, rdi
  setne al
  movzb rax, al
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "ローカル変数",
                node: ASTNode::new(ASTNodeKind::LocalVariable(8), None, None),
                expected: "  mov rax, rbp
  sub rax, 8
  push rax
  pop rax
  mov rax, [rax]
  push rax
"
                .to_string(),
            },
            TestCase {
                name: "ローカル変数の代入",
                node: ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::LocalVariable(8),
                        None,
                        None,
                    ))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                ),
                expected: "  mov rax, rbp
  sub rax, 8
  push rax
  push 42
  pop rdi
  pop rax
  mov [rax], rdi
  push rdi
"
                .to_string(),
            },
        ];

        for case in test_cases {
            reset_label_counter();
            let mut output = String::new();
            gen_stack_insruction_asm(&case.node, &mut output);
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
                node: ASTNode::new(ASTNodeKind::Block, None, None),
                expected: "  push 0\n".to_string(),
            },
            TestCase {
                name: "単一文のブロック",
                node: ASTNode::new(
                    ASTNodeKind::Block,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                    None,
                ),
                expected: "  push 42
"
                .to_string(),
            },
            TestCase {
                name: "複数文のブロック {1; 2; 3;}",
                node: ASTNode::new(
                    ASTNodeKind::Block,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::Block,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                    ))),
                ),
                expected: "  push 1
  pop rax
  push 2
  pop rax
  push 3
"
                .to_string(),
            },
            TestCase {
                name: "while文",
                node: ASTNode::new(
                    ASTNodeKind::While,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: ".Lbegin0:
  push 1
  pop rax
  cmp rax, 0
  je .Lend0
  push 2
  pop rax
  jmp .Lbegin0
.Lend0:
  push 0
"
                .to_string(),
            },
            TestCase {
                name: "if文（else節なし）",
                node: ASTNode::new(
                    ASTNodeKind::If,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::IfBody,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                        None,
                    ))),
                ),
                expected: "  push 1
  pop rax
  cmp rax, 0
  je .Lelse0
  push 42
  jmp .Lend0
.Lelse0:
  push 0
.Lend0:
"
                .to_string(),
            },
            TestCase {
                name: "return文",
                node: ASTNode::new(
                    ASTNodeKind::Return,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                    None,
                ),
                expected: "  push 42
  pop rax
  mov rsp, rbp
  pop rbp
  ret
"
                .to_string(),
            },
            TestCase {
                name: "if文（else節あり）",
                node: ASTNode::new(
                    ASTNodeKind::If,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::IfBody,
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(24), None, None))),
                    ))),
                ),
                expected: "  push 0
  pop rax
  cmp rax, 0
  je .Lelse0
  push 42
  jmp .Lend0
.Lelse0:
  push 24
.Lend0:
"
                .to_string(),
            },
            TestCase {
                name: "for文",
                node: ASTNode::new(
                    ASTNodeKind::For,
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::ForInit,
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Assign,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(0), None, None))),
                        ))),
                        Some(Box::new(ASTNode::new(
                            ASTNodeKind::Less,
                            Some(Box::new(ASTNode::new(
                                ASTNodeKind::LocalVariable(8),
                                None,
                                None,
                            ))),
                            Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                        ))),
                    ))),
                    Some(Box::new(ASTNode::new(
                        ASTNodeKind::ForUpdate,
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
                        Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                    ))),
                ),
                expected: "  mov rax, rbp
  sub rax, 8
  push rax
  push 0
  pop rdi
  pop rax
  mov [rax], rdi
  push rdi
  pop rax
.Lbegin0:
  mov rax, rbp
  sub rax, 8
  push rax
  pop rax
  mov rax, [rax]
  push rax
  push 3
  pop rdi
  pop rax
  cmp rax, rdi
  setl al
  movzb rax, al
  push rax
  pop rax
  cmp rax, 0
  je .Lend0
  push 42
  pop rax
  mov rax, rbp
  sub rax, 8
  push rax
  mov rax, rbp
  sub rax, 8
  push rax
  pop rax
  mov rax, [rax]
  push rax
  push 1
  pop rdi
  pop rax
  add rax, rdi
  push rax
  pop rdi
  pop rax
  mov [rax], rdi
  push rdi
  pop rax
  jmp .Lbegin0
.Lend0:
  push 0
"
                .to_string(),
            },
        ];

        for case in test_cases {
            reset_label_counter();
            let mut output = String::new();
            gen_stack_insruction_asm(&case.node, &mut output);
            assert_eq!(
                output, case.expected,
                "テストケース '{}' が失敗しました",
                case.name
            );
        }
    }
}
