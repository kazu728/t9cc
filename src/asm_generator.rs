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

fn gen_stack_insruction_asm(ast_node: &ASTNode, output: &mut String) -> String {
    match ast_node.kind {
        ASTNodeKind::Num(n) => {
            output.push_str(format!("  push {}\n", n).as_str());
            return output.to_string();
        }
        ASTNodeKind::LocalVariable(_) => {
            gen_local_varibale(ast_node, output);
            output.push_str("  pop rax\n");
            output.push_str("  mov rax, [rax]\n");
            output.push_str("  push rax\n");
            return output.to_string();
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
            return output.to_string();
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
            output.push_str("  push 0\n");
            return output.to_string();
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
            output.push_str("  push 0\n");
            return output.to_string();
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
                output.push_str("  pop rax\n");
            }
            output.push_str(&format!("  jmp {}\n", end_label));
            output.push_str(&format!("{}:\n", else_label));

            if let Some(else_stmt) = &if_body.rhs {
                gen_stack_insruction_asm(else_stmt, output);
                output.push_str("  pop rax\n");
            }
            output.push_str(&format!("{}:\n", end_label));

            output.push_str("  push 0\n");
            return output.to_string();
        }
        ASTNodeKind::IfBody => {
            panic!("IfBody node should not be evaluated directly");
        }
        ASTNodeKind::ForInit | ASTNodeKind::ForUpdate => {
            panic!("ForInit and ForUpdate nodes should not be evaluated directly");
        }
        ASTNodeKind::Return => {
            if let Some(expr) = &ast_node.lhs {
                gen_stack_insruction_asm(expr, output);
                output.push_str("  pop rax\n");
                output.push_str("  mov rsp, rbp\n");
                output.push_str("  pop rbp\n");
                output.push_str("  ret\n");
            }

            return output.to_string();
        }

        // TODO: 2項演算子は別の脚に分ける
        _ => {
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
                    // idivは符号あり除算を行う命令
                    // idvはRDXとRAXを撮って、それを合わせたものを128ビット整数とみなしてそれを引数のレジスタの64ビットの値で割り、商をRAXに、余りをRDXにセットする
                    output.push_str("  idiv rdi\n");
                }
                _ => unimplemented!(),
            }
            output.push_str("  push rax\n");
            return output.to_string();
        }
    }
}

fn gen_local_varibale(ast_node: &ASTNode, output: &mut String) -> String {
    match ast_node.kind {
        ASTNodeKind::LocalVariable(offset) => {
            output.push_str("  mov rax, rbp\n");
            output.push_str(format!("  sub rax, {}\n", offset).as_str());
            output.push_str("  push rax\n");

            output.to_string()
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
                node: ASTNode::new(ASTNodeKind::Num(42), None, None),
                expected: "  push 42\n".to_string(),
            },
            TestCase {
                name: "加算",
                node: ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  add rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "減算",
                node: ASTNode::new(
                    ASTNodeKind::Sub,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(5), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                ),
                expected: "  push 5\n  push 3\n  pop rdi\n  pop rax\n  sub rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "乗算",
                node: ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(4), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(6), None, None))),
                ),
                expected: "  push 4\n  push 6\n  pop rdi\n  pop rax\n  imul rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "除算",
                node: ASTNode::new(
                    ASTNodeKind::Div,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(8), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected:
                    "  push 8\n  push 2\n  pop rdi\n  pop rax\n  cqo\n  idiv rdi\n  push rax\n"
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
                expected: String::from(
                    "  push 1\n  push 2\n  pop rdi\n  pop rax\n  add rax, rdi\n  push rax\n",
                )
                    + "  push 3\n  push 1\n  pop rdi\n  pop rax\n  sub rax, rdi\n  push rax\n"
                    + "  pop rdi\n  pop rax\n  imul rax, rdi\n  push rax\n",
            },
            TestCase {
                name: "等価演算子",
                node: ASTNode::new(
                    ASTNodeKind::Equal,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  cmp rax, rdi\n  sete al\n  movzb rax, al\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "不等価演算子",
                node: ASTNode::new(
                    ASTNodeKind::NotEqual,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  cmp rax, rdi\n  setne al\n  movzb rax, al\n  push rax\n"
                    .to_string(),
            },
            TestCase{
                name: "return文",
                node: ASTNode::new(
                    ASTNodeKind::Return,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                    None,
                ),
                expected: "  push 42\n  pop rax\n  mov rsp, rbp\n  pop rbp\n  ret\n".to_string(),
            },
            TestCase {
                name: "ローカル変数",
                node: ASTNode::new(
                    ASTNodeKind::LocalVariable(8),
                    None,
                    None,
                ),
                expected: "  mov rax, rbp\n  sub rax, 8\n  push rax\n  pop rax\n  mov rax, [rax]\n  push rax\n".to_string(),
            },
            TestCase {
                name: "ローカル変数の代入",
                node: ASTNode::new(
                    ASTNodeKind::Assign,
                    Some(Box::new(ASTNode::new(ASTNodeKind::LocalVariable(8), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                ),
                expected: "  mov rax, rbp\n  sub rax, 8\n  push rax\n  push 42\n  pop rdi\n  pop rax\n  mov [rax], rdi\n  push rdi\n"
                    .to_string(),
            },
            TestCase {
                name: "while文",
                node: ASTNode::new(
                    ASTNodeKind::While,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ),
                expected: ".Lbegin0:\n  push 1\n  pop rax\n  cmp rax, 0\n  je .Lend0\n  push 2\n  pop rax\n  jmp .Lbegin0\n.Lend0:\n  push 0\n"
                    .to_string(),
            },
        ];

        for case in test_cases {
            let result = gen_stack_insruction_asm(&case.node, &mut String::new());
            assert_eq!(
                result, case.expected,
                "テストケース '{}' が失敗しました",
                case.name
            );
        }
    }
}
