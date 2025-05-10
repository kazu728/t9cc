use crate::parser::{ASTNodeKind, MaybeASTNode};

pub fn gen_asm(maybe_ast_node: MaybeASTNode) -> String {
    let mut output = String::new();

    output.push_str(".intel_syntax noprefix\n");
    output.push_str(".global main\n");
    output.push_str("main:\n");

    let mut output = gen_stack_insruction_asm(maybe_ast_node, &mut output);
    output.push_str("  pop rax\n");
    output.push_str("  ret\n");

    return output;
}

fn gen_stack_insruction_asm(maybe_ast_node: MaybeASTNode, output: &mut String) -> String {
    match maybe_ast_node {
        Some(ast_node) => {
            match ast_node.kind {
                ASTNodeKind::Num(n) => {
                    output.push_str(format!("  push {}\n", n).as_str());
                    return output.to_string();
                }
                _ => {
                    gen_stack_insruction_asm(ast_node.lhs, output);
                    gen_stack_insruction_asm(ast_node.rhs, output);
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

        None => output.to_string(),
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
            node: MaybeASTNode,
            expected: String,
        }

        let test_cases = vec![
            TestCase {
                name: "数値",
                node: Some(Box::new(ASTNode::new(ASTNodeKind::Num(42), None, None))),
                expected: "  push 42\n".to_string(),
            },
            TestCase {
                name: "加算",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Add,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  add rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "減算",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Sub,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(5), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(3), None, None))),
                ))),
                expected: "  push 5\n  push 3\n  pop rdi\n  pop rax\n  sub rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "乗算",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Mul,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(4), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(6), None, None))),
                ))),
                expected: "  push 4\n  push 6\n  pop rdi\n  pop rax\n  imul rax, rdi\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "除算",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Div,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(8), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
                expected:
                    "  push 8\n  push 2\n  pop rdi\n  pop rax\n  cqo\n  idiv rdi\n  push rax\n"
                        .to_string(),
            },
            TestCase {
                name: "複合式 (1+2)*(3-1)",
                node: Some(Box::new(ASTNode::new(
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
                ))),
                expected: String::from(
                    "  push 1\n  push 2\n  pop rdi\n  pop rax\n  add rax, rdi\n  push rax\n",
                )
                    + "  push 3\n  push 1\n  pop rdi\n  pop rax\n  sub rax, rdi\n  push rax\n"
                    + "  pop rdi\n  pop rax\n  imul rax, rdi\n  push rax\n",
            },
            TestCase {
                name: "等価演算子",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::Equal,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  cmp rax, rdi\n  sete al\n  movzb rax, al\n  push rax\n"
                    .to_string(),
            },
            TestCase {
                name: "不等価演算子",
                node: Some(Box::new(ASTNode::new(
                    ASTNodeKind::NotEqual,
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
                    Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
                ))),
                expected: "  push 1\n  push 2\n  pop rdi\n  pop rax\n  cmp rax, rdi\n  setne al\n  movzb rax, al\n  push rax\n"
                    .to_string(),
            },
            
        ];

        for case in test_cases {
            let result = gen_stack_insruction_asm(case.node, &mut String::new());
            assert_eq!(
                result, case.expected,
                "テストケース '{}' が失敗しました",
                case.name
            );
        }
    }

    #[test]
    fn test_gen_asm() {
        let input = Some(Box::new(ASTNode::new(
            ASTNodeKind::Add,
            Some(Box::new(ASTNode::new(ASTNodeKind::Num(1), None, None))),
            Some(Box::new(ASTNode::new(ASTNodeKind::Num(2), None, None))),
        )));

        let result = gen_asm(input);

        let expected = r#".intel_syntax noprefix
.global main
main:
  push 1
  push 2
  pop rdi
  pop rax
  add rax, rdi
  push rax
  pop rax
  ret
"#;

        assert_eq!(result, expected);
    }
}
