ASSEMBLY_FILES := $(shell find build -type f -name "*.s")
BINARIES := $(ASSEMBLY_FILES:.s=)

%: %.s
	@echo "Compiling $< -> $@"
	orb -m ubuntu exec gcc -o $@ $<

.PHONY: build	
build: generate_asm build_binaries
	
.PHONY: build_binaries
build_binaries: $(BINARIES)

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f $(BINARIES)

.PHONY: generate_asm
generate_asm:
	@echo "Generating assembly..."
	cargo run --bin t9cc "2*3+4*8;"

.PHONY: check	
check:
	orb -m ubuntu exec ./build/tmp
	@echo $?

.PHONY: e2e
e2e:
	@echo "e2eテストを実行中..."
	@mkdir -p build
	@cargo run --bin test_runner

.PHONY: test_case
test_case:
	@echo "テスト実行: $(INPUT)"
	@mkdir -p build
	@cargo run --bin t9cc "$(INPUT)"
	@orb -m ubuntu exec gcc -o build/test_tmp build/tmp.s
	@orb -m ubuntu exec ./build/test_tmp; echo "終了コード: $$?"