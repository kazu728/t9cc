ASSEMBLY_FILES := $(shell find build -type f -name "*.s")
BINARIES := $(ASSEMBLY_FILES:.s=)

%: %.s
	@echo "Compiling $< -> $@"
	orb exec cc -o $@ $<

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
	cargo run "2*3+4*8"

.PHONY: check	
check:
	orb exec ./build/tmp
	@echo $?