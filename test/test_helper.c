#include <stdio.h>

int foo() {
    printf("OK\n");
    return 42;
}

int add(int x, int y) {
    printf("%d\n", x + y);
    return x + y;
}

int mul3(int x, int y, int z) {
    return x * y * z;
}