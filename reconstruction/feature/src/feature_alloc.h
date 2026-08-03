#ifndef ROCKIT_FEATURE_ALLOC_H
#define ROCKIT_FEATURE_ALLOC_H

#include <stdlib.h>
#include <string.h>

static inline int rockit_feature_allocation_failure_enabled(void) {
    const char *value = getenv("ROCKIT_FEATURE_FAIL_ALLOC");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static inline void *rockit_feature_calloc(size_t count, size_t size) {
    if (rockit_feature_allocation_failure_enabled()) return NULL;
    return calloc(count, size);
}

#endif
