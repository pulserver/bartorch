/*
 * Declarations of what src/csrc/substitute/posix.c provides, included into
 * every translation unit on Windows: GCC 14 refuses a call to an undeclared
 * function, and MinGW's headers declare neither.
 */
#ifndef BARTORCH_SUBSTITUTE_POSIX_H
#define BARTORCH_SUBSTITUTE_POSIX_H

#include <stddef.h>
#include <sys/types.h>

extern int getsubopt(char** optionp, char* const* tokens, char** valuep);
extern ssize_t bartorch_no_readlink(const char* path, char* buf, size_t size);

#endif
