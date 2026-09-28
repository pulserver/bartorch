/*
 * The POSIX functions BART calls that MinGW's C library does not provide.
 *
 * Compiled on Windows only.  getsubopt is POSIX's, since BART parses the
 * sub-options of flags such as --nufft-conf with it.  readlink is asked only
 * for /proc/self/exe, to find a directory of external commands beside the
 * `bart` executable; there is no such file on Windows and no such executable
 * here, and the failure is what BART already expects on a system without one.
 */
#include <errno.h>
#include <string.h>

#include "substitute/posix.h"

int getsubopt(char** optionp, char* const* tokens, char** valuep)
{
	char* option = *optionp;

	if ('\0' == *option)
		return -1;

	char* end = strchr(option, ',');

	if (NULL == end)
		end = option + strlen(option);

	char* equals = memchr(option, '=', (size_t)(end - option));
	size_t len = (size_t)((NULL != equals ? equals : end) - option);

	*optionp = ('\0' != *end) ? end + 1 : end;

	if ('\0' != *end)
		*end = '\0';

	for (int i = 0; NULL != tokens[i]; i++) {

		if ((0 == strncmp(option, tokens[i], len)) && ('\0' == tokens[i][len])) {

			*valuep = (NULL != equals) ? equals + 1 : NULL;
			return i;
		}
	}

	*valuep = option;
	return -1;
}

ssize_t readlink(const char* path, char* buf, size_t size)
{
	(void)path;
	(void)buf;
	(void)size;

	errno = ENOSYS;
	return -1;
}
