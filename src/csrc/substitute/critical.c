/*
 * OpenMP critical sections, entered and left through the runtime, with a
 * record per thread of the sections it holds.
 *
 * BART's `error` leaves a command by a long jump.  A jump out of a critical
 * section skips the call that leaves it, so the section stays locked and the
 * next thread to enter it waits for ever: a bad option leaves `bart_getopt`
 * held and every command after it hangs, a missing input does the same to
 * the section every file is loaded in.  So every critical section compiled
 * into the library is entered through the definitions below, which forward to
 * the runtime and keep, per thread, the sections entered and not yet left;
 * the two places that catch BART's errors leave whatever is still held once
 * the catch returns (`bartorch_leave_held_criticals`).
 *
 * A thread may not enter a section it is already inside, so one it enters
 * while holding can only be held from before a jump, and is left first.  That
 * is the case of BART's cleanup after a failed command, which runs on the
 * same thread before the catch has returned to the library.
 *
 * clang calls `__kmpc_critical` and GCC `GOMP_critical_start` and
 * `GOMP_critical_name_start`; both sets are defined, and each forwards to the
 * runtime the library is bound to -- on Windows the one torch loads
 * (cmake/openmp.cmake), elsewhere the image `omp_get_max_threads` resolved
 * to.  The definitions are hidden, so they bind the library's own objects,
 * BART's and FINUFFT's, and nothing outside it.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE	/* dladdr, on glibc */
#endif

#include "substitute/critical.h"

#ifdef _OPENMP

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define HIDDEN __attribute__((visibility("hidden")))

typedef void (*any_fn)(void);
typedef void (*kmpc_fn)(void* loc, int32_t gtid, void* crit);
typedef void (*kmpc_hint_fn)(void* loc, int32_t gtid, void* crit, uint32_t hint);
typedef void (*gomp_fn)(void);
typedef void (*gomp_name_fn)(void** name);

enum entry {
	KMPC_CRITICAL, KMPC_CRITICAL_WITH_HINT, KMPC_END_CRITICAL,
	GOMP_START, GOMP_END, GOMP_NAME_START, GOMP_NAME_END,
	ENTRIES
};

static const char* const entry_names[ENTRIES] = {
	"__kmpc_critical", "__kmpc_critical_with_hint", "__kmpc_end_critical",
	"GOMP_critical_start", "GOMP_critical_end",
	"GOMP_critical_name_start", "GOMP_critical_name_end",
};

static any_fn lookup(const char* name)
{
	any_fn fn = NULL;
#ifdef _WIN32
	HMODULE rt = GetModuleHandleA("libiomp5md.dll");

	if (NULL != rt)
		fn = (any_fn)(void (*)(void))GetProcAddress(rt, name);
#else
	Dl_info info;

	if (dladdr((void*)omp_get_max_threads, &info) && (NULL != info.dli_fname)) {

		void* rt = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);

		if (NULL != rt) {

			fn = (any_fn)dlsym(rt, name);
			dlclose(rt);
		}
	}

	if (NULL == fn)
		fn = (any_fn)dlsym(RTLD_DEFAULT, name);
#endif
	if (NULL == fn) {

		fprintf(stderr, "bartorch: the OpenMP runtime does not export %s\n", name);
		abort();
	}

	return fn;
}

/* Resolved once; threads racing here all find the same answer. */
static any_fn runtime(enum entry which)
{
	static any_fn cached[ENTRIES];

	any_fn fn = __atomic_load_n(&cached[which], __ATOMIC_ACQUIRE);

	if (NULL == fn) {

		fn = lookup(entry_names[which]);
		__atomic_store_n(&cached[which], fn, __ATOMIC_RELEASE);
	}

	return fn;
}


enum kind { KMPC, GOMP, GOMP_NAME };

struct held {

	enum kind kind;
	void* lock;
	void* loc;
	int32_t gtid;
};

/* Deeper than any nesting of BART's sections; one past it goes unrecorded,
 * which leaves it as the runtime alone would. */
#define MAX_HELD 32

static _Thread_local struct held held[MAX_HELD];
static _Thread_local int nheld;

/* The lock GCC's unnamed section is, as far as the record is concerned. */
static char gomp_unnamed;

static void leave(const struct held* h)
{
	switch (h->kind) {

	case KMPC:
		((kmpc_fn)runtime(KMPC_END_CRITICAL))(h->loc, h->gtid, h->lock);
		break;

	case GOMP:
		((gomp_fn)runtime(GOMP_END))();
		break;

	case GOMP_NAME:
		((gomp_name_fn)runtime(GOMP_NAME_END))((void**)h->lock);
		break;
	}
}

static void remember(enum kind kind, void* lock, void* loc, int32_t gtid)
{
	if (nheld < MAX_HELD)
		held[nheld++] = (struct held){ kind, lock, loc, gtid };
}

static bool forget(const void* lock, struct held* out)
{
	for (int i = nheld - 1; i >= 0; i--) {

		if (held[i].lock != lock)
			continue;

		if (NULL != out)
			*out = held[i];

		memmove(&held[i], &held[i + 1], (size_t)(nheld - i - 1) * sizeof held[0]);
		nheld--;

		return true;
	}

	return false;
}

static void leave_if_held(const void* lock)
{
	struct held stale;

	if (forget(lock, &stale))
		leave(&stale);
}

void bartorch_leave_held_criticals(void)
{
	while (0 < nheld) {

		nheld--;
		leave(&held[nheld]);
	}
}


HIDDEN void __kmpc_critical(void* loc, int32_t gtid, void* crit)
{
	leave_if_held(crit);
	((kmpc_fn)runtime(KMPC_CRITICAL))(loc, gtid, crit);
	remember(KMPC, crit, loc, gtid);
}

HIDDEN void __kmpc_critical_with_hint(void* loc, int32_t gtid, void* crit, uint32_t hint)
{
	leave_if_held(crit);
	((kmpc_hint_fn)runtime(KMPC_CRITICAL_WITH_HINT))(loc, gtid, crit, hint);
	remember(KMPC, crit, loc, gtid);
}

HIDDEN void __kmpc_end_critical(void* loc, int32_t gtid, void* crit)
{
	forget(crit, NULL);
	((kmpc_fn)runtime(KMPC_END_CRITICAL))(loc, gtid, crit);
}

HIDDEN void GOMP_critical_start(void)
{
	leave_if_held(&gomp_unnamed);
	((gomp_fn)runtime(GOMP_START))();
	remember(GOMP, &gomp_unnamed, NULL, 0);
}

HIDDEN void GOMP_critical_end(void)
{
	forget(&gomp_unnamed, NULL);
	((gomp_fn)runtime(GOMP_END))();
}

HIDDEN void GOMP_critical_name_start(void** name)
{
	leave_if_held(name);
	((gomp_name_fn)runtime(GOMP_NAME_START))(name);
	remember(GOMP_NAME, name, NULL, 0);
}

HIDDEN void GOMP_critical_name_end(void** name)
{
	forget(name, NULL);
	((gomp_name_fn)runtime(GOMP_NAME_END))(name);
}

#else

void bartorch_leave_held_criticals(void)
{
}

#endif
