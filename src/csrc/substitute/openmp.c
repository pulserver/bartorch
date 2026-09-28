/*
 * `__kmpc_dispatch_deinit`, for a runtime that does not export it.
 *
 * clang 19 and later end every dynamically scheduled loop with a call to it;
 * LLVM's runtime exports it from 19 on.  On macOS and Windows the library is
 * bound to the runtime torch carries (cmake/openmp.cmake), and the libomp in
 * torch 2.3 to 2.5 predates the entry point: an import of it would fail to
 * load.  So the calls bind here instead, and this forwards to the runtime's
 * own where the process has one.  A runtime without it allocates nothing the
 * call would release, so there it does nothing.
 *
 * Compiled on macOS and Windows with OpenMP only; on Linux the toolchain's
 * compiler and runtime come as a pair.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef void (*deinit_fn)(void* loc, int32_t gtid);

/* Resolved once; threads racing here all find the same answer. */
#define UNRESOLVED ((deinit_fn)(uintptr_t)1)

static deinit_fn runtime_deinit(void)
{
	static deinit_fn cached = UNRESOLVED;

	deinit_fn fn = __atomic_load_n(&cached, __ATOMIC_ACQUIRE);

	if (UNRESOLVED == fn) {
#ifdef _WIN32
		HMODULE rt = GetModuleHandleA("libiomp5md.dll");
		fn = rt ? (deinit_fn)(void (*)(void))GetProcAddress(rt, "__kmpc_dispatch_deinit") : NULL;
#else
		fn = (deinit_fn)dlsym(RTLD_DEFAULT, "__kmpc_dispatch_deinit");
#endif
		__atomic_store_n(&cached, fn, __ATOMIC_RELEASE);
	}

	return fn;
}

__attribute__((visibility("hidden")))
void __kmpc_dispatch_deinit(void* loc, int32_t gtid)
{
	deinit_fn fn = runtime_deinit();

	if (NULL != fn)
		fn(loc, gtid);
}
