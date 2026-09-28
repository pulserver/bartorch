/*
 * The part of C11's <threads.h> BART's misc/lock.c uses, over pthreads.
 *
 * On the include path only where the C library has no <threads.h> of its own,
 * which is MinGW's before winpthreads grew one; winpthreads is what serves it.
 */
#ifndef BARTORCH_C11THREADS_H
#define BARTORCH_C11THREADS_H

#include <pthread.h>

typedef pthread_mutex_t mtx_t;
typedef pthread_cond_t cnd_t;

enum { mtx_plain = 0 };
enum { thrd_success = 0, thrd_error = 1, thrd_busy = 2 };

static inline int mtx_init(mtx_t* mx, int type)
{
	(void)type;
	return pthread_mutex_init(mx, NULL) ? thrd_error : thrd_success;
}

static inline int mtx_lock(mtx_t* mx)
{
	return pthread_mutex_lock(mx) ? thrd_error : thrd_success;
}

static inline int mtx_trylock(mtx_t* mx)
{
	return pthread_mutex_trylock(mx) ? thrd_busy : thrd_success;
}

static inline int mtx_unlock(mtx_t* mx)
{
	return pthread_mutex_unlock(mx) ? thrd_error : thrd_success;
}

static inline void mtx_destroy(mtx_t* mx)
{
	pthread_mutex_destroy(mx);
}

static inline int cnd_init(cnd_t* cnd)
{
	return pthread_cond_init(cnd, NULL) ? thrd_error : thrd_success;
}

static inline int cnd_wait(cnd_t* cnd, mtx_t* mx)
{
	return pthread_cond_wait(cnd, mx) ? thrd_error : thrd_success;
}

static inline int cnd_broadcast(cnd_t* cnd)
{
	return pthread_cond_broadcast(cnd) ? thrd_error : thrd_success;
}

static inline void cnd_destroy(cnd_t* cnd)
{
	pthread_cond_destroy(cnd);
}

#endif
