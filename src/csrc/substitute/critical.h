/* Internal: the OpenMP critical sections each thread holds. */
#ifndef BARTORCH_CRITICAL_H
#define BARTORCH_CRITICAL_H

/* Leaves every critical section the calling thread entered and has not left.
 * Called where BART's error catcher returns: an error raised inside a section
 * jumps past the call that would have left it. */
void bartorch_leave_held_criticals(void);

#endif
