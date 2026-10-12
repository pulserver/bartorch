# Issue reports

Bugs, documentation defects and feature requests go to the
[issue tracker](https://github.com/pulserver/bartorch/issues/new/choose), whose
forms ask for the information below.  Search the existing issues for the error
message and the function name first.

A bug report contains:

- the observed and the expected behaviour, with the complete traceback;
- a minimal script on synthetic data, {func}`bartorch.tools.phantom` or a
  random tensor with a fixed seed, rather than acquired data;
- the shapes, dtypes and devices of the inputs, the axis and trajectory
  conventions assumed, and the options used, including regularization terms
  and Fourier-transform normalization;
- `bartorch.__version__`, the output of {func}`bartorch.build_info` (which
  includes the BART and FINUFFT versions compiled in), the Python and PyTorch
  versions, the operating system and the installation command;
- for a build from a checkout, `git rev-parse HEAD` and
  `git -C external/bart rev-parse HEAD`;
- on a CUDA device, the GPU model, the driver version, `torch.version.cuda`,
  `torch.cuda.is_available()` and {func}`bartorch.cuda_available`.

If importing bartorch or loading its library fails, report that error; the
functions above need the library.

A numerical discrepancy report states the reference compared with (an
independent implementation, an explicit sum, a closed form) and the relative
error observed.  A performance report states the problem size, the thread
count, whether the timing includes warm-up and device synchronization, and the
peak memory.

Share only data you may publish, and remove patient identifiers, credentials
and private paths from logs.  A security vulnerability is reported privately
({doc}`security`), not as an issue.
