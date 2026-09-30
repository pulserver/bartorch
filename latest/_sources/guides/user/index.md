# User guide

The tasks that come before a first reconstruction: checking that a platform is
supported, installing bartorch with the optional components a workflow needs,
and bringing acquired data into the array layout and trajectory units the
package expects.  The last three pages say where to report a defect, ask a
question or disclose a vulnerability.  The reasons behind the conventions are
in {doc}`../../explanation/index`, and complete reconstructions in the
{doc}`examples <../../examples/index>`.

| Page | Purpose |
| --- | --- |
| {doc}`prerequisites` | Python and PyTorch requirements, supported platforms, and the optional extras |
| {doc}`installation` | Installing PyTorch and bartorch, extras, source builds, CUDA, the non-Cartesian backends, the OpenMP runtime and the command line |
| {doc}`conventions` | Arranging k-space as tensors, converting trajectories to grid units, and reading and writing CFL files |
| {doc}`issues` | What a bug report or a numerical discrepancy report contains |
| {doc}`discussions` | Where questions and proposals go |
| {doc}`security` | Reporting a vulnerability privately |

```{toctree}
:hidden:

prerequisites
installation
conventions
issues
discussions
security
```
