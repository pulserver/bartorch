# The BART fork

BART is developed at [codeberg.org/mrirecon/bart](https://codeberg.org/mrirecon/bart),
which is the canonical upstream; the GitHub repository `mrirecon/bart` is an
archived mirror and is not used.  bartorch builds BART from
[pulserver/bart](https://github.com/pulserver/bart), a downstream integration
fork: the upstream history, branches and tags unchanged, and a small stack of
downstream patches on top.  The submodule `external/bart` pins a revision of
that fork.

## Branches

| Branch | Contents |
| --- | --- |
| `master` | Codeberg's `master`, fast-forwarded on each sync and never committed to |
| `downstream` | `master` plus the patch stack; the default branch, and the branch the submodule follows |
| Other upstream branches | Copied from Codeberg as they are, for reference |

Codeberg's tags are carried unchanged.  Two kinds of tag are the fork's own:
`downstream-YYYYMMDD`, the patch stack as it stood before a sync, and
`github-mirror-final`, the last commit of the archived GitHub mirror, which is
Codeberg's `v1.0.00` with a README change and exists only there.

The fork's clone uses two remotes:

```bash
git clone https://github.com/pulserver/bart.git
cd bart
git remote add upstream https://codeberg.org/mrirecon/bart.git
git fetch upstream --tags
```

## What belongs in the fork

A downstream patch is a generic change to BART: a portability fix, or a hook
that lets BART be embedded in or extended by another program.  Each patch is
one commit, or a short series, that is understandable on its own and carries
its reason in its message.  Where practical, such a change is also proposed
upstream on Codeberg; once upstream carries it, the downstream patch is
dropped at the next sync.

bartorch's own implementation stays in this repository: the C ABI, the
FINUFFT, cuFINUFFT and cuFFTDx substitutions, the FFT and BLAS/LAPACK tables,
and everything that refers to PyTorch or Python.  These are compiled in place
of BART's translation units, as `AGENTS.md` lists, not written into the fork.

## Syncing with Codeberg

The patch stack is rebased onto the new upstream, so that it remains a list of
patches on top of an unmodified upstream rather than a history of merges:

```bash
git fetch upstream --tags
git checkout master
git merge --ff-only upstream/master
git tag downstream-$(date +%Y%m%d) downstream
git checkout downstream
git rebase master
git push origin master --tags
git push --force-with-lease origin downstream
```

Every conflict in the rebase is resolved explicitly, and a patch that upstream
has made redundant is dropped.  The tag keeps the previous stack, and with it
every revision an earlier bartorch pinned, reachable after `downstream` is
rewritten.  The submodule is then moved as {doc}`workflow` describes.
