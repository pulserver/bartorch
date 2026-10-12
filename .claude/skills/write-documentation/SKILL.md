---
name: write-documentation
description: Write, move or audit bartorch documentation — docstrings, user-guide, developer-guide, explanation, API and gallery pages, and the README. Use before creating or substantially modifying any documentation, and when deciding which documentation type a piece of material belongs to.
---

# Write documentation

`docs/guides/developer/documentation.md` is the documentation guide and
`docs/guides/developer/terminology.md` the project's terminology; both bind.
Read them before writing. The `Documentation` section of `AGENTS.md` states
the layout rules this procedure applies.

## Choose the type first

| Location | Type | Answers |
|---|---|---|
| `docs/guides/user/` | One page: platforms, installation, getting help | How do I install it? |
| `docs/guides/developer/` | Contributor procedure and conventions | How is this repository developed? |
| `docs/explanation/` | Conceptual explanation | Why does it work this way? |
| `docs/examples/` | Executable scripts: the Course, then the Tours | What does a representative reconstruction look like? |
| `docs/api/` | Reference, from the docstrings | What exactly does this object do? |
| `docs/misc/` | License, related projects, citation | |

Material in the wrong type is moved, not deleted. The prose style of one type
is not carried into another.

## Mechanics

- An API page lists its objects in tables whose first column is an `{obj}`
  role. `docs/api_objects.py` collects them into the untracked holder that
  writes the per-object stubs. A new public name needs a row:
  `tests/test_docs.py` holds every public name to a row and every row to a
  public name, and `tests/test_docstrings.py` holds each documented default to
  the signature.
- An example is `docs/examples/<section>/NN-title.py`; its module docstring is
  the page. Figure layout and other code a reader would not type goes between
  `# sphinx_gallery_start_ignore` and `# sphinx_gallery_end_ignore`. Figures
  use `docs/gallery_style.py`; no script sets font sizes or DPI.
- A Course lesson has a short introduction, learning objectives and
  previous/next lines; a Tour opens with its objective and a prerequisites
  line, and has none. `docs/gallery/index.md`, `course.md` and `tours.md`
  carry the tables `tests/test_docs.py` holds to `GALLERY_SECTIONS`.
- An explanation page with more than one `##` section opens with a TL;DR
  admonition (`:class: tldr`) directly under its title.

## Verify and build

Verify semantics against the implementation, its tests, BART's source and the
literature; existing prose is not evidence. Then:

```bash
./scripts/build_docs.sh --execute
./scripts/run_tests.sh tests/test_docs.py tests/test_docstrings.py
```

Inspect the rendered pages and the sidebar, in the light and the dark theme.
