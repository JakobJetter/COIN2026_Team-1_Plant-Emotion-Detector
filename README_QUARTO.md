# COINs / Springer T1-book report (Quarto)

A Quarto project that renders a single PDF resembling the COINs
conference (Springer T1-book) template. Each section lives in its own
`.qmd` file under `sections/`.

## Layout

```
coins-report/
├── _quarto.yml              # project + PDF format config (engine, geometry, partials)
├── report.qmd               # metadata (title/abstract/…) + section includes
├── references.bib           # bibliography
├── pyproject.toml           # Python deps for executable code cells (uv)
├── partials/
│   └── title.tex            # Springer-style left-aligned title/abstract block
├── tex/
│   └── preamble.tex         # fonts, headings, captions, running head
├── figures/                 # put images here
└── sections/
    ├── 01-introduction.qmd
    ├── 02-related-work.qmd
    ├── 03-hypotheses.qmd
    ├── 04-method.qmd
    ├── 05-results.qmd
    ├── 06-discussion.qmd
    ├── 07-conclusion.qmd
    └── 08-appendix.qmd
```

## Prerequisites

- **Quarto CLI** (`quarto --version`) — install from <https://quarto.org>.
- **A LaTeX engine** — `pdflatex` from TeX Live, or run `quarto install tinytex`.
- **uv** (only if you use `python` code cells) — <https://docs.astral.sh/uv/>.

The PDF is built with **pdflatex** and **mathptmx** (Times), which are
present in every TeX Live / TinyTeX install — no extra fonts required.

## Build

With uv (recommended, so code cells run in a managed environment):

```bash
uv run quarto render          # builds _output/report.pdf
```

Or, if you have no code cells / don't use uv:

```bash
quarto render                 # builds _output/report.pdf
# or a single file:
quarto render report.qmd --to pdf
```

Live preview while writing:

```bash
uv run quarto preview report.qmd
```

The output lands in `_output/report.pdf` (set by `output-dir` in
`_quarto.yml`). `keep-tex: true` also writes `report.tex` for debugging.

## Editing

- **Title, authors, abstract, running head:** edit the YAML at the top of
  `report.qmd`. `paperabstract` is plain text (single paragraph). To allow
  Markdown/citations in the abstract, rename the key to `abstract` in
  `report.qmd` **and** change `$paperabstract$` to `$abstract$` in
  `partials/title.tex`.
- **Content:** edit the files in `sections/`. Do **not** add a YAML header
  to a section file — they are fragments included by `report.qmd`.
- **Add a section:** create `sections/09-foo.qmd` starting with `# Foo`,
  then add `{{< include sections/09-foo.qmd >}}` in `report.qmd`.
- **Figures:** drop an image in `figures/` and reference it, e.g.
  `![Caption.](figures/pipeline.png){#fig-x width=70%}`, then cross-ref
  with `@fig-x`. Captions render as **Fig. N.** automatically.
- **Tables:** use a Markdown table with a `{#tbl-x}` id; captions render as
  **Table N.** Cross-ref with `@tbl-x`.
- **Citations:** add entries to `references.bib` and cite with `[@key]`.

## Matching the template exactly

The geometry (A4; margins top 5.0 / bottom 5.6 / left 4.7 / right 4.6 cm),
10 pt Times body, 0.42 cm paragraph indent, bold numbered headings,
bold-italic subsections, and bold **Fig.**/**Table** caption labels are
all set in `_quarto.yml` + `tex/preamble.tex`. Tweak those two files to
adjust the look.

### References style

Default output uses Quarto's built-in citeproc (author–date). To match the
Springer **numbered** reference list from the template, drop a Springer CSL
next to this README and point to it:

```yaml
# in _quarto.yml, under format: pdf:
csl: springer-lecture-notes-in-computer-science.csl
```

CSL files: <https://github.com/citation-style-language/styles>
(`springer-lecture-notes-in-computer-science.csl` or
`springer-basic-author-date.csl`).

## Note

The original template is a Word/Springer macro template; this project
reproduces its *appearance* in LaTeX/PDF via Quarto. Minor pixel-level
differences (exact leading, heading sizes) can be tuned in
`tex/preamble.tex`.
