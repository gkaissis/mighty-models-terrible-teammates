# Brief for agents working on a post

You are working in a copy of the HTAIL article kit (**house style v2.0.0**): a small Astro project that turns one Markdown file into one finished article page and publishes it to GitHub Pages. The design is fixed and approved. Your job is the post: its text, figures, references and front matter.

For a matching PDF or arXiv preprint in the same HTAIL visual identity (Spectral, TeX Gyre Pagella Math, IBM Plex Mono, and pure-white publication figures), pair this repository with the companion `latex-kit/` template.

## The contract

- Write the post in `src/pages/index.md`. Put figures in `src/figures/` and references in `refs.bib`.
- Do not edit `src/styles/`, `src/layouts/`, `src/lib/`, `src/scripts/` or `src/assets/fonts/`. They are the shared look. If the post seems to need a change there, stop and tell the author what is missing; the fix belongs in the kit, not in one post.
- The page must make **zero requests to external hosts**. No analytics, no embeds, no fonts or scripts from a CDN, no hot-linked images.
- Finish by running the checks in "Before you hand over". Report what you checked and anything you could not check.

## Commands

```bash
npm install
npm run dev        # http://localhost:4321, rebuilds as you save
npm run build      # writes the finished page to dist/
npm run check      # type and template check
```

Node 22 or newer (Node 24 recommended). `refs.bib` is read when a page is built: after editing it, save `index.md` again (or restart `npm run dev`) to see the change.

## Files

| Path | What it is | Edit? |
|---|---|---|
| `src/pages/index.md` | The post: front matter, then Markdown | Yes |
| `src/figures/` | Images used by the post | Yes |
| `refs.bib` | BibTeX entries the post cites | Yes |
| `public/` | Files served as they are (favicon, licence texts, downloads) | Add only |
| `.github/workflows/deploy.yml` | Builds and publishes to GitHub Pages (Node 24 actions) | Only the branch name |
| `src/styles/core.css` | The look, HTAIL house style v2.0.0 | No |
| `src/styles/fonts.css` | Self-hosted Spectral and IBM Plex Mono declarations | No |
| `src/layouts/Article.astro` | The page: header, contents, text, footer, lightbox | No |
| `src/lib/` | MathJax v4 Pagella SVG, margin notes, quotes, folds, codeboxes, xrefs, BibTeX | No |
| `src/scripts/article.ts` | Progressive enhancements (lightbox, copy/wrap, folds, scrollspy) | No |

## Front matter

`layout` and `title` are required. Leave out anything that does not apply; the page adapts.

| Field | Meaning |
|---|---|
| `layout` | Always `../layouts/Article.astro` |
| `title` | The headline (sentence case, no trailing full stop) |
| `subtitle` | One or two sentences under the headline, set in italic |
| `type` | Document type shown before the date, for example `Technical Report` or `Research Note` |
| `date`, `updated` | `YYYY-MM-DD`, formatted automatically in British ordinal style (`25th of November 2026`; if `updated` is present, `. Updated ...`) |
| `lang` | Optional BCP-47 language tag (defaults to `en`; use `en-GB` for British English) |
| `authors` | List. Each: `name`, optional `url` (preferred link target), `orcid` (fallback link when `url` is omitted), `affiliation` (an id or list of ids), `equal: true`, `corresponding: true`. Add `family` and `given` when the last word of the name is not the family name |
| `affiliations` | List. Each: `id`, `name`, optional `url` |
| `abstract` | One paragraph; used as the page description for search engines and link previews, not shown on the page |
| `doi` | Bare DOI, for example `10.5281/zenodo.1234567` |
| `preprint` | URL string (labelled `Preprint`) or `{ url, label }` (for example `{ url: "https://arxiv.org/abs/...", label: "arXiv" }`), shown in both header and footer |
| `repository` | URL of the source code repository (shown as `Code` in the header and `Source repository` in the footer) |
| `dataset` | URL string (labelled `Data`) or `{ url, label }` (for example `{ url: "https://huggingface.co/datasets/...", label: "Data (HuggingFace)" }`), shown in both header and footer |
| `license` | Licence of the text, for example `CC BY 4.0`, or `{ name, url }`. Ask the author; do not choose one |
| `bibtex` | A complete BibTeX entry for citing the post (displayed with a Copy BibTeX button). Left out, one is generated from the fields above |
| `legal` | URL of the legal notice and privacy page, shown in the footer |
| `site` | `{ name, url }`: an optional link back to a home page, shown at the top left |

Quote any value that contains a colon, or the build fails: `subtitle: "Two claims: one old, one new."`

## Writing

Plain Markdown, with editorial directives and build-time enhancements.

**Sections and subsections.** `##` starts a section (separated by a hairline rule); `###` starts an upright SemiBold subsection. Both appear in the scrollable side contents list (subsections indented), and hovering over either shows a `§` button that copies a direct link. There is no `#` in the body: the title comes from the front matter.

**Cross-references.** Mentions of `Section N`, `Sections M and N`, `Figure N`, `Table N`, `Table A1`, `Appendix A` or `Appendices A–C` automatically become inline colour-coded underlined links (dark red for sections, warm ochre-brown for appendices, teal for figures, forest green for tables). Following a link into a folded appendix opens the appendix automatically before scrolling.

**Margin notes** are ordinary footnotes. The kit numbers them in reading order and sets each beside the paragraph that calls it; notes longer than 220 characters fold automatically into a margin summary and expand on click, target or print.

```markdown
Conference deadlines play a shrinking role in how we plan.[^deadlines]

[^deadlines]: The note text. It can contain *emphasis*, links and maths.
```

**Citations** take a key from `refs.bib`. They become margin notes in the same numbering; a repeated source is shortened and points back to its first note.

```markdown
The approach follows Shneiderman.[@shneiderman2022] Two sources can share a note.[@knuth1984; @tufte2001]
```

Put the marker after the punctuation, as above. Every key must exist in `refs.bib`; a missing one is printed in red on the page and reported in the build log as an `[article]` warning. Give each entry a `doi` where one exists.

**Asides** are margin notes without a number. An aside sits beside the block before it.

```markdown
:::aside
A remark that belongs beside the paragraph above.
:::
```

**Speaker-attributed quotes** format verbatim model outputs or dialogue transcripts with a colour-coded accent border (`teal`, `red`, `forest`, `ochre` or `neutral`; defaults to `teal`). Speaker and metadata share one baseline and align flush with the text measure (`629 px`). Separate metadata clauses with commas or parentheses—never middle dots.

```markdown
:::quote{speaker="Claude Opus 5.5" meta="Config Repair, Round 1" accent="teal"}
"Both runs ended with identical results, but HR shared its full constraint list up front."
:::
```

**Folded appendices and disclosures.** Any `### Appendix <Letter>: ...` heading automatically wraps its subsection in a collapsed appendix fold (`.appendix-fold`) that stays listed in the side TOC and expands automatically on anchor jump or print. For general collapsible blocks anywhere in the body, use `:::fold{summary="..."}` (or standard HTML `<details>`).

```markdown
:::fold{summary="Technical details"}
Optional methodology or worked example.
:::
```

**Figures** have three placements. The text inside the block is the caption and takes Markdown; captioned figures are numbered automatically and enlarge on click in a paper-card lightbox with their full HTML caption.

```markdown
:::figure{src="../figures/loss.png" alt="Training loss falls from 2.3 to 0.4 over 50 epochs and flattens after epoch 30."}
The caption, as a full sentence.
:::

:::figure{src="../figures/overview.png" alt="..." .wide}
A wide figure spans the text and margin columns (960 px).
:::

:::figure{src="../figures/spark.svg" alt="..." .margin}
:::
```

- `alt` is required and describes what the figure shows, including the finding. Do not repeat the caption.
- A text-column figure is 630 px wide, a wide one 960 px, a margin one 250 px. Export raster images at twice that width.
- Paths are relative to `index.md`. Images under `src/figures/` are optimised by the build; do not put figures in `public/`.

**Maths vs. prose numerals.** Maths is compiled at build time with **MathJax v4 SVG** using **TeX Gyre Pagella Math** (with accessible hidden MathML for screen readers and copy-paste): `$...$` inline and `$$...$$` on its own lines for display. Write a literal dollar sign as `\$`. Follow scholarly publishing conventions strictly:

- **Use `$...$` for mathematical notation:** variables, test statistics, equations, algebraic constraints and symbolic thresholds (`$n = 160$`, `$p < 0.001$`, `$R^2 = 0.5351$`, `$\Delta_{\text{Info}}$`, `$2^k$`, `$\ge 5$`, `$\text{cfg.cache\_ttl} \times \text{cfg.retry\_cap} \le 180$`).
- **Use plain text (not `$...$`) for prose and tabular numerals:** plain scores, counts, ratios, percentages, percentage-point deltas, confidence intervals and multipliers (`0.97`, `9,568`, `704/720`, `−0.8 pp`, `[−3.6, +1.7]`, `1.8×` to `4.9×`). Always use true Unicode minus (`−`, U+2212) rather than an ASCII hyphen for negative numbers, true Unicode multiplication (`×`, U+00D7) for prose multipliers, and a non-breaking space (`\u00A0`) between a number and `pp` in tables so units never wrap onto a second line.

**Verbatim code and prompt boxes.** Fenced blocks are highlighted at build time with the WCAG AA+ `htail-paper` Shiki theme and wrapped with a language bar, copy button, wrap toggle and aligned line numbers (for $\ge 2$ lines). Code blocks (`bash`, `python`, `json`, etc.) start unwrapped; `text` / `markdown` prompt blocks start wrapped.

**Tables.** Standard Markdown tables; align number columns to the right (`---:`). Right-aligned headers align over their numeric data columns, and right-aligned data cells stay on a single line (`white-space: nowrap` on screen, wrapping cleanly in print) inside a horizontally scrollable `.table-wrap`. A paragraph immediately following a table that starts with `*Table N: ...*` or `*Table A1: ...*` is automatically formatted as a styled `.table-caption` with an italic `Table N.` lead-in and indexed for cross-references.

## Figures in the house style

When you make a chart for the post, draw it to match the page.

- Ground `#F1EADC` (or transparent on the paper `#F9F4EA`), ink `#000000`, labels `#5E564D`, grid lines no darker than `#E0D6C3`. (When exporting matching figures for `latex-kit/` PDF preprints, use pure white `#FFFFFF` ground with Spectral serif and `pdf.fonttype = 42`.)
- Series colours, in this order: dark red `#680F0D`, teal `#0E3A43`, forest `#1B3A18`. Ochre `#A85C19` is for highlights and washes only. For more than three series, use greys and label the lines directly.
- A serif face for labels (Spectral, or Georgia as the fallback). No sans-serif anywhere.
- Never use middle dots (U+00B7) or em-dashes in chart axis labels or titles; use commas or parentheses instead.
- Label lines directly where you can; drop legends, frames and background grids that carry no information.
- Prefer SVG for web charts and vector PDF for LaTeX preprints. An SVG used as a figure cannot load web fonts, so name `Georgia, serif` in it or convert text to outlines.

## The look, for reference

You do not implement these; they are why the files above are closed.

- Self-hosted typography: **Spectral** (300 display, 400 body, 500 table headers, 600 SemiBold for `<strong>` and headings; italic for labels), **TeX Gyre Pagella Math** (build-time SVG outlines), and **IBM Plex Mono** (400/500 normal, 400 italic, complete glyph set).
- Body 18.5 px with a line height of 1.75; text column 34 em (`629 px`) with uniform `box-sizing: border-box` so quotes, code boxes, disclosures and headings share the exact same right edge; old-style figures in running text, lining tabular figures in tables.
- Uniform ragged-right web justification (`text-align: left`, `text-wrap: pretty`, `hyphens: none`) across all prose, lists, blockquotes, captions and margin notes; `text-wrap: balance` on headings.
- Dark red marks structural details: bullets, note numbers, the contents marker, the `§` on headings, link underlines, the end-of-article square.
- Standard links carry a 1 px dark-red underline (2 px on hover); internal cross-references carry colour-coded 1 px underlines (2 px on hover).
- Print has its own layout: A4, text column 400 px, margin column 220 px, expanded folds and appendices, wrapped verbatim boxes, captions above figures.

## House style for prose

These are defaults; the author's draft overrides them.

- British spelling, as on g-k.ai ("centred", "memorisation", "organisation", "normalise", "behaviour").
- **No middle dots (U+00B7) anywhere**—not in datelines, quote metadata, resource bars, tables or figures. Use commas, periods, pipes or parentheses instead.
- Sentence case for the title and headings; no full stop at the end of a heading.
- Say who did what: "we" for the lab's work, "I" for the author's own view.
- Define an abbreviation at first use.
- Numbers in the text must match the figures, tables and sources. If you change one, change all of them.

Do not invent results, quotations, references or DOIs. If a claim needs a source and you do not have one, leave a visible `TODO(source)` in the text and list it in your report.

## Before you hand over

1. `npm run check` and `npm run build` finish without errors, and the build log has no `[article]` warnings (a missing reference, bad quote accent or a figure without `alt`).
2. Confirm zero middle dots (U+00B7) appear in source files, SVGs or built HTML (`dist/index.html`).
3. Open the built page and read it at desktop width, at phone width (375 px) and in print preview.
4. Every figure has `alt` text, a caption and opens cleanly in the lightbox; every table, quote box, code box and equation aligns cleanly with the text measure and remains readable at phone width.
5. Every link and cross-reference opens (including links into folded appendices); every citation marker shows its reference.
6. The front matter is the post's own: title, type, authors, date, repository, dataset. No sample text from the kit remains in `index.md`, and `refs.bib` holds only what is cited.
7. In the browser's network panel (and built HTML), every request goes to the page's own host.

## Publishing

The workflow in `.github/workflows/deploy.yml` builds on every push to `main` using native Node 24 GitHub Actions and publishes to GitHub Pages. It passes the site address and base path to the build, so nothing in the post names the repository.

One-time setup, done by the author: in the repository on GitHub, Settings → Pages → Build and deployment → Source: "GitHub Actions".

The page is served under a path such as `/repository-name/`. Link to other files with relative paths or `#section` anchors, never with a leading `/`.

Do not push, publish, change repository settings or make the repository public unless the author asks you to.
