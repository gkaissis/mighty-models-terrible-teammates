# Article kit (HTAIL house style v2.1.0)

One Markdown file in, one finished static article page out, published to GitHub Pages. The kit pairs the editorial look of g-k.ai (Spectral, the warm paper palette, margin notes, A4 print layout) with technical-report ergonomics (build-time TeX Gyre Pagella Math, baseline-aligned speaker quotes, clickable figure lightbox, folded appendices, verbatim prompt/code boxes and subsection navigation), with zero external runtime requests.

For matching PDF and arXiv preprints in the same HTAIL visual identity (Spectral, TeX Gyre Pagella Math, IBM Plex Mono, and pure-white publication figures), see the companion `latex-kit/` template.

## Start a post

1. Copy this folder to a new directory, without `node_modules/` and `dist/`, and make it a git repository.
2. Run `npm install`, then `npm run dev`.
3. Replace the sample in `src/pages/index.md`. The sample demonstrates every v2.0.0 element, so its source doubles as the template.
4. Push to GitHub and set Settings → Pages → Source to "GitHub Actions". Every push to `main` then publishes the page.

`AGENTS.md` is the full brief: files, front matter, house prose rules, syntax for notes, citations, quotes, folds, figures and tables, and pre-publish checks. `CLAUDE.md` points Claude Code at it. See `CHANGELOG.md` for release notes (`v1.0.0` → `v2.0.0`).

## What is inside

- `src/styles/core.css`: the shared stylesheet, HTAIL house style v2.0.0 (`box-sizing: border-box` exact column alignment, ragged-right `text-wrap: pretty` prose, baseline-locked speaker quotes, right-aligned numeric tables, lightbox, codeboxes, print). Posts do not edit it.
- `src/styles/fonts.css`: self-hosted **Spectral** (300/400/500/600 normal, 300/400/600 italic) and **IBM Plex Mono** (400/500 normal, 400 italic, complete glyph set).
- `src/layouts/Article.astro`: header with document type, British ordinal date, authors (linking `url` or ORCID), affiliations, Cite + Copy BibTeX, Copy link, DOI/Preprint/Code/Data links; scrollable contents column with subsections; text + margin column; native `<dialog>` figure lightbox; footer.
- `src/lib/math.mjs`: build-time **MathJax v4 SVG** renderer with **TeX Gyre Pagella Math**, adjacent-punctuation binding (`.math-keep`) and accessible MathML (`sr-only`), shipping zero client-side maths JS.
- `src/lib/article.mjs`: turns footnotes and `[@key]` citations into numbered margin notes (automatically folding long notes > 220 characters), handles `:::figure`, `:::aside`, `:::quote` and `:::fold`, wraps code/prompt boxes with line numbers and wrap/copy controls, formats table captions, folds `### Appendix X:` subsections and links internal cross-references with colour-coded underlines.
- `astro.config.mjs`: configures the Markdown pipeline and the WCAG AA+ `htail-paper` Shiki syntax theme.
- `.github/workflows/deploy.yml`: native Node 24 GitHub Actions workflow for GitHub Pages.
- `scripts/lint_prose.py`: deterministic HTAIL prose, spelling, typography and SVG hygiene checker wired into `npm run lint:prose` and `npm run check`.
- Self-hosted font licences in `public/licenses/` (`OFL-Spectral.txt`, `OFL-IBM-Plex-Mono.txt`, `GFL-TeX-Gyre-Pagella.txt`). The built page makes zero requests to external hosts.

## Updating an existing post

To bring an existing post up to date with house style v2.1.0, copy `astro.config.mjs`, `package.json`, `AGENTS.md`, `README.md`, `CLAUDE.md`, `scripts/`, `src/styles/`, `src/layouts/`, `src/lib/`, `src/scripts/`, `src/assets/fonts/` and `public/licenses/` over the post's copies, then run `npm install`. The post's own files (`src/pages/`, `src/figures/`, `refs.bib`) remain untouched.
