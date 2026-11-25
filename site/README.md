# Article kit

One Markdown file in, one finished article page out, published to GitHub Pages. The page uses the core look of g-k.ai (Spectral, the paper palette, margin notes, an A4 print layout) without anything personal to that site, so it suits standalone posts and research pages with several authors.

## Start a post

1. Copy this folder to a new directory, without `node_modules/` and `dist/`, and make it a git repository.
2. Run `npm install`, then `npm run dev`.
3. Replace the sample in `src/pages/index.md`. The sample uses every element the page offers, so its source is the template.
4. Push to GitHub and set Settings → Pages → Source to "GitHub Actions". Every push to `main` then publishes the page.

`AGENTS.md` is the full brief: files, front matter, syntax for notes, citations and figures, the checks before publishing. `CLAUDE.md` points Claude Code at it.

## What is inside

- `src/styles/core.css`: the shared stylesheet, version 1. Posts do not edit it.
- `src/layouts/Article.astro`: header with authors, affiliations, "Cite this post" and "Copy link"; contents column; text with a margin column; footer.
- `src/lib/`: turns footnotes and `[@key]` citations into numbered margin notes, places figures, reads `refs.bib`.
- Self-hosted Spectral and JetBrains Mono (SIL Open Font License, texts in `public/licenses/`) and KaTeX for maths. The built page makes no requests to other hosts.

## Updating the kit

The kit lives in the g-k.ai repository under `article-kit/`. To bring a post up to date, copy `src/styles/`, `src/layouts/`, `src/lib/`, `src/scripts/` and `src/assets/fonts/` from there over the post's copies; the post's own files (`src/pages/`, `src/figures/`, `refs.bib`) are not touched.
