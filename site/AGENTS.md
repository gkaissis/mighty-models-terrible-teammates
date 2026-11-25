# Brief for agents working on a post

You are working in a copy of the article kit: a small Astro project that turns one Markdown file into one finished article page and publishes it to GitHub Pages. The design is fixed and approved. Your job is the post: its text, figures, references and front matter.

## The contract

- Write the post in `src/pages/index.md`. Put figures in `src/figures/` and references in `refs.bib`.
- Do not edit `src/styles/`, `src/layouts/`, `src/lib/`, `src/scripts/` or `src/assets/fonts/`. They are the shared look. If the post seems to need a change there, stop and tell the author what is missing; the fix belongs in the kit, not in one post.
- The page must make no requests to other hosts. No analytics, no embeds, no fonts or scripts from a CDN, no hot-linked images.
- Finish by running the checks in "Before you hand over". Report what you checked and anything you could not check.

## Commands

```bash
npm install
npm run dev        # http://localhost:4321, rebuilds as you save
npm run build      # writes the finished page to dist/
npm run check      # type and template check
```

Node 22 or newer. `refs.bib` is read when a page is built: after editing it, save `index.md` again (or restart `npm run dev`) to see the change.

## Files

| Path | What it is | Edit? |
|---|---|---|
| `src/pages/index.md` | The post: front matter, then Markdown | Yes |
| `src/figures/` | Images used by the post | Yes |
| `refs.bib` | BibTeX entries the post cites | Yes |
| `public/` | Files served as they are (favicon, licence texts, downloads) | Add only |
| `.github/workflows/deploy.yml` | Builds and publishes to GitHub Pages | Only the branch name |
| `src/styles/core.css` | The look, version 1 | No |
| `src/layouts/Article.astro` | The page: header, contents, text, footer | No |
| `src/lib/` | Margin notes, citations, figures, BibTeX | No |

## Front matter

`layout` and `title` are required. Leave out anything that does not apply; the page adapts.

| Field | Meaning |
|---|---|
| `layout` | Always `../layouts/Article.astro` |
| `title` | The headline |
| `subtitle` | One or two sentences under the headline, set in italic |
| `date`, `updated` | `YYYY-MM-DD` |
| `authors` | List. Each: `name`, optional `url`, `orcid`, `affiliation` (an id or a list of ids), `equal: true`, `corresponding: true`. Add `family` and `given` when the last word of the name is not the family name |
| `affiliations` | List. Each: `id`, `name`, optional `url` |
| `abstract` | One paragraph; used as the page description for search engines and link previews, not shown on the page |
| `doi` | Bare DOI, for example `10.5281/zenodo.1234567` |
| `repository` | URL of the code or data |
| `license` | Licence of the text, for example `CC BY 4.0`, or `{ name, url }`. Ask the author; do not choose one |
| `bibtex` | A complete BibTeX entry for citing the post. Left out, one is generated from the fields above |
| `legal` | URL of the legal notice and privacy page, shown in the footer |
| `site` | `{ name, url }`: an optional link back to a home page, shown at the top left |

Quote any value that contains a colon, or the build fails: `subtitle: "Two claims: one old, one new."`

## Writing

Plain Markdown, with four additions.

**Sections.** `##` starts a section; sections form the contents list. Use `###` rarely. There is no `#` in the body: the title comes from the front matter.

**Margin notes** are ordinary footnotes. The kit numbers them in reading order and sets each beside the paragraph that calls it.

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

**Figures** have three placements. The text inside the block is the caption and takes Markdown; captioned figures are numbered automatically.

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

**Maths** uses KaTeX: `$...$` inline and `$$...$$` on its own lines for display. Write a literal dollar sign as `\$`.

**Code** goes in fenced blocks with a language name. **Tables** are Markdown tables; align number columns to the right (`---:`).

## Figures in the house style

When you make a chart for the post, draw it to match the page.

- Ground `#F1EADC` (or transparent on the paper `#F9F4EA`), ink `#000000`, labels `#5E564D`, grid lines no darker than `#E0D6C3`.
- Series colours, in this order: dark red `#680F0D`, teal `#0E3A43`, forest `#1B3A18`. Ochre `#A85C19` is for highlights and washes only. For more than three series, use greys and label the lines directly.
- A serif face for labels (Spectral, or Georgia as the fallback). No sans-serif anywhere.
- Label lines directly where you can; drop legends, frames and background grids that carry no information.
- Prefer SVG for charts and diagrams. An SVG used as a figure cannot load fonts, so name `Georgia, serif` in it or convert text to outlines.

## The look, for reference

You do not implement these; they are why the files above are closed.

- One typeface, Spectral, self-hosted: display weight 300, text 400, headings 500, italic for labels. Code in JetBrains Mono.
- Body 18.5 px with a line height of 1.75; text column about 34 em; old-style figures in text, lining figures in tables.
- Dark red marks everything small: bullets, note numbers, the contents marker, the § on headings, link underlines, the end-of-article square.
- Links carry a 1 px dark-red underline, 2 px on hover.
- Print has its own layout: A4, text column 400 px, margin column 220 px, italic section headings, captions above figures.

## House style for prose

These are defaults; the author's draft overrides them.

- British spelling, as on g-k.ai ("centred", "memorisation", "organisation").
- Sentence case for the title and headings; no full stop at the end of a heading.
- Say who did what: "we" for the lab's work, "I" for the author's own view.
- Define an abbreviation at first use.
- Numbers in the text must match the figures, tables and sources. If you change one, change all of them.

Do not invent results, quotations, references or DOIs. If a claim needs a source and you do not have one, leave a visible `TODO(source)` in the text and list it in your report.

## Before you hand over

1. `npm run build` and `npm run check` finish without errors, and the build log has no `[article]` warnings (a missing reference or a figure without `alt`).
2. Open the built page and read it at desktop width, at phone width (375 px) and in print preview.
3. Every figure has `alt` text and a caption; every table and equation is readable at phone width (they scroll sideways if too wide).
4. Every link opens; every citation marker shows its reference.
5. The front matter is the post's own: title, authors, date, repository. No sample text from the kit remains in `index.md`, and `refs.bib` holds only what is cited.
6. In the browser's network panel, every request goes to the page's own host.

## Publishing

The workflow in `.github/workflows/deploy.yml` builds on every push to `main` and publishes to GitHub Pages. It passes the site address and base path to the build, so nothing in the post names the repository.

One-time setup, done by the author: in the repository on GitHub, Settings → Pages → Build and deployment → Source: "GitHub Actions".

The page is served under a path such as `/repository-name/`. Link to other files with relative paths or `#section` anchors, never with a leading `/`.

Do not push, publish, change repository settings or make the repository public unless the author asks you to.
