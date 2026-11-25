// The two plugins that turn a plain Markdown post into the article layout.
//
// remarkArticle (Markdown tree):
//   :::figure{src="../figures/x.png" alt="..." .wide}   -> <figure> with an optimised image and a caption
//   :::aside                                             -> an unnumbered margin note
//   [@key] and [@a; @b]                                  -> citation markers, resolved from refs.bib
//   reading time                                         -> frontmatter.minutes
//
// rehypeArticle (HTML tree):
//   footnotes and citations become numbered notes in the margin column, beside the paragraph that cites them;
//   margin figures and asides join the margin of the block before them; figures are numbered;
//   tables get a scroll wrapper; the last paragraph gets the end mark.

import fs from 'node:fs';
import path from 'node:path';
import { formatReference, parseBibtex, shortRef } from './bibtex.mjs';

const warn = (file, message) => console.warn('[article] ' + (file.path ? path.relative(process.cwd(), file.path) + ': ' : '') + message);

const CITE = /\[((?:@[\w:.+/-]+)(?:\s*;\s*@[\w:.+/-]+)*)\]/g;

function walk(node, fn, parent = null, index = 0) {
  if (fn(node, parent, index) === false) return;
  if (node.children) for (let i = 0; i < node.children.length; i++) walk(node.children[i], fn, node, i);
}

// ---------------------------------------------------------------- remark

export function remarkArticle() {
  return (tree, file) => {
    const source = String(file.value ?? '');
    let words = 0;

    const raw = (node) => {
      const p = node.position;
      return p ? source.slice(p.start.offset, p.end.offset) : '';
    };

    const visit = (node) => {
      if (!node.children) return;
      const next = [];
      for (const child of node.children) {
        if (child.type === 'containerDirective' && child.name === 'figure') {
          next.push(figure(child, file));
          continue;
        }
        if (child.type === 'containerDirective' && child.name === 'aside') {
          child.data = { hName: 'aside', hProperties: { className: ['note', 'unnumbered'] } };
          visit(child);
          next.push(child);
          continue;
        }
        if (child.type === 'textDirective' || child.type === 'leafDirective' || child.type === 'containerDirective') {
          // Not one of ours: put the author's text back exactly as written.
          const text = { type: 'text', value: raw(child) };
          next.push(child.type === 'textDirective' ? text : { type: 'paragraph', children: [text] });
          continue;
        }
        if (child.type === 'text') {
          words += (child.value.match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) || []).length;
          next.push(...cites(child));
          continue;
        }
        visit(child);
        next.push(child);
      }
      node.children = next;
    };
    visit(tree);

    file.data.astro ??= {};
    file.data.astro.frontmatter ??= {};
    file.data.astro.frontmatter.minutes = Math.max(1, Math.round(words / 230));
  };
}

function cites(node) {
  const out = [];
  let last = 0;
  for (const m of node.value.matchAll(CITE)) {
    if (m.index > last) out.push({ type: 'text', value: node.value.slice(last, m.index) });
    const keys = m[1].split(';').map((k) => k.trim().slice(1));
    out.push({ type: 'cite', data: { hName: 'span', hProperties: { dataCite: keys.join(' ') } }, children: [] });
    last = m.index + m[0].length;
  }
  if (!out.length) return [node];
  if (last < node.value.length) out.push({ type: 'text', value: node.value.slice(last) });
  // "word [@key]" reads better with the marker tight against the word
  for (let i = 1; i < out.length; i++) {
    if (out[i].type === 'cite' && out[i - 1].type === 'text') out[i - 1].value = out[i - 1].value.replace(/[ \t]+$/, '');
  }
  return out.filter((n) => n.type !== 'text' || n.value);
}

function figure(node, file) {
  const a = node.attributes || {};
  const classes = (a.class || '').split(/\s+/).filter(Boolean);
  const kind = classes.includes('margin') ? 'margin' : classes.includes('wide') ? 'wide' : null;
  if (!a.src) warn(file, ':::figure needs a src attribute');
  if (a.alt == null) warn(file, ':::figure "' + (a.src || '') + '" needs an alt attribute (use alt="" only for pure decoration)');
  const caption = [];
  for (const child of node.children) {
    if (child.type === 'paragraph') {
      if (caption.length) caption.push({ type: 'text', value: ' ' });
      caption.push(...child.children);
    }
  }
  const children = [{ type: 'image', url: a.src || '', alt: a.alt || '', title: null }];
  if (caption.length) children.push({ type: 'paragraph', data: { hName: 'figcaption' }, children: caption });
  return {
    type: 'containerDirective',
    name: 'figure',
    data: { hName: 'figure', hProperties: { className: ['fig', ...(kind ? [kind] : [])] } },
    children,
  };
}

// ---------------------------------------------------------------- rehype

const el = (tagName, properties = {}, children = []) => ({ type: 'element', tagName, properties, children });
const text = (value) => ({ type: 'text', value });
const has = (node, cls) => node.type === 'element' && [].concat(node.properties?.className || []).includes(cls);
const blank = (node) => node.type === 'text' && !node.value.trim();

const plainText = (node) => {
  if (!node) return '';
  if (Array.isArray(node)) return node.map(plainText).join('');
  if (node.type === 'text') return node.value;
  if (node.type === 'element') {
    if (has(node, 'katex-mathml') || has(node, 'on-paper')) return '';
    return plainText(node.children || []);
  }
  return '';
};

const teaserText = (s, max = 78) => {
  const clean = s.replace(/\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  const cut = clean.slice(0, max);
  const sp = cut.lastIndexOf(' ');
  return (sp > 40 ? cut.slice(0, sp) : cut).replace(/[.,;:—–-]+$/, '') + '…';
};

const FOLD_NOTE_CHARS = 220;
const XREF_RE = /\b(Sections?\s+\d+(?:\s*(?:and|[–-])\s*\d+)?|Appendices\s+[A-E]\s*(?:and|[–-])\s*[A-E]|Appendix\s+[A-E]|Figures?\s+\d+(?:\s*(?:and|[–-])\s*\d+)?|Tables?\s+(?:[1-9]|[A-E]\d+)(?:\s*(?:and|[–-])\s*(?:[1-9]|[A-E]\d+))?)\b/g;

const headingSlug = (s) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');

export function rehypeArticle(options = {}) {
  const bibPath = path.resolve(options.bibliography || 'refs.bib');

  return (tree, file) => {
    const bib = fs.existsSync(bibPath) ? parseBibtex(fs.readFileSync(bibPath, 'utf8')) : {};

    // 1. Lift the footnote definitions out of the section GFM puts at the end.
    const footnotes = new Map();
    tree.children = tree.children.filter((node) => {
      if (node.type !== 'element' || node.tagName !== 'section' || node.properties?.dataFootnotes == null) return true;
      walk(node, (li) => {
        if (li.type !== 'element' || li.tagName !== 'li' || !li.properties?.id) return;
        const content = [];
        for (const block of li.children) {
          if (block.type !== 'element') continue;
          if (content.length) content.push(text(' '));
          content.push(...(block.tagName === 'p' ? block.children : [block]));
        }
        const kept = content.filter((c) => !(c.type === 'element' && c.properties?.dataFootnoteBackref != null));
        while (kept.length && blank(kept[kept.length - 1])) kept.pop();
        const tail = kept[kept.length - 1];
        if (tail?.type === 'text') tail.value = tail.value.trimEnd();
        footnotes.set(String(li.properties.id), kept);
        return false;
      });
      return false;
    });

    // 2. A paragraph that holds only an image is a figure.
    tree.children = tree.children.map((node) => {
      if (node.type !== 'element' || node.tagName !== 'p') return node;
      const solid = node.children.filter((c) => !blank(c));
      if (solid.length !== 1 || solid[0].type !== 'element' || solid[0].tagName !== 'img') return node;
      const img = solid[0];
      const caption = img.properties.title;
      delete img.properties.title;
      return el('figure', { className: ['fig'] }, caption ? [img, el('figcaption', {}, [text(String(caption))])] : [img]);
    });

    // 3. Number the notes in reading order and seat each one beside its block.
    let n = 0;
    let figures = 0;
    const seen = new Map();   // footnote id or citation key -> number already given
    const out = [];
    let lastBlock = -1;

    const note = (num, content, foldable = false) => {
      const txt = plainText(content).replace(/\s+/g, ' ').trim();
      if (foldable && txt.length > FOLD_NOTE_CHARS) {
        return el('details', { className: ['note', 'note-fold'], id: 'note-' + num }, [
          el('summary', { className: ['note-sum'] }, [
            el('span', { className: ['note-n'] }, [text(String(num))]),
            text(' '),
            el('span', { className: ['note-teaser'] }, [text(teaserText(txt))]),
            text(' '),
            el('span', { className: ['note-toggle'], ariaHidden: 'true' }, []),
          ]),
          el('div', { className: ['note-body'] }, content),
        ]);
      }
      return el('p', { className: ['note'], id: 'note-' + num }, [el('span', { className: ['note-n'] }, [text(String(num))]), text(' '), ...content]);
    };
    const ref = (num, first) =>
      el('a', { className: ['ref'], href: '#note-' + num, ...(first ? { id: 'ref-' + num } : {}), ariaLabel: 'Note ' + num }, [text(String(num))]);

    const reference = (key) => {
      const entry = bib[key];
      if (!entry) {
        warn(file, 'no entry "' + key + '" in ' + path.basename(bibPath));
        return [el('span', { className: ['missing'] }, [text('Missing reference: ' + key)])];
      }
      return formatReference(entry).map((part) =>
        typeof part === 'string' ? text(part)
          : part.em != null ? el('i', {}, [text(part.em)])
          : el('a', { href: part.href }, [el('span', { className: ['on-screen'] }, [text(part.text)]), el('span', { className: ['on-paper'] }, [text(part.print)])]));
    };

    const seat = (node) => {
      // returns the notes this block calls for, rewriting its markers in place
      const notes = [];
      walk(node, (child, parent, index) => {
        if (child.type !== 'element') return;
        if (child.tagName === 'sup' && child.children.some((c) => c.type === 'element' && c.properties?.dataFootnoteRef != null)) {
          const a = child.children.find((c) => c.type === 'element' && c.properties?.dataFootnoteRef != null);
          const id = String(a.properties.href || '').replace(/^#/, '');
          let num = seen.get(id);
          const first = num == null;
          if (first) {
            num = ++n;
            seen.set(id, num);
            notes.push(note(num, footnotes.get(id) || [text('Missing footnote.')], true));
          }
          parent.children[index] = ref(num, first);
          return false;
        }
        if (child.tagName === 'span' && child.properties?.dataCite != null) {
          const keys = String(child.properties.dataCite).split(/\s+/).filter(Boolean);
          const fresh = keys.filter((k) => !seen.has('cite:' + k));
          const num = ++n;
          const content = [];
          keys.forEach((key, i) => {
            if (i) content.push(el('br'));
            if (fresh.includes(key)) {
              seen.set('cite:' + key, num);
              content.push(...reference(key));
            } else {
              const entry = bib[key];
              content.push(text((entry ? shortRef(entry) : key) + ', see note ' + seen.get('cite:' + key) + '.'));
            }
          });
          notes.push(note(num, content));
          parent.children[index] = ref(num, true);
          return false;
        }
      });
      return notes;
    };

    const toRow = (i) => {
      if (!has(out[i], 'row')) out[i] = el('div', { className: ['row'] }, [el('div', { className: ['col'] }, [out[i]]), el('div', { className: ['margin'] }, [])]);
      return out[i].children[1];
    };

    for (let idx = 0; idx < tree.children.length; idx++) {
      let node = tree.children[idx];
      if (node.type !== 'element') { out.push(node); continue; }

      if (node.tagName === 'figure' && has(node, 'fig')) {
        const cap = node.children.find((c) => c.type === 'element' && c.tagName === 'figcaption');
        if (cap && !has(node, 'margin')) {
          figures++;
          node.properties.id = 'fig-' + figures;
          cap.children.unshift(el('span', { className: ['fig-n'] }, [text('Figure ' + figures + '.')]), text(' '));
        }
      }
      if (node.tagName === 'table') {
        let tableId = null;
        for (let j = idx + 1; j < tree.children.length; j++) {
          const sib = tree.children[j];
          if (sib.type === 'element') {
            if (sib.tagName === 'p') {
              const m = /^\s*Table\s+([1-9]|[A-E]\d+)\s*:/.exec(plainText(sib));
              if (m) tableId = 'table-' + m[1].toLowerCase();
            }
            break;
          }
        }
        node = el('div', { className: ['table-wrap'], ...(tableId ? { id: tableId } : {}) }, [node]);
      }

      const floating = has(node, 'unnumbered') || (has(node, 'fig') && has(node, 'margin'));
      const notes = seat(node);

      if (floating) {
        if (lastBlock < 0) { out.push(el('p', {}, [])); lastBlock = out.length - 1; }
        toRow(lastBlock).children.push(node, ...notes);
        continue;
      }
      out.push(node);
      const isHeading = /^h[1-6]$/.test(node.tagName);
      if (!isHeading) lastBlock = out.length - 1;
      if (notes.length) toRow(out.length - 1).children.push(...notes);
    }

    // 4. End mark after the last paragraph of running text.
    for (let i = out.length - 1; i >= 0; i--) {
      const node = out[i];
      if (node.type !== 'element') continue;
      const p = node.tagName === 'p' ? node : has(node, 'row') && node.children[0].children[0]?.tagName === 'p' ? node.children[0].children[0] : null;
      if (!p) continue;
      p.children.push(el('span', { className: ['endmark'], ariaHidden: 'true' }, []));
      break;
    }

    // 5. Turn internal section, appendix, figure and table references into jump lozenges.
    const targets = new Map();
    for (const root of out) {
      walk(root, (node) => {
        if (node.type !== 'element') return;
        if (node.tagName === 'h2') {
          const title = plainText(node);
          const m = /^\s*(\d+)\.\s/.exec(title);
          if (m) {
            const id = node.properties?.id ? String(node.properties.id) : headingSlug(title);
            targets.set('section ' + m[1], { href: '#' + id, cls: 'xref-sec' });
          }
        } else if (node.tagName === 'h3') {
          const title = plainText(node);
          const m = /^\s*Appendix\s+([A-E])\s*:/i.exec(title);
          if (m) {
            const id = node.properties?.id ? String(node.properties.id) : headingSlug(title);
            targets.set('appendix ' + m[1].toLowerCase(), { href: '#' + id, cls: 'xref-app' });
          }
        } else if (node.tagName === 'figure' && node.properties?.id) {
          const id = String(node.properties.id);
          const m = /^fig-(\d+)$/.exec(id);
          if (m) targets.set('figure ' + m[1], { href: '#' + id, cls: 'xref-fig' });
        } else if (node.tagName === 'div' && has(node, 'table-wrap') && node.properties?.id) {
          const id = String(node.properties.id);
          const m = /^table-([a-e]?\d+)$/.exec(id);
          if (m) targets.set('table ' + m[1], { href: '#' + id, cls: 'xref-tbl' });
        }
      });
    }

    const makeXref = (kind, code, label, selfId) => {
      const info = targets.get(kind + ' ' + code.toLowerCase());
      if (!info || (selfId && info.href === '#' + selfId)) return text(label);
      return el('a', { className: ['xref', info.cls], href: info.href }, [text(label)]);
    };

    const linkText = (value, selfId) => {
      const nodes = [];
      let last = 0;
      let changed = false;
      XREF_RE.lastIndex = 0;
      for (const m of value.matchAll(XREF_RE)) {
        const full = m[0];
        const after = value.slice(m.index + full.length);
        // Do not turn a table caption's own leading "Table X:" label into a link.
        if (/^Table\s+(?:[1-9]|[A-E]\d+)$/.test(full) && /^\s*:/.test(after)) continue;

        let pieces = null;
        let sub;
        if ((sub = /^(Sections?)\s+(\d+)(\s*(?:and|[–-])\s*)(\d+)$/.exec(full))) {
          pieces = [
            makeXref('section', sub[2], 'Section ' + sub[2], selfId),
            text(sub[3]),
            makeXref('section', sub[4], 'Section ' + sub[4], selfId),
          ];
        } else if ((sub = /^Appendices\s+([A-E])(\s*(?:and|[–-])\s*)([A-E])$/.exec(full))) {
          pieces = [
            makeXref('appendix', sub[1], 'Appendix ' + sub[1], selfId),
            text(sub[2]),
            makeXref('appendix', sub[3], 'Appendix ' + sub[3], selfId),
          ];
        } else if ((sub = /^(Figures?)\s+(\d+)(\s*(?:and|[–-])\s*)(\d+)$/.exec(full))) {
          pieces = [
            makeXref('figure', sub[2], 'Figure ' + sub[2], selfId),
            text(sub[3]),
            makeXref('figure', sub[4], 'Figure ' + sub[4], selfId),
          ];
        } else if ((sub = /^(Tables?)\s+([1-9]|[A-E]\d+)(\s*(?:and|[–-])\s*)([1-9]|[A-E]\d+)$/.exec(full))) {
          pieces = [
            makeXref('table', sub[2], 'Table ' + sub[2], selfId),
            text(sub[3]),
            makeXref('table', sub[4], 'Table ' + sub[4], selfId),
          ];
        } else if ((sub = /^Section\s+(\d+)$/.exec(full))) {
          pieces = [makeXref('section', sub[1], full, selfId)];
        } else if ((sub = /^Appendix\s+([A-E])$/.exec(full))) {
          pieces = [makeXref('appendix', sub[1], full, selfId)];
        } else if ((sub = /^Figure\s+(\d+)$/.exec(full))) {
          pieces = [makeXref('figure', sub[1], full, selfId)];
        } else if ((sub = /^Table\s+([1-9]|[A-E]\d+)$/.exec(full))) {
          pieces = [makeXref('table', sub[1], full, selfId)];
        }

        if (!pieces || pieces.every((p) => p.type === 'text')) continue;
        if (m.index > last) nodes.push(text(value.slice(last, m.index)));
        nodes.push(...pieces);
        last = m.index + full.length;
        changed = true;
      }
      if (!changed) return null;
      if (last < value.length) nodes.push(text(value.slice(last)));
      return nodes;
    };

    const SKIP_TAGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'code', 'pre', 'summary']);
    const applyXrefs = (node, selfId = null) => {
      if (!node || node.type !== 'element' || !node.children) return;
      if (SKIP_TAGS.has(node.tagName) || has(node, 'katex') || has(node, 'fig-n')) return;
      const nextSelf = (node.tagName === 'figure' && node.properties?.id) ? String(node.properties.id) : selfId;
      const next = [];
      for (const child of node.children) {
        if (child.type === 'text') {
          const linked = linkText(child.value, nextSelf);
          if (linked) next.push(...linked);
          else next.push(child);
        } else {
          applyXrefs(child, nextSelf);
          next.push(child);
        }
      }
      node.children = next;
    };
    for (const root of out) applyXrefs(root);

    tree.children = out;
  };
}
