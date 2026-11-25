// A small BibTeX reader: enough for the usual @article / @inproceedings / @book / @misc entries.
// Returns { key: { type, key, fields } } with field names lower-cased and values cleaned of TeX.

const ACCENTS = {
  '"': { a: 'ä', o: 'ö', u: 'ü', e: 'ë', i: 'ï', A: 'Ä', O: 'Ö', U: 'Ü' },
  "'": { a: 'á', e: 'é', i: 'í', o: 'ó', u: 'ú', y: 'ý', c: 'ć', n: 'ń', s: 'ś', z: 'ź', A: 'Á', E: 'É', O: 'Ó', S: 'Ś' },
  '`': { a: 'à', e: 'è', i: 'ì', o: 'ò', u: 'ù', A: 'À', E: 'È' },
  '^': { a: 'â', e: 'ê', i: 'î', o: 'ô', u: 'û' },
  '~': { a: 'ã', n: 'ñ', o: 'õ', N: 'Ñ' },
  c: { c: 'ç', s: 'ş', C: 'Ç' },
  v: { c: 'č', s: 'š', z: 'ž', r: 'ř', e: 'ě', C: 'Č', S: 'Š', Z: 'Ž' },
  u: { g: 'ğ', a: 'ă' },
  H: { o: 'ő', u: 'ű' },
  k: { a: 'ą', e: 'ę' },
};
const SYMBOLS = { ss: 'ß', o: 'ø', O: 'Ø', l: 'ł', L: 'Ł', ae: 'æ', AE: 'Æ', aa: 'å', AA: 'Å', i: 'ı', '&': '&', '%': '%', _: '_', '#': '#', $: '$' };

export function untex(s) {
  return String(s)
    .replace(/\\(["'`^~])\s*\{?\\?([A-Za-z])\}?/g, (_, a, c) => ACCENTS[a]?.[c] ?? c)
    .replace(/\\([cvuHk])\s*\{([A-Za-z])\}/g, (_, a, c) => ACCENTS[a]?.[c] ?? c)
    .replace(/\\(ss|ae|AE|aa|AA|[oOlLi])(?![A-Za-z])\s*/g, (_, c) => SYMBOLS[c])
    .replace(/\\([&%_#$])/g, (_, c) => SYMBOLS[c])
    .replace(/\\(?:emph|textit|textbf|textsc|texttt|mathrm|url)\s*\{([^{}]*)\}/g, '$1')
    .replace(/---/g, '—')
    .replace(/--/g, '–')
    .replace(/~/g, ' ')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function parseBibtex(src) {
  const out = {};
  let i = 0;
  const n = src.length;
  const ws = () => { while (i < n && /\s/.test(src[i])) i++; };
  const value = () => {
    // one or more pieces joined by #
    let v = '';
    for (;;) {
      ws();
      if (src[i] === '{') {
        let depth = 0;
        const start = i + 1;
        for (; i < n; i++) {
          if (src[i] === '{') depth++;
          else if (src[i] === '}' && --depth === 0) break;
        }
        v += src.slice(start, i);
        i++;
      } else if (src[i] === '"') {
        let depth = 0;
        const start = ++i;
        for (; i < n; i++) {
          if (src[i] === '{') depth++;
          else if (src[i] === '}') depth--;
          else if (src[i] === '"' && depth === 0) break;
        }
        v += src.slice(start, i);
        i++;
      } else {
        const m = /^[^\s,#}]+/.exec(src.slice(i));
        if (!m) break;
        v += m[0];
        i += m[0].length;
      }
      ws();
      if (src[i] === '#') { i++; continue; }
      break;
    }
    return v;
  };

  while (i < n) {
    const at = src.indexOf('@', i);
    if (at < 0) break;
    i = at + 1;
    const head = /^([A-Za-z]+)\s*\{\s*/.exec(src.slice(i));
    if (!head) continue;
    const type = head[1].toLowerCase();
    i += head[0].length;
    if (type === 'comment' || type === 'preamble' || type === 'string') continue;
    const keyEnd = src.indexOf(',', i);
    if (keyEnd < 0) break;
    const key = src.slice(i, keyEnd).trim();
    i = keyEnd + 1;
    const fields = {};
    for (;;) {
      ws();
      if (i >= n || src[i] === '}') { i++; break; }
      const name = /^([A-Za-z][\w-]*)\s*=/.exec(src.slice(i));
      if (!name) { i++; continue; }
      i += name[0].length;
      fields[name[1].toLowerCase()] = value();
      ws();
      if (src[i] === ',') i++;
    }
    out[key] = { type, key, fields };
  }
  return out;
}

/** "Last, First" or "First Last" -> { first, last } */
export function parseNames(s) {
  if (!s) return [];
  return s.split(/\s+and\s+/i).map((raw) => {
    const name = untex(raw);
    if (name.includes(',')) {
      const [last, first = ''] = name.split(',').map((p) => p.trim());
      return { first, last };
    }
    const parts = name.split(' ');
    if (parts.length === 1) return { first: '', last: parts[0] };
    return { first: parts.slice(0, -1).join(' '), last: parts[parts.length - 1] };
  });
}

const full = (p) => [p.first, p.last].filter(Boolean).join(' ');

export function authorList(entry) {
  const people = parseNames(entry.fields.author || entry.fields.editor);
  if (!people.length) return '';
  if (people.length === 1) return full(people[0]);
  if (people.length === 2) return full(people[0]) + ' and ' + full(people[1]);
  if (people.length === 3) return full(people[0]) + ', ' + full(people[1]) + ' and ' + full(people[2]);
  return full(people[0]) + ' et al.';
}

export function shortRef(entry) {
  const people = parseNames(entry.fields.author || entry.fields.editor);
  const who = !people.length ? untex(entry.fields.title || entry.key)
    : people.length === 1 ? people[0].last
    : people.length === 2 ? people[0].last + ' and ' + people[1].last
    : people[0].last + ' et al.';
  return [who, entry.fields.year].filter(Boolean).join(' ');
}

/**
 * One reference as a list of parts: strings, { em: 'text' } for italics, { href, text, print } for links
 * (`print` is what a paper copy shows in place of the link text).
 * Books:    Author, <i>Title</i>. Publisher, Year. doi
 * Articles: Author, "Title". <i>Journal</i>, Year. doi
 */
export function formatReference(entry) {
  const f = entry.fields;
  const parts = [];
  const authors = authorList(entry);
  const title = untex(f.title || '');
  const container = untex(f.journal || f.booktitle || f.howpublished || '');
  const isWhole = ['book', 'phdthesis', 'mastersthesis', 'techreport', 'manual', 'booklet'].includes(entry.type) || !container;
  if (authors) parts.push(authors + ', ');
  if (isWhole) parts.push({ em: title }, '. ');
  else parts.push('“' + title + '”. ', { em: container }, ', ');
  const where = isWhole ? untex(f.publisher || f.school || f.institution || f.organization || f.howpublished || '') : '';
  const tail = [where, f.year ? untex(f.year) : ''].filter(Boolean).join(', ');
  if (tail) parts.push(tail + '. ');
  else if (!isWhole) parts[parts.length - 1] = '. ';
  const doi = f.doi ? untex(f.doi).replace(/^https?:\/\/(dx\.)?doi\.org\//, '') : '';
  const eprint = f.eprint && /arxiv/i.test(f.archiveprefix || f.eprinttype || 'arxiv') ? untex(f.eprint) : '';
  if (doi) parts.push({ href: 'https://doi.org/' + doi, text: 'doi', print: 'doi:' + doi });
  else if (eprint) parts.push({ href: 'https://arxiv.org/abs/' + eprint, text: 'arXiv', print: 'arXiv:' + eprint });
  else if (f.url) parts.push({ href: untex(f.url), text: 'link', print: untex(f.url).replace(/^https?:\/\//, '') });
  else if (typeof parts[parts.length - 1] === 'string') parts[parts.length - 1] = parts[parts.length - 1].trimEnd();
  return parts;
}
