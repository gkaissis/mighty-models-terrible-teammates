// Front matter helpers for the article layout.

const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
  'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];

const LICENSES = {
  'cc by 4.0': 'https://creativecommons.org/licenses/by/4.0/',
  'cc by-sa 4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
  'cc by-nc 4.0': 'https://creativecommons.org/licenses/by-nc/4.0/',
  'cc by-nc-sa 4.0': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
  'cc by-nd 4.0': 'https://creativecommons.org/licenses/by-nd/4.0/',
  'cc0 1.0': 'https://creativecommons.org/publicdomain/zero/1.0/',
  'mit': 'https://opensource.org/license/mit',
  'apache-2.0': 'https://www.apache.org/licenses/LICENSE-2.0',
};

export function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(String(value));
  return Number.isNaN(d.getTime()) ? null : d;
}

/** 22 November 2026 */
export function longDate(value) {
  const d = toDate(value);
  return d ? new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d) : '';
}

export const isoDate = (value) => toDate(value)?.toISOString().slice(0, 10) ?? '';

/** "a four-minute read" */
export function readingTime(minutes) {
  if (!minutes) return '';
  const word = WORDS[minutes] ?? String(minutes);
  const article = /^(eight|eleven|eighteen|8|11|18|8\d)/.test(word) ? 'an' : 'a';
  return article + ' ' + word + '-minute read';
}

export function license(value) {
  if (!value) return null;
  if (typeof value === 'object') return { name: value.name, url: value.url };
  return { name: String(value), url: LICENSES[String(value).toLowerCase()] ?? null };
}

/** Normalises authors and affiliations; affiliations are numbered only when there is more than one. */
export function people(frontmatter) {
  const affiliations = (frontmatter.affiliations ?? []).map((a, i) =>
    typeof a === 'string' ? { id: a, name: a, n: i + 1 } : { id: a.id ?? a.name, name: a.name, url: a.url, n: i + 1 });
  const byId = new Map(affiliations.map((a) => [a.id, a]));
  const authors = (frontmatter.authors ?? []).map((a) => {
    const author = typeof a === 'string' ? { name: a } : { ...a };
    const refs = [].concat(author.affiliation ?? author.affiliations ?? []);
    author.marks = refs.map((id) => {
      if (!byId.has(id)) {
        const added = { id, name: id, n: affiliations.length + 1 };
        affiliations.push(added);
        byId.set(id, added);
      }
      return byId.get(id).n;
    });
    author.href = author.url ?? (author.orcid ? 'https://orcid.org/' + author.orcid : null);
    return author;
  });
  return {
    authors,
    affiliations,
    numbered: affiliations.length > 1,
    anyEqual: authors.some((a) => a.equal),
    anyCorresponding: authors.some((a) => a.corresponding),
  };
}

function familyGiven(author) {
  if (author.family) return [author.family, author.given ?? ''];
  const parts = author.name.trim().split(/\s+/);
  return [parts[parts.length - 1], parts.slice(0, -1).join(' ')];
}

/** The post's own BibTeX entry, unless the front matter supplies one. */
export function bibtex(frontmatter, authors, url) {
  if (frontmatter.bibtex) return String(frontmatter.bibtex).trim();
  const year = toDate(frontmatter.date)?.getUTCFullYear() ?? '';
  const first = authors[0] ? familyGiven(authors[0])[0] : 'post';
  const word = (String(frontmatter.title).toLowerCase().match(/[a-z]{4,}/) ?? ['post'])[0];
  const key = (first + year + word).toLowerCase().normalize('NFKD').replace(/[^a-z0-9]/g, '');
  const rows = [
    ['author', authors.map((a) => familyGiven(a).filter(Boolean).join(', ')).join(' and ')],
    ['title', frontmatter.title],
    ['year', year],
    ['url', url],
    ['doi', frontmatter.doi],
  ].filter(([, v]) => v);
  const pad = Math.max(...rows.map(([k]) => k.length));
  return '@misc{' + key + ',\n' + rows.map(([k, v]) => '  ' + k.padEnd(pad) + ' = {' + v + '}').join(',\n') + '\n}';
}
