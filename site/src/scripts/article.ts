// Small enhancements; the article reads fine without them.

const copy = (value: string) => navigator.clipboard?.writeText(value).catch(() => {});
const here = () => location.origin + location.pathname;

// Section headings: a § appears on hover and copies the section link.
const headings = [...document.querySelectorAll<HTMLElement>('.body h2[id]')];
for (const h of headings) {
  const anchor = document.createElement('button');
  anchor.type = 'button';
  anchor.className = 'anc';
  anchor.textContent = '§';
  anchor.setAttribute('aria-label', 'Copy link to this section');
  const note = document.createElement('span');
  note.className = 'copied';
  note.setAttribute('aria-live', 'polite');
  let timer = 0;
  anchor.addEventListener('click', () => {
    copy(here() + '#' + h.id);
    history.replaceState(null, '', '#' + h.id);
    note.textContent = 'link copied';
    note.classList.add('is-on');
    clearTimeout(timer);
    timer = window.setTimeout(() => note.classList.remove('is-on'), 1600);
  });
  h.prepend(anchor);
  h.append(note);
}

// Cite and copy-link actions in the header.
const citeBtn = document.querySelector<HTMLButtonElement>('[data-cite]');
const citeBox = document.getElementById('citebox');
citeBtn?.addEventListener('click', () => {
  if (!citeBox) return;
  citeBox.hidden = !citeBox.hidden;
  citeBtn.textContent = citeBox.hidden ? 'Cite this post' : 'Hide citation';
  citeBtn.setAttribute('aria-expanded', String(!citeBox.hidden));
});

const copyBtn = document.querySelector<HTMLButtonElement>('[data-copy]');
copyBtn?.addEventListener('click', () => {
  copy(copyBtn.dataset.copy || here());
  copyBtn.textContent = 'Link copied';
  window.setTimeout(() => (copyBtn.textContent = 'Copy link'), 1600);
});

// A note lights up when its number is hovered, focused or followed.
// Folded margin notes open automatically when their superscript is clicked or targeted.
const notes = [...document.querySelectorAll<HTMLElement>('.note[id]')];
const openFold = (el: HTMLElement | null) => {
  if (el instanceof HTMLDetailsElement) el.open = true;
};
for (const ref of document.querySelectorAll<HTMLAnchorElement>('a.ref')) {
  const target = document.getElementById(ref.hash.slice(1));
  if (!target) continue;
  const light = () => {
    for (const n of notes) n.classList.toggle('is-on', n === target);
  };
  ref.addEventListener('mouseenter', light);
  ref.addEventListener('focus', light);
  ref.addEventListener('click', (e) => {
    openFold(target);
    light();
    // beside the text there is nowhere to jump to; stacked below it, scroll the note into view
    const beside = Math.abs(target.getBoundingClientRect().top - ref.getBoundingClientRect().top) < window.innerHeight * 0.6;
    if (beside) e.preventDefault();
  });
}

const openHashNote = () => {
  if (!location.hash) return;
  const target = document.getElementById(location.hash.slice(1));
  if (target?.classList.contains('note-fold')) openFold(target);
};
openHashNote();
addEventListener('hashchange', openHashNote);

// Expand folded margin notes automatically when printing, then restore afterwards.
let openedForPrint: HTMLDetailsElement[] = [];
addEventListener('beforeprint', () => {
  openedForPrint = [];
  for (const d of document.querySelectorAll<HTMLDetailsElement>('details.note-fold:not([open])')) {
    d.open = true;
    openedForPrint.push(d);
  }
});
addEventListener('afterprint', () => {
  for (const d of openedForPrint) d.open = false;
  openedForPrint = [];
});

// The contents column marks the section being read.
const links = new Map(
  [...document.querySelectorAll<HTMLAnchorElement>('.toc a')].map((a) => [decodeURIComponent(a.hash.slice(1)), a]),
);
if (links.size && headings.length) {
  let queued = false;
  const mark = () => {
    queued = false;
    let current = headings[0];
    for (const h of headings) if (h.getBoundingClientRect().top <= 140) current = h;
    const atEnd = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
    if (atEnd) current = headings[headings.length - 1];
    for (const [id, a] of links) {
      if (id === current.id) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  };
  const queue = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(mark);
    }
  };
  addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', queue);
  mark();
}
