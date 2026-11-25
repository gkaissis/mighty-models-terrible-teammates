// Small progressive enhancements; the article reads cleanly without JavaScript.

const copy = (value: string) => navigator.clipboard?.writeText(value).catch(() => {});
const here = () => location.origin + location.pathname;
const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

// Section and subsection headings: a § appears on hover and copies a link to that heading.
const anchorHeadings = [...document.querySelectorAll<HTMLElement>('.body h2[id], .body h3[id]')].filter(
  (h) => !h.closest('summary'),
);
for (const h of anchorHeadings) {
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

// Header actions: cite toggle, BibTeX copy button, copy canonical link.
const citeBtn = document.querySelector<HTMLButtonElement>('[data-cite]');
const citeBox = document.getElementById('citebox');
citeBtn?.addEventListener('click', () => {
  if (!citeBox) return;
  citeBox.hidden = !citeBox.hidden;
  citeBtn.textContent = citeBox.hidden ? 'Cite this post' : 'Hide citation';
  citeBtn.setAttribute('aria-expanded', String(!citeBox.hidden));
});

const copyCiteBtn = document.querySelector<HTMLButtonElement>('[data-copy-cite]');
if (copyCiteBtn && citeBox) {
  let timer = 0;
  copyCiteBtn.addEventListener('click', () => {
    const text = citeBox.querySelector<HTMLElement>('pre.citebox')?.textContent?.trim() ?? '';
    copy(text);
    copyCiteBtn.textContent = 'Copied';
    clearTimeout(timer);
    timer = window.setTimeout(() => (copyCiteBtn.textContent = 'Copy BibTeX'), 1600);
  });
}

const copyBtn = document.querySelector<HTMLButtonElement>('[data-copy]');
if (copyBtn) {
  let timer = 0;
  copyBtn.addEventListener('click', () => {
    copy(copyBtn.dataset.copy || here());
    copyBtn.textContent = 'Link copied';
    clearTimeout(timer);
    timer = window.setTimeout(() => (copyBtn.textContent = 'Copy link'), 1600);
  });
}

// Helper: open a <details> target and every closed ancestor <details>.
const openAncestors = (el: HTMLElement | null): boolean => {
  let opened = false;
  let cur: HTMLElement | null = el;
  while (cur) {
    if (cur instanceof HTMLDetailsElement && !cur.open) {
      cur.open = true;
      opened = true;
    }
    cur = cur.parentElement;
  }
  return opened;
};

// Margin notes: soft highlight on hover/focus that clears cleanly on mouseleave/blur unless pinned by click.
const notes = [...document.querySelectorAll<HTMLElement>('.note[id]')];
let pinnedNote: HTMLElement | null = null;
const highlightNote = (target: HTMLElement | null) => {
  for (const n of notes) n.classList.toggle('is-on', n === target);
};

for (const ref of document.querySelectorAll<HTMLAnchorElement>('a.ref')) {
  const target = document.getElementById(ref.hash.slice(1));
  if (!target) continue;
  ref.addEventListener('mouseenter', () => highlightNote(target));
  ref.addEventListener('focus', () => highlightNote(target));
  ref.addEventListener('mouseleave', () => highlightNote(pinnedNote));
  ref.addEventListener('blur', () => highlightNote(pinnedNote));
  ref.addEventListener('click', (e) => {
    openAncestors(target);
    pinnedNote = target;
    highlightNote(target);
    const beside = Math.abs(target.getBoundingClientRect().top - ref.getBoundingClientRect().top) < window.innerHeight * 0.6;
    if (beside) e.preventDefault();
  });
}

document.addEventListener('click', (e) => {
  if (!pinnedNote) return;
  const t = e.target as HTMLElement | null;
  if (t?.closest('a.ref') || t?.closest('.note')) return;
  pinnedNote = null;
  highlightNote(null);
});

// Internal anchors (TOC, cross-references, hash links): open any folded <details> ancestors before scrolling.
const revealHashTarget = (scroll = false) => {
  if (!location.hash || location.hash === '#') return;
  const id = decodeURIComponent(location.hash.slice(1));
  const target = document.getElementById(id);
  if (!target) return;
  const opened = openAncestors(target);
  if (target.classList.contains('note')) {
    pinnedNote = target;
    highlightNote(target);
  }
  if (opened || scroll) {
    requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
    });
  }
};

for (const link of document.querySelectorAll<HTMLAnchorElement>('a[href^="#"]')) {
  if (link.classList.contains('ref')) continue;
  link.addEventListener('click', (e) => {
    const hash = link.getAttribute('href');
    if (!hash || hash === '#') return;
    const id = decodeURIComponent(hash.slice(1));
    const target = document.getElementById(id);
    if (!target) return;
    const hasClosedAncestor = Boolean(
      (target instanceof HTMLDetailsElement && !target.open) || target.closest('details:not([open])'),
    );
    if (hasClosedAncestor) {
      e.preventDefault();
      openAncestors(target);
      history.pushState(null, '', hash);
      requestAnimationFrame(() => {
        target.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
      });
    }
  });
}

revealHashTarget(false);
addEventListener('hashchange', () => revealHashTarget(true));

// Verbatim codeboxes: wrap toggle and copy button.
for (const box of document.querySelectorAll<HTMLElement>('.codebox')) {
  const wrapBtn = box.querySelector<HTMLButtonElement>('[data-wrap-toggle]');
  wrapBtn?.addEventListener('click', () => {
    const wrapped = !box.classList.contains('is-wrapped');
    box.classList.toggle('is-wrapped', wrapped);
    box.dataset.wrap = wrapped ? 'on' : 'off';
    wrapBtn.setAttribute('aria-pressed', String(wrapped));
    wrapBtn.textContent = wrapped ? 'Wrap: on' : 'Wrap: off';
  });

  const copyCodeBtn = box.querySelector<HTMLButtonElement>('[data-copy-code]');
  if (copyCodeBtn) {
    let timer = 0;
    copyCodeBtn.addEventListener('click', () => {
      const pre = box.querySelector<HTMLElement>('pre');
      const lines = pre ? [...pre.querySelectorAll<HTMLElement>('.line')] : [];
      const payload = lines.length
        ? lines.map((l) => l.textContent ?? '').join('\n')
        : pre?.textContent ?? '';
      copy(payload);
      copyCodeBtn.textContent = 'Copied';
      clearTimeout(timer);
      timer = window.setTimeout(() => (copyCodeBtn.textContent = 'Copy'), 1600);
    });
  }
}

// Click-to-enlarge figure lightbox (native <dialog> with full HTML figcaption).
const lightbox = document.getElementById('lightbox') as HTMLDialogElement | null;
if (lightbox && typeof lightbox.showModal === 'function') {
  const lbImg = lightbox.querySelector<HTMLImageElement>('.lightbox-img');
  const lbCap = lightbox.querySelector<HTMLElement>('.lightbox-cap');
  const lbClose = lightbox.querySelector<HTMLButtonElement>('[data-lightbox-close]');
  let lastTrigger: HTMLElement | null = null;

  const openLightbox = (img: HTMLImageElement) => {
    if (!lbImg || !lbCap) return;
    lbImg.src = img.currentSrc || img.src;
    lbImg.alt = img.alt || '';
    const cap = img.closest('figure')?.querySelector('figcaption');
    if (cap) {
      lbCap.innerHTML = cap.innerHTML;
      lbCap.hidden = false;
    } else {
      lbCap.innerHTML = '';
      lbCap.hidden = true;
    }
    lastTrigger = img;
    lightbox.showModal();
  };

  for (const img of document.querySelectorAll<HTMLImageElement>('.fig img')) {
    const figN = img.closest('figure')?.querySelector('.fig-n')?.textContent?.trim();
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', figN ? `Enlarge ${figN.replace(/\.$/, '')}` : 'Enlarge figure');
    img.addEventListener('click', () => openLightbox(img));
    img.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openLightbox(img);
      }
    });
  }

  lbClose?.addEventListener('click', () => lightbox.close());
  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) lightbox.close();
  });
  lightbox.addEventListener('close', () => {
    lastTrigger?.focus();
    lastTrigger = null;
  });
}

// Expand all folded details (margin notes, appendices, technical blocks) when printing, then restore.
let openedForPrint: HTMLDetailsElement[] = [];
addEventListener('beforeprint', () => {
  openedForPrint = [];
  for (const d of document.querySelectorAll<HTMLDetailsElement>('.body details:not([open])')) {
    d.open = true;
    openedForPrint.push(d);
  }
});
addEventListener('afterprint', () => {
  for (const d of openedForPrint) d.open = false;
  openedForPrint = [];
});

// Side contents column scrollspy: tracks sections, subsections, and appendix headings.
const tocNav = document.querySelector<HTMLElement>('.toc');
const tocLinks = new Map(
  [...document.querySelectorAll<HTMLAnchorElement>('.toc a')].map((a) => [decodeURIComponent(a.hash.slice(1)), a]),
);
const trackedHeadings = [
  ...document.querySelectorAll<HTMLElement>('.body h2[id], .body h3[id], .body details.appendix-fold[id]'),
].filter((el) => tocLinks.has(el.id));

const isHiddenInFold = (el: HTMLElement) => {
  let cur = el.parentElement;
  while (cur) {
    if (cur instanceof HTMLDetailsElement && !cur.open) {
      const sum = el.closest('summary');
      if (!sum || sum.parentElement !== cur) return true;
    }
    cur = cur.parentElement;
  }
  return false;
};

if (tocLinks.size && trackedHeadings.length) {
  let queued = false;
  const mark = () => {
    queued = false;
    const visible = trackedHeadings.filter((h) => !isHiddenInFold(h));
    if (!visible.length) return;
    let current = visible[0];
    for (const h of visible) {
      if (h.getBoundingClientRect().top <= 140) current = h;
    }
    const atEnd = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
    if (atEnd) current = visible[visible.length - 1];
    for (const [id, a] of tocLinks) {
      if (id === current.id) {
        a.setAttribute('aria-current', 'true');
        if (tocNav && tocNav.scrollHeight > tocNav.clientHeight) {
          const navRect = tocNav.getBoundingClientRect();
          const aRect = a.getBoundingClientRect();
          if (aRect.top < navRect.top + 12 || aRect.bottom > navRect.bottom - 12) {
            a.scrollIntoView({ block: 'nearest' });
          }
        }
      } else {
        a.removeAttribute('aria-current');
      }
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
  for (const d of document.querySelectorAll<HTMLDetailsElement>('.body details')) {
    d.addEventListener('toggle', queue);
  }
  mark();
}
