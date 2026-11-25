// Build-time TeX math renderer powered by MathJax v4 SVG with TeX Gyre Pagella Math.
// Ships zero client-side math JavaScript and zero external font requests.

import { mathjax } from '@mathjax/src/js/mathjax.js';
import '@mathjax/src/js/util/asyncLoad/esm.js';
import { TeX } from '@mathjax/src/js/input/tex.js';
import { SVG } from '@mathjax/src/js/output/svg.js';
import { liteAdaptor } from '@mathjax/src/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/js/handlers/html.js';
import { SerializedMmlVisitor } from '@mathjax/src/js/core/MmlTree/SerializedMmlVisitor.js';
import { STATE } from '@mathjax/src/js/core/MathItem.js';
import { MathJaxPagellaFont } from '@mathjax/mathjax-pagella-font/js/svg.js';

import '@mathjax/src/js/input/tex/base/BaseConfiguration.js';
import '@mathjax/src/js/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/js/input/tex/bbox/BboxConfiguration.js';
import '@mathjax/src/js/input/tex/boldsymbol/BoldsymbolConfiguration.js';
import '@mathjax/src/js/input/tex/braket/BraketConfiguration.js';
import '@mathjax/src/js/input/tex/cancel/CancelConfiguration.js';
import '@mathjax/src/js/input/tex/cases/CasesConfiguration.js';
import '@mathjax/src/js/input/tex/color/ColorConfiguration.js';
import '@mathjax/src/js/input/tex/extpfeil/ExtpfeilConfiguration.js';
import '@mathjax/src/js/input/tex/mathtools/MathtoolsConfiguration.js';
import '@mathjax/src/js/input/tex/newcommand/NewcommandConfiguration.js';
import '@mathjax/src/js/input/tex/noerrors/NoErrorsConfiguration.js';
import '@mathjax/src/js/input/tex/noundefined/NoUndefinedConfiguration.js';
import '@mathjax/src/js/input/tex/textcomp/TextcompConfiguration.js';
import '@mathjax/src/js/input/tex/textmacros/TextMacrosConfiguration.js';
import '@mathjax/src/js/input/tex/unicode/UnicodeConfiguration.js';
import '@mathjax/src/js/input/tex/units/UnitsConfiguration.js';
import '@mathjax/src/js/input/tex/upgreek/UpgreekConfiguration.js';

const TEX_PACKAGES = [
  'base',
  'ams',
  'bbox',
  'boldsymbol',
  'braket',
  'cancel',
  'cases',
  'color',
  'extpfeil',
  'mathtools',
  'newcommand',
  'noerrors',
  'noundefined',
  'textcomp',
  'textmacros',
  'unicode',
  'units',
  'upgreek',
];

let ctx = null;

function getContext() {
  if (ctx) return ctx;
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({ packages: TEX_PACKAGES });
  const svg = new SVG({
    fontData: new MathJaxPagellaFont(),
    fontCache: 'local',
    mtextInheritFont: true,
    linebreaks: { inline: false },
  });
  const doc = mathjax.document('', { InputJax: tex, OutputJax: svg });
  const mmlVisitor = new SerializedMmlVisitor();
  ctx = { adaptor, doc, mmlVisitor };
  return ctx;
}

const hasClass = (node, cls) =>
  node?.type === 'element' && [].concat(node.properties?.className || []).includes(cls);

const isBlank = (node) => node?.type === 'text' && !node.value.trim();

function nodeText(node) {
  if (!node) return '';
  if (node.type === 'text') return node.value;
  if (node.children) return node.children.map(nodeText).join('');
  return '';
}

const TEX_SYMBOLS = {
  ge: '≥',
  geq: '≥',
  le: '≤',
  leq: '≤',
  times: '×',
  cdot: '×',
  pm: '±',
  mp: '∓',
  ne: '≠',
  neq: '≠',
  approx: '≈',
  sim: '~',
  equiv: '≡',
  to: '→',
  rightarrow: '→',
  leftarrow: '←',
  leftrightarrow: '↔',
  infty: '∞',
  mid: '|',
  lVert: '‖',
  rVert: '‖',
  bmod: 'mod',
  Delta: 'Δ',
  Sigma: 'Σ',
  Omega: 'Ω',
  Lambda: 'Λ',
  Gamma: 'Γ',
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  epsilon: 'ε',
  varepsilon: 'ε',
  theta: 'θ',
  lambda: 'λ',
  mu: 'μ',
  pi: 'π',
  rho: 'ρ',
  sigma: 'σ',
  tau: 'τ',
  phi: 'φ',
  varphi: 'φ',
  chi: 'χ',
  psi: 'ψ',
  omega: 'ω',
};

/** Compact plain-text representation used for collapsed margin-note teasers. */
function readableTex(src) {
  return src
    .replace(/\{,\}/g, ',')
    .replace(/\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|operatorname)\s*\{([^{}]*)\}/g, '$1')
    .replace(/\\([%_$&#])/g, '$1')
    .replace(/\\[,;:! ]/g, ' ')
    .replace(/\\([A-Za-z]+)\b/g, (_, name) => TEX_SYMBOLS[name] ?? name)
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

async function compileMath(tex, display, inlineTag) {
  const { adaptor, doc, mmlVisitor } = getContext();
  const mjxNode = await mathjax.handleRetriesFor(() => doc.convert(tex, { display }));

  // Mark the visual container and all <svg> children decorative so screen readers read assistive MathML once.
  adaptor.setAttribute(mjxNode, 'aria-hidden', 'true');
  for (const child of adaptor.childNodes(mjxNode)) {
    if (adaptor.kind(child) === 'svg') {
      adaptor.removeAttribute(child, 'role');
      adaptor.setAttribute(child, 'aria-hidden', 'true');
      adaptor.setAttribute(child, 'focusable', 'false');
    }
  }
  const svgMarkup = adaptor.outerHTML(mjxNode);

  const mmlTree = await mathjax.handleRetriesFor(() => doc.convert(tex, { display, end: STATE.CONVERT }));
  const mmlMarkup = mmlVisitor.visitTree(mmlTree);

  const wrapperTag = display && !inlineTag ? 'div' : 'span';
  return {
    type: 'element',
    tagName: wrapperTag,
    properties: {
      className: ['math', display ? 'math-display' : 'math-inline'],
    },
    data: {
      tex,
      mathText: readableTex(tex),
    },
    children: [
      { type: 'raw', value: svgMarkup },
      {
        type: 'element',
        tagName: 'mjx-assistive-mml',
        properties: {
          className: ['sr-only'],
          display: display ? 'block' : 'inline',
        },
        children: [{ type: 'raw', value: mmlMarkup }],
      },
    ],
  };
}

export function rehypeMath() {
  return async (tree) => {
    const jobs = [];

    const collect = (node, parent, index, insideParagraph) => {
      if (!node || node.type !== 'element') return;

      // Display math from remark-math@6 + remark-rehype: <pre><code class="language-math math-display">
      if (node.tagName === 'pre') {
        const solid = (node.children || []).filter((c) => !isBlank(c));
        if (solid.length === 1 && solid[0].type === 'element' && solid[0].tagName === 'code' && hasClass(solid[0], 'language-math')) {
          const code = solid[0];
          const display = hasClass(code, 'math-display') || !hasClass(code, 'math-inline');
          jobs.push({ parent, index, tex: nodeText(code).trim(), display, inlineTag: insideParagraph });
          return;
        }
      }

      // Inline or standalone <code class="language-math">, <span class="math-inline">, <div class="math-display">
      const isCodeMath = node.tagName === 'code' && hasClass(node, 'language-math');
      const isSpanInline = node.tagName === 'span' && hasClass(node, 'math-inline') && !hasClass(node, 'math');
      const isDivDisplay = node.tagName === 'div' && hasClass(node, 'math-display') && !hasClass(node, 'math');

      if (isCodeMath || isSpanInline || isDivDisplay) {
        const display = isDivDisplay || hasClass(node, 'math-display');
        jobs.push({ parent, index, tex: nodeText(node).trim(), display, inlineTag: insideParagraph });
        return;
      }

      if (!node.children) return;
      const inP = insideParagraph || node.tagName === 'p';
      for (let i = 0; i < node.children.length; i++) {
        collect(node.children[i], node, i, inP);
      }
    };

    if (tree.children) {
      for (let i = 0; i < tree.children.length; i++) {
        collect(tree.children[i], tree, i, false);
      }
    }

    for (const job of jobs) {
      if (!job.tex) continue;
      job.parent.children[job.index] = await compileMath(job.tex, job.display, job.inlineTag);
    }

    // Glue adjacent closing punctuation (and opening paren/bracket) to inline math so ')' never orphans on a new line.
    const bindPunctuation = (node) => {
      if (!node || !node.children) return;
      const next = [];
      const kids = node.children;
      for (let i = 0; i < kids.length; i++) {
        const cur = kids[i];
        if (cur?.type === 'element' && hasClass(cur, 'math-inline')) {
          let lead = '';
          const prev = next[next.length - 1];
          if (prev?.type === 'text') {
            const lm = /([(\[“‘"']+)$/.exec(prev.value);
            if (lm) {
              lead = lm[1];
              prev.value = prev.value.slice(0, -lead.length);
              if (!prev.value) next.pop();
            }
          }
          let trail = '';
          const after = kids[i + 1];
          if (after?.type === 'text') {
            const tm = /^([.,;:!?)\]”’"']+)/.exec(after.value);
            if (tm) {
              trail = tm[1];
              after.value = after.value.slice(trail.length);
              if (!after.value) i++;
            }
          }
          if (lead || trail) {
            const wrapped = [];
            if (lead) wrapped.push({ type: 'text', value: lead });
            wrapped.push(cur);
            if (trail) wrapped.push({ type: 'text', value: trail });
            next.push({ type: 'element', tagName: 'span', properties: { className: ['math-keep'] }, children: wrapped });
            continue;
          }
        } else {
          bindPunctuation(cur);
        }
        next.push(cur);
      }
      node.children = next;
    };
    bindPunctuation(tree);
  };
}
