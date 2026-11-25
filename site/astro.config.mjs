import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import { rehypeArticle, remarkArticle } from './src/lib/article.mjs';
import { rehypeMath } from './src/lib/math.mjs';

// HTAIL Paper Shiki theme: WCAG AA+ verified on #F1EADC tinted paper ground.
const htailPaperTheme = {
  name: 'htail-paper',
  type: 'light',
  fg: '#1A1613',
  bg: '#F1EADC',
  colors: {
    'editor.background': '#F1EADC',
    'editor.foreground': '#1A1613',
  },
  settings: [
    {
      settings: { foreground: '#1A1613', background: '#F1EADC' },
    },
    {
      scope: ['comment', 'punctuation.definition.comment'],
      settings: { foreground: '#6A6158', fontStyle: 'italic' },
    },
    {
      scope: [
        'keyword',
        'storage',
        'storage.type',
        'storage.modifier',
        'keyword.control',
        'keyword.operator',
        'punctuation.accessor',
      ],
      settings: { foreground: '#680F0D' },
    },
    {
      scope: ['string', 'string.quoted', 'string.template', 'punctuation.definition.string'],
      settings: { foreground: '#1B3A18' },
    },
    {
      scope: ['string.unquoted', 'string.unquoted.argument.shell', 'meta.argument.shell'],
      settings: { foreground: '#1A1613' },
    },
    {
      scope: [
        'entity.name.function',
        'support.function',
        'meta.function-call',
        'entity.name.command',
        'entity.name.command.shell',
        'support.function.builtin',
        'entity.name.type',
        'entity.name.class',
        'support.type',
        'support.class',
      ],
      settings: { foreground: '#0E3A43' },
    },
    {
      scope: ['constant.numeric', 'constant.language', 'constant.character', 'constant.other'],
      settings: { foreground: '#5C2446' },
    },
    {
      scope: [
        'entity.other.attribute-name',
        'support.type.property-name',
        'meta.object-literal.key',
        'variable.parameter.option',
        'constant.other.option',
        'constant.other.option.dash.shell',
        'punctuation.definition.parameter.shell',
        'entity.name.tag',
      ],
      settings: { foreground: '#8A4510' },
    },
    {
      scope: ['variable', 'variable.parameter', 'variable.other'],
      settings: { foreground: '#1A1613' },
    },
    {
      scope: ['punctuation', 'meta.brace', 'punctuation.separator', 'punctuation.terminator'],
      settings: { foreground: '#5E564D' },
    },
    {
      scope: ['markup.heading', 'entity.name.section'],
      settings: { foreground: '#680F0D', fontStyle: 'bold' },
    },
    {
      scope: ['markup.italic'],
      settings: { fontStyle: 'italic' },
    },
    {
      scope: ['markup.bold'],
      settings: { fontStyle: 'bold' },
    },
    {
      scope: ['markup.inserted'],
      settings: { foreground: '#1B3A18' },
    },
    {
      scope: ['markup.deleted'],
      settings: { foreground: '#680F0D' },
    },
  ],
};

// `site` and `base` are passed on the command line by the GitHub Pages workflow
// (astro build --site ... --base ...), so nothing here names a repository or a domain.
export default defineConfig({
  output: 'static',
  devToolbar: { enabled: false },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkDirective, remarkArticle],
      rehypePlugins: [rehypeMath, [rehypeArticle, { bibliography: 'refs.bib' }]],
    }),
    shikiConfig: { theme: htailPaperTheme },
  },
  vite: { build: { assetsInlineLimit: 0 } },
});
