import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import rehypeKatex from 'rehype-katex';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import { rehypeArticle, remarkArticle } from './src/lib/article.mjs';

// `site` and `base` are passed on the command line by the GitHub Pages workflow
// (astro build --site ... --base ...), so nothing here names a repository or a domain.
export default defineConfig({
  output: 'static',
  devToolbar: { enabled: false },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkDirective, remarkArticle],
      rehypePlugins: [rehypeKatex, [rehypeArticle, { bibliography: 'refs.bib' }]],
    }),
    shikiConfig: { theme: 'css-variables' },
  },
  vite: { build: { assetsInlineLimit: 0 } },
});
