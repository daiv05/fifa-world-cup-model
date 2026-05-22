import { defineConfig } from 'vitepress'
import { enConfig } from './locales/en.mjs'
import { esConfig } from './locales/es.mjs'

export default defineConfig({
  title: 'FIFA World Cup 2026 Model',
  description: 'ML pipeline to predict World Cup 2026 champion probabilities',

  head: [
    ['link', { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }],
    ['meta', { name: 'theme-color', content: '#0d6e27' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:image', content: '/og-image.png' }],
  ],

  locales: {
    root: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      ...enConfig,
    },
    es: {
      label: 'Español',
      lang: 'es',
      link: '/es/',
      ...esConfig,
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/daiv05/fifa-world-cup-model' },
    ],
    search: {
      provider: 'local',
    },
  },

  markdown: {
    math: true,
    lineNumbers: true,
  },

  sitemap: {
    hostname: 'https://fifa-world-cup-model.vercel.app',
  },
})
