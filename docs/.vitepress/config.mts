import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'GTaP',
  description:
    'A pragma-based task-parallel programming system for GPUs, implemented in CUDA C++.',
  lang: 'en-US',
  base: '/GTaP/',
  cleanUrls: true,
  lastUpdated: true,
  appearance: false,

  head: [
    ['meta', { name: 'theme-color', content: '#1c2d3c' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'GTaP — Task parallelism for GPUs' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Express structured fork-join task parallelism directly on GPUs with a pragma-based CUDA C++ programming model.'
      }
    ]
  ],

  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  themeConfig: {
  siteTitle: 'GTaP',
  nav: [
    {
      text: 'Tutorial',
      link: '/tutorial/installation',
      activeMatch: '^/tutorial/'
    },
    {
      text: 'API Reference',
      link: '/reference/pragmas',
      activeMatch: '^/reference/'
    }
  ],

    sidebar: {
      '/tutorial/': [
        {
          text: 'Tutorial',
          items: [
            { text: 'Installation', link: '/tutorial/installation' },
            { text: 'Quickstart', link: '/tutorial/quickstart' },
            { text: 'Execution Modes', link: '/tutorial/execution-modes' },
            { text: 'Rules and Pitfalls', link: '/tutorial/rules-and-pitfalls' },
            { text: 'Profiling', link: '/tutorial/profiling' }
          ]
        }
      ],
      '/reference/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Pragmas', link: '/reference/pragmas' },
            { text: 'Runtime Functions', link: '/reference/runtime-functions' },
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'Profiling', link: '/reference/profiling' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/yukim0359/GTaP' }
    ],
    search: { provider: 'local' },
    outline: { level: 2, label: 'On this page' },
    footer: {
      message: 'GTaP is a research prototype under active development.'
    }
  }
})
