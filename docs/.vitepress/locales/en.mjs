export const enConfig = {
  title: 'FIFA World Cup 2026 Model',
  description: 'ML pipeline to predict FIFA World Cup 2026 champion probabilities across all 48 national teams.',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/en/' },
      { text: 'Data Engineering', link: '/en/data-engineering/' },
      { text: 'Modeling', link: '/en/modeling/' },
      { text: 'Simulation', link: '/en/simulation/' },
      { text: 'Results', link: '/en/results/' },
    ],

    sidebar: {
      '/en/data-engineering/': [
        {
          text: 'Data Engineering',
          items: [
            { text: 'Overview', link: '/en/data-engineering/' },
          ],
        },
        {
          text: 'Repository',
          items: [
            { text: 'Project Structure', link: '/en/data-engineering/project-structure' },
            { text: 'Notebooks', link: '/en/data-engineering/notebooks' },
          ],
        },
      ],
      '/en/modeling/': [
        {
          text: 'Modeling',
          items: [
            { text: 'Overview', link: '/en/modeling/' },
          ],
        },
      ],
      '/en/simulation/': [
        {
          text: 'Simulation',
          items: [
            { text: 'Overview', link: '/en/simulation/' },
          ],
        },
      ],
      '/en/results/': [
        {
          text: 'Results',
          items: [
            { text: 'Overview', link: '/en/results/' },
          ],
        },
      ],
    },

    editLink: {
      pattern: 'https://github.com/daiv05/fifa-world-cup-model/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },

    footer: {
      message: 'Released under the MIT License.',
    },

    docFooter: {
      prev: 'Previous page',
      next: 'Next page',
    },

    outline: {
      label: 'On this page',
    },

    returnToTopLabel: 'Return to top',
    sidebarMenuLabel: 'Menu',
    darkModeSwitchLabel: 'Theme',
  },
}
