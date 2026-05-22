export const esConfig = {
  title: 'Modelo Mundial FIFA 2026',
  description: 'Pipeline de ML para predecir las probabilidades de campeón del Mundial FIFA 2026 para las 48 selecciones nacionales.',

  themeConfig: {
    nav: [
      { text: 'Inicio', link: '/es/' },
      { text: 'Ingeniería de Datos', link: '/es/ingenieria-datos/' },
      { text: 'Modelado', link: '/es/modelado/' },
      { text: 'Simulación', link: '/es/simulacion/' },
      { text: 'Resultados', link: '/es/resultados/' },
    ],

    sidebar: {
      '/es/ingenieria-datos/': [
        {
          text: 'Ingeniería de Datos',
          items: [
            { text: 'Resumen', link: '/es/ingenieria-datos/' },
          ],
        },
        {
          text: 'Repositorio',
          items: [
            { text: 'Estructura del Proyecto', link: '/es/ingenieria-datos/estructura-proyecto' },
            { text: 'Notebooks', link: '/es/ingenieria-datos/notebooks' },
          ],
        },
      ],
      '/es/modelado/': [
        {
          text: 'Modelado',
          items: [
            { text: 'Resumen', link: '/es/modelado/' },
          ],
        },
      ],
      '/es/simulacion/': [
        {
          text: 'Simulación',
          items: [
            { text: 'Resumen', link: '/es/simulacion/' },
          ],
        },
      ],
      '/es/resultados/': [
        {
          text: 'Resultados',
          items: [
            { text: 'Resumen', link: '/es/resultados/' },
          ],
        },
      ],
    },

    editLink: {
      pattern: 'https://github.com/davidderas50/fifa-world-cup-model/edit/main/docs/:path',
      text: 'Editar esta página en GitHub',
    },

    footer: {
      message: 'Publicado bajo Licencia MIT.',
    },

    docFooter: {
      prev: 'Página anterior',
      next: 'Página siguiente',
    },

    outline: {
      label: 'En esta página',
    },

    returnToTopLabel: 'Volver al inicio',
    sidebarMenuLabel: 'Menú',
    darkModeSwitchLabel: 'Tema',
    langMenuLabel: 'Cambiar idioma',
  },
}
