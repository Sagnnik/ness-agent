import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE_URL = 'https://nessagent.dev'
const SITE_NAME = 'Ness Agent'
const SITE_DESCRIPTION =
  'A hackable coding-agent harness for engineers who need the loop within reach.'
const BLOG_DESCRIPTION =
  'Working notes on engineering the harness, its operating surfaces, and its extension seams.'
const DOCS_DESCRIPTION =
  'Documentation for the Ness Agent SDK, CLI, configuration, and runtime architecture.'
const WEBSITE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const DIST_DIR = path.join(WEBSITE_DIR, 'dist')
const CONTENT_DIR = path.join(WEBSITE_DIR, 'content', 'blog')

const NEWS = [
  {
    slug: 'concurrent-threads-and-export',
    title: 'Concurrent threads, /export, and on-demand reflection',
    description:
      'Run multiple CLI turns in parallel, export durable sessions to HTML, reflect on demand, and read absolute paths outside the project root.',
    date: '2026-08-15',
  },
  {
    slug: 'codex-provider-and-login',
    title: 'Codex subscription, /login, and install scripts',
    description:
      'Sign in with a Codex subscription or OpenRouter key, switch providers in-session, and install Ness from curl or PowerShell.',
    date: '2026-08-12',
  },
  {
    slug: 'mcp-runtime-and-skills',
    title: 'MCP runtime, skills roots, and cache-safe compaction',
    description:
      'Adapter-neutral MCP for SDK hosts, first-class .ness/mcp.json and OAuth CLI, multi-root skills, and a leaner compaction surface.',
    date: '2026-08-07',
  },
  {
    slug: 'initial-public-release',
    title: 'Initial public release',
    description:
      'Ness Agent and Ness are now public: one Python package, a reusable harness, and an interactive coding surface.',
    date: '2026-07-31',
  },
]

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function safeJson(value) {
  return JSON.stringify(value).replaceAll('<', '\\u003c')
}

function parseFrontmatter(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  const data = {}
  if (!match) return data

  for (const line of match[1].split(/\r?\n/)) {
    const separator = line.indexOf(':')
    if (separator === -1) continue
    const key = line.slice(0, separator).trim()
    let value = line.slice(separator + 1).trim()
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    }
    data[key] = value
  }
  return data
}

async function readBlogPosts() {
  const folders = await readdir(CONTENT_DIR, { withFileTypes: true })
  const posts = []
  for (const folder of folders) {
    if (!folder.isDirectory()) continue
    const file = path.join(CONTENT_DIR, folder.name, 'index.md')
    const data = parseFrontmatter(await readFile(file, 'utf8'))
    posts.push({
      slug: data.slug || folder.name,
      title: data.title || 'Untitled field note',
      description: data.description || '',
      date: data.date || '',
      image: data.image || '',
    })
  }
  return posts
}

async function resolveImage(image) {
  const value = String(image || '').trim()
  if (!value) return undefined
  if (/^https?:\/\//i.test(value)) return value
  if (value.startsWith('/')) return new URL(value, SITE_URL).toString()

  // Blog assets are fingerprinted by Vite. Resolve the source filename to the
  // emitted asset so a future cover can use `assets/cover.png` in frontmatter.
  const filename = path.basename(value)
  const stem = filename.slice(0, filename.lastIndexOf('.'))
  const emitted = (await readdir(path.join(DIST_DIR, 'assets'))).find(
    (candidate) => candidate === filename || candidate.startsWith(`${stem}-`),
  )
  return emitted ? new URL(`/assets/${emitted}`, SITE_URL).toString() : undefined
}

function metadataTags(metadata) {
  const tags = [
    `<meta name="description" content="${escapeHtml(metadata.description)}" />`,
    '<meta name="robots" content="index,follow,max-image-preview:large" />',
    `<link rel="canonical" href="${escapeHtml(metadata.url)}" />`,
    `<meta property="og:site_name" content="${escapeHtml(SITE_NAME)}" />`,
    `<meta property="og:type" content="${escapeHtml(metadata.type)}" />`,
    `<meta property="og:title" content="${escapeHtml(metadata.title)}" />`,
    `<meta property="og:description" content="${escapeHtml(metadata.description)}" />`,
    `<meta property="og:url" content="${escapeHtml(metadata.url)}" />`,
    `<meta name="twitter:card" content="${metadata.image ? 'summary_large_image' : 'summary'}" />`,
    `<meta name="twitter:title" content="${escapeHtml(metadata.title)}" />`,
    `<meta name="twitter:description" content="${escapeHtml(metadata.description)}" />`,
  ]

  if (metadata.image) {
    tags.push(`<meta property="og:image" content="${escapeHtml(metadata.image)}" />`)
    tags.push(`<meta name="twitter:image" content="${escapeHtml(metadata.image)}" />`)
  }

  if (metadata.type === 'article' && metadata.publishedTime) {
    tags.push(
      `<meta property="article:published_time" content="${escapeHtml(metadata.publishedTime)}" />`,
      '<meta property="article:author" content="Sagnnik Biswas" />',
    )
  }

  const schema =
    metadata.type === 'article'
      ? {
          '@context': 'https://schema.org',
          '@type': 'Article',
          headline: metadata.title,
          description: metadata.description,
          datePublished: metadata.publishedTime,
          author: { '@type': 'Person', name: 'Sagnnik Biswas' },
          mainEntityOfPage: metadata.url,
          ...(metadata.image ? { image: [metadata.image] } : {}),
        }
      : {
          '@context': 'https://schema.org',
          '@type': 'WebSite',
          name: SITE_NAME,
          description: metadata.description,
          url: SITE_URL,
        }
  tags.push(`<script id="ness-seo-jsonld" type="application/ld+json">${safeJson(schema)}</script>`)
  return tags.join('\n    ')
}

function renderHtml(template, metadata) {
  const cleaned = template
    .replace(/\s*<title>[\s\S]*?<\/title>/i, '')
    .replace(
      /\s*<meta\b[^>]*(?:name|property)="(?:description|robots|og:[^"]+|twitter:[^"]+|article:[^"]+)"[^>]*\/?>/gi,
      '',
    )
    .replace(/\s*<link\b[^>]*rel="canonical"[^>]*\/?>/gi, '')
    .replace(/\s*<script\b[^>]*id="ness-seo-jsonld"[^>]*>[\s\S]*?<\/script>/gi, '')

  return cleaned.replace(
    '</head>',
    `    <title>${escapeHtml(metadata.title)}</title>\n    ${metadataTags(metadata)}\n  </head>`,
  )
}

function urlFor(pathname) {
  const normalizedPath = pathname === '/' ? '/' : `${pathname.replace(/\/+$/, '')}/`
  return new URL(normalizedPath, SITE_URL).toString()
}

function makeMetadata({ pathname, title, description, type = 'website', date, image }) {
  return {
    title,
    description,
    type,
    publishedTime: date,
    image,
    url: urlFor(pathname),
  }
}

async function writeRoute(template, pathname, metadata) {
  const routeDir = pathname === '/' ? DIST_DIR : path.join(DIST_DIR, pathname.slice(1))
  await mkdir(routeDir, { recursive: true })
  await writeFile(path.join(routeDir, 'index.html'), renderHtml(template, metadata))
}

async function writeSitemap(blogPosts) {
  const pages = [
    { pathname: '/home' },
    { pathname: '/news' },
    ...NEWS.map((item) => ({ pathname: `/news/${item.slug}`, date: item.date })),
    { pathname: '/blog' },
    ...blogPosts.map((post) => ({ pathname: `/blog/${post.slug}`, date: post.date })),
    { pathname: '/docs' },
    ...['sdk', 'sdk-api', 'cli', 'configuration', 'architecture'].map((slug) => ({
      pathname: `/docs/${slug}`,
    })),
  ]
  const entries = pages
    .map(
      ({ pathname, date }) =>
        `  <url>\n    <loc>${escapeHtml(urlFor(pathname))}</loc>${
          date ? `\n    <lastmod>${escapeHtml(date)}</lastmod>` : ''
        }\n  </url>`,
    )
    .join('\n')
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries}\n</urlset>\n`
  await writeFile(path.join(DIST_DIR, 'sitemap.xml'), sitemap)
}

const template = await readFile(path.join(DIST_DIR, 'index.html'), 'utf8')
const blogPosts = await readBlogPosts()

await writeRoute(
  template,
  '/',
  makeMetadata({
    pathname: '/home',
    title: 'Ness Agent — Own the Loop',
    description: SITE_DESCRIPTION,
  }),
)
await writeRoute(
  template,
  '/home',
  makeMetadata({
    pathname: '/home',
    title: 'Ness Agent — Own the Loop',
    description: SITE_DESCRIPTION,
  }),
)
await writeRoute(
  template,
  '/news',
  makeMetadata({
    pathname: '/news',
    title: 'Release Dispatches — Ness Agent',
    description: 'Release notes and field updates from the Ness Agent coding-agent harness.',
  }),
)
for (const item of NEWS) {
  await writeRoute(
    template,
    `/news/${item.slug}`,
    makeMetadata({
      pathname: `/news/${item.slug}`,
      title: `${item.title} — Ness Agent`,
      description: item.description,
      date: item.date,
    }),
  )
}
await writeRoute(
  template,
  '/blog',
  makeMetadata({ pathname: '/blog', title: 'Blog — Ness Agent', description: BLOG_DESCRIPTION }),
)
for (const post of blogPosts) {
  await writeRoute(
    template,
    `/blog/${post.slug}`,
    makeMetadata({
      pathname: `/blog/${post.slug}`,
      title: `${post.title} — Ness Agent`,
      description: post.description,
      type: 'article',
      date: post.date,
      image: await resolveImage(post.image),
    }),
  )
}
await writeRoute(
  template,
  '/docs',
  makeMetadata({ pathname: '/docs', title: 'Documentation — Ness Agent', description: DOCS_DESCRIPTION }),
)
for (const slug of ['sdk', 'sdk-api', 'cli', 'configuration', 'architecture']) {
  await writeRoute(
    template,
    `/docs/${slug}`,
    makeMetadata({
      pathname: `/docs/${slug}`,
      title: `${slug === 'sdk-api' ? 'SDK API' : slug[0].toUpperCase() + slug.slice(1)} Documentation — Ness Agent`,
      description: DOCS_DESCRIPTION,
    }),
  )
}
await writeSitemap(blogPosts)
