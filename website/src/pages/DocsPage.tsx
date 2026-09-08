import { Menu, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router'
import architectureGuide from '@root/docs/architecture.md?raw'
import cliGuide from '@root/docs/cli.md?raw'
import configurationGuide from '@root/docs/configuration.md?raw'
import docsIndex from '@root/docs/README.md?raw'
import sdkApiReference from '@root/docs/sdk-api.md?raw'
import sdkGuide from '@root/docs/sdk.md?raw'
import { MarkdownDocument } from './shared/MarkdownDocument'
import { SiteShell } from './shared/SiteShell'
import { docsPath } from '@/content/docsRoutes'

const OVERVIEW = `# Ness Agent documentation

Ness Agent is an experimental, hackable coding-agent harness for engineers who want to **own the loop**. One package contains the Python SDK for embedding the loop and **Ness**, the terminal operator surface for coding sessions.

> **0.x experimental** — public APIs may change until 1.0. Pin versions in production and follow the changelog when upgrading. Current release: **0.2.4**.

## Two surfaces, one harness

| Surface | Use it when |
| --- | --- |
| **SDK** | Your app, script, or internal tool needs the loop, model, tools, prompts, memory, and policy as code. |
| **Ness CLI** | You need an interactive coding session with plan/act modes, threads, worktrees, and project-local controls. |

## Install

\`\`\`bash
curl -fsSL https://nessagent.dev/install.sh | sh  # CLI (macOS/Linux)
powershell -c "irm https://nessagent.dev/install.ps1 | iex"  # CLI (Windows)
uv tool install --upgrade ness-agent               # CLI install or update with uv
pip install ness-agent                              # SDK in a project environment
ness --version                                      # verify the install
\`\`\`

## What's new in 0.2.4

- **Image-aware reads** — \`read()\` sends normalized raster images to vision models while keeping base64 payloads out of durable events, traces, hooks, and display output.
- **Long-turn compaction** — older work inside an oversized active continuation can be summarized while a coherent recent tool-call suffix remains verbatim.
- **Safer context limits** — over-budget continuations fail explicitly when no safe retained suffix can fit, and image blocks receive bounded token estimates.
- **Harbor evals** — Terminal-Bench 2.1 adapters and configs support OpenRouter and ChatGPT-authenticated Codex runs on Modal.

Full notes: [release dispatch](/news/image-aware-reads-and-mid-turn-compaction) · [changelog](../CHANGELOG.md).

The guides below keep the repository documentation close to the product source.`

const DOCS = [
  { slug: 'overview', label: 'Overview', index: '00', content: `${OVERVIEW}\n\n---\n\n${docsIndex}` },
  {
    slug: 'sdk',
    label: 'SDK',
    index: '01',
    content: `${sdkGuide}\n\n---\n\n## Detailed API map\n\nFor signatures and contracts for every public export in \`ness_agent.__all__\`, open the [SDK API reference](/docs/sdk-api).`,
  },
  {
    slug: 'sdk-api',
    label: 'SDK API',
    index: '02',
    content: sdkApiReference,
  },
  { slug: 'cli', label: 'CLI', index: '03', content: cliGuide },
  { slug: 'configuration', label: 'Configuration', index: '04', content: configurationGuide },
  { slug: 'architecture', label: 'Architecture', index: '05', content: architectureGuide },
] as const

const GITHUB_DOCS = 'https://github.com/Sagnnik/ness-agent/blob/main'
const GITHUB_ROOT = 'https://github.com/Sagnnik/ness-agent/blob/main'

function resolveDocsLink(href: string) {
  if (href.startsWith('#') || /^(?:https?:|mailto:)/i.test(href)) return href

  const [rawPath, anchor] = href.split('#')
  const path = rawPath.replace(/^\.\//, '')
  const file = path.replace(/\.md$/, '')
  const bare = file.replace(/^\.\.\//, '')

  const known = DOCS.find(
    (doc) =>
      doc.slug === file ||
      doc.slug === bare ||
      `${doc.slug}.md` === path ||
      file.endsWith(`/${doc.slug}`) ||
      file.endsWith(`/${doc.slug}.md`.replace(/\.md$/, '')),
  )
  if (known) {
    return `${docsPath(known.slug)}${anchor ? `#${anchor}` : ''}`
  }

  if (path === '../README' || path === '../README.md' || path === 'README.md') {
    return '/docs'
  }

  if (path === '../CHANGELOG' || path === '../CHANGELOG.md' || path === 'CHANGELOG.md') {
    return `${GITHUB_ROOT}/CHANGELOG.md`
  }

  if (path.startsWith('../')) {
    return `${GITHUB_DOCS}/${path.replace(/^\.\.\//, '')}`
  }

  return href
}

function scrollToHash(hash: string) {
  if (!hash || hash === '#') return
  const id = decodeURIComponent(hash.slice(1))
  window.requestAnimationFrame(() => {
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
  })
}

function slugifyHeading(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
}

function extractHeadings(markdown: string) {
  const headings: { id: string; label: string }[] = []
  for (const line of markdown.split(/\r?\n/)) {
    const match = /^##\s+(.+)$/.exec(line.trim())
    if (!match) continue
    const label = match[1].replace(/`/g, '').trim()
    const id = slugifyHeading(label)
    if (id) headings.push({ id, label })
  }
  return headings
}

export function DocsPage() {
  const { section } = useParams()
  const { pathname, hash } = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeHeading, setActiveHeading] = useState('')
  const activeSlug = section ?? 'overview'
  const active = DOCS.find((doc) => doc.slug === activeSlug) ?? DOCS[0]
  const headings = useMemo(() => extractHeadings(active.content), [active.content])

  useEffect(() => {
    setMenuOpen(false)
    setActiveHeading(hash ? decodeURIComponent(hash.slice(1)) : '')
    if (hash) {
      scrollToHash(hash)
    } else {
      window.scrollTo({ top: 0, behavior: 'instant' })
    }
  }, [pathname, hash, active.content])

  useEffect(() => {
    if (headings.length === 0) return

    let observer: IntersectionObserver | undefined
    const timer = window.setTimeout(() => {
      const nodes = headings
        .map((heading) => document.getElementById(heading.id))
        .filter((node): node is HTMLElement => Boolean(node))

      if (nodes.length === 0) return

      observer = new IntersectionObserver(
        (entries) => {
          const visible = entries
            .filter((entry) => entry.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
          if (visible[0]?.target.id) {
            setActiveHeading(visible[0].target.id)
          }
        },
        {
          rootMargin: '-20% 0px -65% 0px',
          threshold: [0, 0.25, 0.5, 1],
        },
      )

      for (const node of nodes) observer.observe(node)
    }, 60)

    return () => {
      window.clearTimeout(timer)
      observer?.disconnect()
    }
  }, [headings, active.content])

  return (
    <SiteShell className="site-shell--docs" footer={false}>
      <header className="docs-header">
        <div>
          <p className="site-kicker">[ OPERATOR MANUAL // 0.x ]</p>
          <h1>{active.label}</h1>
        </div>
        <button
          type="button"
          className="docs-menu-button"
          onClick={() => setMenuOpen((open) => !open)}
          aria-expanded={menuOpen}
          aria-controls="docs-nav"
        >
          {menuOpen ? <X size={16} /> : <Menu size={16} />}
          sections
        </button>
      </header>
      <div className="docs-layout">
        <aside className={`docs-sidebar${menuOpen ? ' is-open' : ''}`} id="docs-nav">
          <p>DOCUMENT INDEX</p>
          <nav aria-label="Documentation sections">
            {DOCS.map((doc) => {
              const to = docsPath(doc.slug)
              const isActive = active.slug === doc.slug
              return (
                <div key={doc.slug} className="docs-sidebar__group">
                  <Link to={to} className={isActive ? 'is-active' : undefined}>
                    <span>{doc.index}</span>
                    {doc.label}
                  </Link>
                  {isActive && headings.length > 0 ? (
                    <ul className="docs-sidebar__toc">
                      {headings.map((heading) => (
                        <li key={heading.id}>
                          <a
                            href={`#${heading.id}`}
                            className={
                              activeHeading === heading.id ? 'is-active' : undefined
                            }
                            onClick={(event) => {
                              event.preventDefault()
                              setActiveHeading(heading.id)
                              scrollToHash(`#${heading.id}`)
                              window.history.replaceState(
                                null,
                                '',
                                `${docsPath(doc.slug)}#${heading.id}`,
                              )
                            }}
                          >
                            {heading.label}
                          </a>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              )
            })}
          </nav>
          <p className="docs-sidebar__note">
            Repository sources are adapted here. API interfaces remain experimental before 1.0.
          </p>
        </aside>
        <article className="docs-document markdown-body">
          <MarkdownDocument content={active.content} resolveLink={resolveDocsLink} />
        </article>
      </div>
    </SiteShell>
  )
}
