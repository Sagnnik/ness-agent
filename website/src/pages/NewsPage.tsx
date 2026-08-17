import { ArrowLeft, ArrowUpRight, CornerDownRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link, useParams } from 'react-router'
import { SiteShell } from './shared/SiteShell'

const CHANGELOG = 'https://github.com/Sagnnik/ness-agent/blob/main/CHANGELOG.md'
const BLOG_SLUG = 'inside-coding-agent-building-the-harness-around-the-model'

type Highlight = readonly [string, string]

type Release = {
  slug: string
  title: string
  date: string
  version: string
  summary: string
  sectionLabel: string
  intro: string
  highlights: readonly Highlight[]
  fieldNote?: ReactNode
}

const RELEASES: readonly Release[] = [
  {
    slug: 'concurrent-threads-and-export',
    title: 'Concurrent threads, /export, and on-demand reflection',
    date: '2026-08-15',
    version: 'v0.2.2',
    summary:
      'Run multiple CLI turns in parallel, export durable sessions to HTML, reflect on demand, and read absolute paths outside the project root.',
    sectionLabel: '00.2.2 // ADDED',
    intro:
      '0.2.2 makes the Ness CLI multi-threaded at the turn level, adds session export and manual reflection, widens SDK read access for absolute paths, and hardens Codex streaming against transient overload.',
    highlights: [
      [
        'Concurrent thread runtimes',
        '/threads switches between active turns without interrupting them; /new can start a fresh turn in the background while live threads show working, waiting, and cancelling states.',
      ],
      [
        'Session export',
        '/export <path.html> writes a self-contained HTML transcript with pre-compaction events, normalized JSONL download, image-byte omission, and overwrite protection.',
      ],
      [
        'On-demand reflection',
        '/reflection and Session.run_reflection() update session memory immediately; ReflectionResult is now part of the public SDK surface.',
      ],
      [
        'Absolute read paths',
        'The SDK read tool accepts absolute file paths outside the configured project root; relative paths still resolve from the project root.',
      ],
      [
        'Codex SSE retries',
        'Transient Codex stream failures such as server_is_overloaded retry with bounded exponential backoff, jitter, and Retry-After support.',
      ],
    ],
  },
  {
    slug: 'codex-provider-and-login',
    title: 'Codex ChatGPT sign-in, /login, and install scripts',
    date: '2026-08-12',
    version: 'v0.2.1',
    summary:
      'Sign in to Codex with ChatGPT or use an OpenRouter key, switch providers in-session, and install Ness from curl or PowerShell.',
    sectionLabel: '00.2.1 // ADDED',
    intro:
      "0.2.1 adds a pluggable provider layer with Codex authentication through ChatGPT's managed sign-in flow, refreshes the TUI around /login and /status, and ships cross-platform install scripts. The Codex model transport remains experimental. One database schema change requires action before upgrading.",
    highlights: [
      [
        'Experimental Codex integration',
        '/login authenticates Codex through the installed codex CLI app-server; credentials stay in Ness global config and never touch ~/.codex. Ness uses a separate experimental Codex Responses transport for inference.',
      ],
      [
        'Provider picker',
        'Switch between Codex and OpenRouter mid-session, reconnect, or log out from a dedicated /login picker that rebuilds the model while preserving the thread.',
      ],
      [
        'Subscription cost tracking',
        'CostTracker and TokenUsage now record billing_mode and cost_source so subscription-backed turns are labeled separately from API-estimated spend.',
      ],
      [
        'Session naming',
        'Session.set_name(), /rename <name>, and local timestamps in /threads for saved conversations.',
      ],
      [
        'Breaking threads.db',
        'The threads table now requires a name column. Back up or remove .ness/threads/threads.db before upgrading; no automatic migration.',
      ],
    ],
  },
  {
    slug: 'mcp-runtime-and-skills',
    title: 'MCP runtime, skills roots, and cache-safe compaction',
    date: '2026-08-07',
    version: 'v0.2.0',
    summary:
      'Adapter-neutral MCP for SDK hosts, first-class .ness/mcp.json and OAuth CLI, multi-root skills, and a leaner compaction surface.',
    sectionLabel: '00.2 // ADDED',
    intro:
      '0.2.0 splits MCP policy from connection runtime, opens explicit skill roots to host apps, and tightens compaction around the bound model. A few public APIs break on purpose — pin and read the changelog before upgrading.',
    highlights: [
      [
        'MCPRuntime',
        'Public, adapter-neutral MCPRuntime and MCPServerSpec so apps can connect without Ness project files or UI policy.',
      ],
      [
        'Ness MCP CLI',
        'Native .ness/mcp.json, stdio and Streamable HTTP, plus ness mcp status / login / logout / import for Cursor and Claude shapes.',
      ],
      [
        'Multi-root skills',
        'AgentSpec.skills_dirs plus merge_skill_dirs() / default_skill_search_dirs(); Ness loads .ness/skills and well-known agent roots.',
      ],
      [
        'Cache-safe summarize',
        'ness_agent.summarize() and durable summary checkpoints; compaction uses the main bound model and keeps the active turn verbatim.',
      ],
      [
        'Breaking cleanup',
        'RunResult.usage → usage_total; MCPManager removed in favor of MCPRuntime; automatic .env → global JSON migration dropped.',
      ],
    ],
  },
  {
    slug: 'initial-public-release',
    title: 'Initial public release',
    date: '2026-07-31',
    version: 'v0.1.0',
    summary:
      'Ness Agent and Ness are now public: one Python package, a reusable harness, and an interactive coding surface.',
    sectionLabel: '00.1 // ADDED',
    intro:
      'The initial release establishes Ness Agent as an experimental, hackable harness. APIs may change before 1.0; the seams are meant to be inspected.',
    highlights: [
      ['SDK + CLI', 'LangGraph agent loop and an interactive coding adapter in the same package.'],
      ['Tools + policy', 'Built-in tools, permissions, memory, skills, hooks, and MCP support.'],
      [
        'Context layers',
        'L0–L3 prompt assembly, ephemeral overlays, cache-aware compaction, and reflection.',
      ],
      [
        '/goal verification',
        'Bounded worker attempts with an independent judge and repair instructions on failure.',
      ],
      [
        'Project-local',
        'A versionable .ness/ surface for behavior, plus editable global instructions/templates.',
      ],
    ],
    fieldNote: (
      <>
        For the longer framing, read{' '}
        <Link to={`/blog/${BLOG_SLUG}`} className="site-inline-link">
          Harness Engineering: A Ness Agent Intro
        </Link>
        .
      </>
    ),
  },
]

function ReleaseDetail({ release }: { release: Release }) {
  return (
    <>
      <header className="dispatch-header">
        <p className="site-kicker">[ RELEASE DISPATCH // {release.version} ]</p>
        <h1>{release.title}</h1>
        <p>{release.summary}</p>
        <p className="dispatch-header__changelog">
          <a href={CHANGELOG} target="_blank" rel="noopener noreferrer">
            changelog
          </a>
        </p>
        <dl className="dispatch-meta">
          <div>
            <dt>DATE</dt>
            <dd>{release.date}</dd>
          </div>
          <div>
            <dt>STATUS</dt>
            <dd>RELEASED</dd>
          </div>
          <div>
            <dt>CHANNEL</dt>
            <dd>PUBLIC</dd>
          </div>
        </dl>
      </header>

      <section className="dispatch-body" aria-labelledby="release-highlights">
        <div className="dispatch-body__rail">{release.sectionLabel}</div>
        <div>
          <h2 id="release-highlights">
            {release.version === 'v0.1.0' ? 'first transmission' : 'field notes'}
          </h2>
          <p>{release.intro}</p>
          <ol className="dispatch-list">
            {release.highlights.map(([title, description], index) => (
              <li key={title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <strong>{title}</strong>
                <p>{description}</p>
              </li>
            ))}
          </ol>
          {release.fieldNote ? (
            <p className="dispatch-field-note">{release.fieldNote}</p>
          ) : null}
        </div>
      </section>

      <footer className="dispatch-footer">
        <Link to="/docs" className="site-link-button">
          <CornerDownRight size={15} aria-hidden="true" />
          inspect the docs
        </Link>
        <Link to="/news" className="site-inline-link">
          back to dispatches
        </Link>
      </footer>
    </>
  )
}

function NewsMissing({ slug }: { slug: string }) {
  return (
    <section className="site-empty">
      <p className="site-kicker">[ NO DISPATCH ]</p>
      <h1>release not found</h1>
      <p>
        No news item matches <code>{slug}</code>.
      </p>
      <Link to="/news" className="site-inline-link">
        <ArrowLeft size={14} /> return to dispatches
      </Link>
    </section>
  )
}

export function NewsPage() {
  const { slug } = useParams()
  const release = slug ? RELEASES.find((item) => item.slug === slug) : undefined

  return (
    <SiteShell className="site-shell--news">
      {slug ? (
        release ? <ReleaseDetail release={release} /> : <NewsMissing slug={slug} />
      ) : (
        <>
          <header className="news-hero">
            <div className="news-hero__left">
              <p className="site-kicker">[ RELEASE DISPATCHES ]</p>
              <h1>field updates</h1>
            </div>
            <p className="news-hero__right">
              Short release records from the harness.
            </p>
          </header>
          <section className="release-index" aria-label="Release dispatches">
            {RELEASES.map((item) => (
              <article key={item.slug}>
                <div className="release-index__meta">
                  <span>{item.version}</span>
                  <time dateTime={item.date}>{item.date}</time>
                </div>
                <div className="release-index__body">
                  <h2>
                    <Link to={`/news/${item.slug}`}>{item.title}</Link>
                  </h2>
                  <p>{item.summary}</p>
                  <Link to={`/news/${item.slug}`} className="site-inline-link">
                    open dispatch <ArrowUpRight size={14} aria-hidden="true" />
                  </Link>
                </div>
              </article>
            ))}
          </section>
        </>
      )}
    </SiteShell>
  )
}
