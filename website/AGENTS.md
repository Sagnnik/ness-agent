# Website agent notes — Ness Agent

Landing experience is a **full-viewport TUI** inspired by the real Ness CLI, not a SaaS marketing page. Secondary surfaces (`/home`, `/news`, `/blog`, `/docs`) are normal scroll pages in the same mono telemetry aesthetic.

## Canon reference
- Screenshot: `website/assets/viewport.png`
- Product README: repo root `README.md`
- Tagline: **own the loop**

## Tokens
| Role | Value |
|------|--------|
| Background | `#070a12` |
| Text | slate-300-ish |
| Prompt | green `act >` |
| Hints / rules | muted purple |
| Context gauge | cyan |
| Navbar accent | `#E61919` |
| Font | JetBrains Mono (or equivalent mono) |

## Routes
| Path | Purpose |
|------|---------|
| `/` | Redirect → `/v1b` |
| `/v1b` | Canon TUI + red-accent navbar |
| `/home` | Product overview using landing transcript narrative |
| `/news`, `/news/:slug` | Release dispatches (newest first; currently v0.2.4 + v0.2.3 + v0.2.2 + v0.2.1 + v0.2.0 + v0.1.0) |
| `/blog`, `/blog/:slug` | Folder-based markdown from `content/blog/<slug>/` |
| `/docs`, `/docs/:section` | Overview · SDK · SDK API · CLI · Configuration · Architecture |

## Content layout
```
website/content/blog/<slug>/index.md
website/content/blog/<slug>/assets/
docs/*.md                    # guides + SDK API reference (imported into /docs UI)
```

## Packages
Already present: React 19, Vite, Tailwind 4, `motion` / `framer-motion`, `react-router`, `lucide-react`, `gsap`, `@gsap/react`, JetBrains Mono, markdown/shiki tooling.

Prefer **`motion`** (not mixed Framer APIs). Use **GSAP** for SVG logo, boot, and chrome typewriter on the TUI.

## Interaction contract (TUI)
1. Boot: SVG logo points/chords animate → settle into A-mark.
2. Scroll/arrows advance steps: chrome write → transcript append.
3. Whitelisted slash commands only: `/copy` `/help` `/home` `/news` `/blog` `/docs`.
4. `/copy` copies install command; `/help` appends TUI help in transcript.

## Skills
- Follow: `.agents/skills/industrial-brutalist-ui` (Tactical Telemetry), `image-to-code`.
- Avoid for this surface: `gpt-taste`, `design-taste-frontend`, `minimalist-ui`, imagegen section boards.

## Do not
- Git commit unless the user explicitly asks.
