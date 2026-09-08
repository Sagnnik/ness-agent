import {
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type TouchEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, Clipboard, CornerDownLeft } from 'lucide-react'
import { useNavigate } from 'react-router'
import { Link } from 'react-router'
import {
  blocksThroughGroup,
  INSTALL_COMMAND,
  KNOWN_COMMANDS,
  POWERSHELL_INSTALL_COMMAND,
  type MergedTranscriptGroup,
  transcriptDisplayItems,
  TRANSCRIPT_CONTAINER_LAST,
  TUI_STEP_GROUPS,
  TUI_STEPS,
} from './steps'
import { DOCS_QUICK_LINKS, docsPath } from '@/content/docsRoutes'
import { NessLogo } from './NessLogo'
import type { CommandOutput, KnownCommand, TuiMode, Variant } from './types'
import { useTheme } from './useTheme'
import { VariantNavbar } from './VariantNavbar'

interface TuiShellProps {
  variant: Variant
}

const COMMAND_HINTS = [
  '/home',
  '/news',
  '/blog',
  '/docs',
  '/help',
  '/copy',
] as const

/** Tab-complete a slash command. `/h` prefers `/help`; `/ho` → `/home`. */
function autocompleteCommand(partial: string): string | null {
  const query = partial.trim().toLowerCase()
  if (!query.startsWith('/') || query.length < 2) return null

  const matches = COMMAND_HINTS.filter((command) => command.startsWith(query))
  if (matches.length === 0) return null
  if (matches.length === 1) return matches[0]

  // Only `/help` and `/home` share a prefix; bare `/h` resolves to help.
  if (query === '/h' && matches.includes('/help')) return '/help'

  let common: string = matches[0]
  for (const match of matches.slice(1)) {
    let end = 0
    while (
      end < common.length &&
      end < match.length &&
      common[end] === match[end]
    ) {
      end += 1
    }
    common = common.slice(0, end)
  }

  return common.length > query.length ? common : null
}

interface BootOverlayProps {
  variant: Variant
  onComplete: () => void
}

function BootOverlay({ variant, onComplete }: BootOverlayProps) {
  return (
    <motion.div
      className="boot-overlay"
      exit={{ opacity: 0 }}
      transition={{ duration: 0.24 }}
      aria-label="Ness Agent booting"
    >
      <div className="boot-overlay__scope">
        <span>NESS_BOOT // {variant.toUpperCase()}</span>
        <span>CHORD_LOCK 14/14</span>
      </div>
      <div className="boot-overlay__stage">
        <NessLogo
          booting
          intensity={1}
          onComplete={onComplete}
        />
        <div className="boot-overlay__log" aria-live="polite">
          <span>mounting prompt layers</span>
          <span>binding operator surface</span>
          <strong>loop ownership confirmed</strong>
        </div>
      </div>
    </motion.div>
  )
}

function StepLines({
  step,
  onCopy,
  copiedCommand,
}: {
  step: (typeof TUI_STEPS)[number]
  onCopy: (command: string) => void
  copiedCommand: string | null
}) {
  return step.lines.map((line, lineIndex) => {
    if (
      step.kind === 'install' &&
      (line === INSTALL_COMMAND || line === POWERSHELL_INSTALL_COMMAND)
    ) {
      const isCopied = copiedCommand === line
      const isWindows = line === POWERSHELL_INSTALL_COMMAND
      return (
        <div className="install-row" key={line}>
          <code>
            <span aria-hidden="true">{isWindows ? 'PS>' : '$'}</span> {line}
          </code>
          <button
            type="button"
            onClick={() => onCopy(line)}
            aria-label={`Copy ${isWindows ? 'Windows' : 'macOS and Linux'} install command`}
            title={`Copy ${isWindows ? 'Windows' : 'macOS and Linux'} install command`}
          >
            {isCopied ? <Check size={14} /> : <Clipboard size={14} />}
            <span>{isCopied ? 'copied' : 'copy'}</span>
          </button>
        </div>
      )
    }

    if (step.kind === 'docs' && lineIndex === 0) {
      return (
        <div className="docs-command" key={line}>
          <CornerDownLeft size={13} aria-hidden="true" />
          <Link className="docs-command__prefix" to="/docs">
            /docs
          </Link>
          <span className="docs-command__links" aria-label="Documentation sections">
            {DOCS_QUICK_LINKS.map((link, index) => (
              <span key={link.slug} className="docs-command__link-item">
                {index > 0 ? (
                  <span className="docs-command__sep" aria-hidden="true">
                    ·
                  </span>
                ) : null}
                <Link to={docsPath(link.slug)}>{link.label}</Link>
              </span>
            ))}
          </span>
        </div>
      )
    }

    const [label, ...rest] = line.split('  ')
    const hasLabel = rest.length > 0 && label.length < 12
    return (
      <p key={line}>
        {hasLabel ? <b>{label}</b> : null}
        {hasLabel ? `  ${rest.join('  ')}` : line}
      </p>
    )
  })
}

function MergedTranscriptStep({
  group,
  appearDelay = 0,
  onCopy,
  copiedCommand,
}: {
  group: MergedTranscriptGroup
  appearDelay?: number
  onCopy: (command: string) => void
  copiedCommand: string | null
}) {
  const mergedSteps = group.blockIds.map((index) => TUI_STEPS[index])

  return (
    <motion.section
      className={`transcript-step transcript-step--${group.modifier}`}
      data-stream={group.dataStream}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: 'easeOut', delay: appearDelay }}
    >
      <div className="transcript-step__rail">
        <span>{group.railLabel}</span>
        <span className="transcript-step__index">
          {String(group.displayIndex).padStart(2, '0')} /{' '}
          {String(TRANSCRIPT_CONTAINER_LAST).padStart(2, '0')}
        </span>
      </div>
      <div className="transcript-step__content">
        {mergedSteps.map((step) => (
          <div
            key={step.id}
            className={`transcript-step__block${
              step.kind === 'manifesto' ? ' transcript-step__block--manifesto' : ''
            }`}
          >
            <span className="transcript-step__eyebrow">{step.eyebrow}</span>
            <h2>{step.heading}</h2>
            <StepLines
              step={step}
              onCopy={onCopy}
              copiedCommand={copiedCommand}
            />
          </div>
        ))}
      </div>
    </motion.section>
  )
}

function TranscriptStep({
  index,
  appearDelay = 0,
  onCopy,
  copiedCommand,
}: {
  index: number
  appearDelay?: number
  onCopy: (command: string) => void
  copiedCommand: string | null
}) {
  const step = TUI_STEPS[index]
  const displayIndex = TRANSCRIPT_CONTAINER_LAST

  return (
    <motion.section
      className={`transcript-step transcript-step--${step.kind}`}
      data-stream={index % 2 === 0 ? 'plan' : 'act'}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: 'easeOut', delay: appearDelay }}
    >
      <div className="transcript-step__rail">
        <span>{step.eyebrow}</span>
        <span className="transcript-step__index">
          {String(displayIndex).padStart(2, '0')} /{' '}
          {String(TRANSCRIPT_CONTAINER_LAST).padStart(2, '0')}
        </span>
      </div>
      <div className="transcript-step__content">
        <h2>{step.heading}</h2>
        <StepLines
          step={step}
          onCopy={onCopy}
          copiedCommand={copiedCommand}
        />
      </div>
    </motion.section>
  )
}

const OUTPUT_TTL_MS: Record<CommandOutput['tone'], number> = {
  success: 2800,
  error: 4200,
  info: 5200,
}

function CommandBlock({ output }: { output: CommandOutput }) {
  return (
    <motion.output
      className={`command-output command-output--${output.tone}`}
      initial={{ opacity: 0, x: -8, height: 'auto' }}
      animate={{ opacity: 1, x: 0, height: 'auto' }}
      exit={{
        opacity: 0,
        x: -8,
        height: 0,
        paddingTop: 0,
        paddingBottom: 0,
        marginTop: 0,
        borderLeftWidth: 0,
      }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
      style={{ overflow: 'hidden' }}
    >
      <span className="command-output__prompt">
        {output.mode} &gt; {output.command}
      </span>
      {output.lines.map((line) => (
        <span key={line}>{line}</span>
      ))}
    </motion.output>
  )
}

export function TuiShell({ variant }: TuiShellProps) {
  const navigate = useNavigate()
  const root = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const transcript = useRef<HTMLDivElement>(null)
  const wheelLock = useRef(false)
  const wheelAcc = useRef(0)
  const touchStart = useRef(0)
  const outputId = useRef(0)
  const outputTimers = useRef(new Map<number, number>())
  const prevVisibleStep = useRef(-1)

  const [booted, setBooted] = useState(false)
  const [stepIndex, setStepIndex] = useState(0)
  const [visibleStep, setVisibleStep] = useState(-1)
  const [command, setCommand] = useState('')
  const [outputs, setOutputs] = useState<CommandOutput[]>([])
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)
  const [mode, setMode] = useState<TuiMode>('act')

  const activeGroup = TUI_STEP_GROUPS[Math.max(0, stepIndex)] ?? TUI_STEP_GROUPS[0]
  const step = TUI_STEPS[activeGroup[activeGroup.length - 1]]
  const { theme } = useTheme()
  const groupCount = TUI_STEP_GROUPS.length

  const dismissOutput = useCallback((id: number) => {
    setOutputs((current) => current.filter((entry) => entry.id !== id))
    const timer = outputTimers.current.get(id)
    if (timer !== undefined) {
      window.clearTimeout(timer)
      outputTimers.current.delete(id)
    }
  }, [])

  const addOutput = useCallback(
    (tone: CommandOutput['tone'], typedCommand: string, lines: readonly string[]) => {
      outputId.current += 1
      const id = outputId.current
      setOutputs((current) => [
        ...current,
        {
          id,
          tone,
          mode,
          command: typedCommand,
          lines,
        },
      ])

      const timer = window.setTimeout(() => {
        dismissOutput(id)
      }, OUTPUT_TTL_MS[tone])
      outputTimers.current.set(id, timer)
    },
    [dismissOutput, mode],
  )

  useEffect(
    () => () => {
      for (const timer of outputTimers.current.values()) {
        window.clearTimeout(timer)
      }
      outputTimers.current.clear()
    },
    [],
  )

  const copyInstall = useCallback(async (installCommand = INSTALL_COMMAND) => {
    try {
      await navigator.clipboard.writeText(installCommand)
      setCopiedCommand(installCommand)
      addOutput('success', '/copy', [`copied: ${installCommand}`])
      window.setTimeout(() => setCopiedCommand(null), 1800)
    } catch {
      addOutput('error', '/copy', [
        'clipboard unavailable; select the install row manually',
      ])
    }
  }, [addOutput])

  const completeBoot = useCallback(() => {
    setBooted(true)
  }, [])

  const changeStep = useCallback((delta: number) => {
    setStepIndex((current) =>
      Math.min(TUI_STEP_GROUPS.length - 1, Math.max(0, current + delta)),
    )
  }, [])

  useEffect(() => {
    if (!booted) return
    setVisibleStep(stepIndex)
  }, [booted, stepIndex])

  useEffect(() => {
    const pane = transcript.current
    if (!pane) return

    const steppedBack = visibleStep < prevVisibleStep.current
    prevVisibleStep.current = visibleStep

    // Stepping back removes the last block in place; forcing scrollTop fights the layout.
    if (steppedBack) return

    // Wait a frame so motion layout settles before pinning to bottom.
    const id = window.requestAnimationFrame(() => {
      pane.scrollTop = pane.scrollHeight
    })
    return () => window.cancelAnimationFrame(id)
  }, [visibleStep, outputs])

  useEffect(() => {
    const shell = root.current
    if (!shell || !booted) return

    const onWheel = (event: WheelEvent) => {
      const pane = transcript.current
      const target = event.target
      if (
        pane &&
        target instanceof Node &&
        pane.contains(target) &&
        pane.scrollHeight > pane.clientHeight + 1
      ) {
        const atTop = pane.scrollTop <= 1
        const atBottom =
          pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 2
        const scrollingDown = event.deltaY > 0
        const scrollingUp = event.deltaY < 0

        // Pinned to bottom: wheel always owns step replay. Mid-transcript only:
        // let native scroll until an edge, then step. (Later steps overflow; without
        // this, scroll-up tries to pan the pane instead of going 7→6→5.)
        if (!atBottom) {
          if ((scrollingDown && !atBottom) || (scrollingUp && !atTop)) {
            wheelAcc.current = 0
            return
          }
        }
      }

      event.preventDefault()
      if (wheelLock.current) return

      // Accumulate small trackpad deltas; a single mouse-notch usually clears this.
      wheelAcc.current += event.deltaY
      if (Math.abs(wheelAcc.current) < 28) return

      wheelLock.current = true
      changeStep(wheelAcc.current > 0 ? 1 : -1)
      wheelAcc.current = 0
      window.setTimeout(() => {
        wheelLock.current = false
      }, 180)
    }

    shell.addEventListener('wheel', onWheel, { passive: false })
    return () => shell.removeEventListener('wheel', onWheel)
  }, [booted, changeStep])

  useEffect(() => {
    if (!booted) return

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        navigate('/home')
        return
      }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        changeStep(event.key === 'ArrowDown' ? 1 : -1)
      }
      if (event.key === 'Tab' && event.shiftKey) {
        event.preventDefault()
        setMode((current) => (current === 'act' ? 'plan' : 'act'))
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [booted, changeStep, navigate])

  const executeCommand = useCallback(
    async (rawCommand: string) => {
      const normalized = rawCommand.trim().toLowerCase()
      if (!normalized) return

      if (!KNOWN_COMMANDS.has(normalized)) {
        addOutput('error', normalized, [
          `unknown command: ${normalized}`,
          'type /help for the command index',
        ])
        return
      }

      const known = normalized as KnownCommand
      if (known === '/copy') {
        await copyInstall()
        return
      }
      if (known === '/help') {
        addOutput('info', known, [
          '/copy  copy the install command',
          '/home  /news  /blog  /docs  open site surfaces',
          '↑/↓ or wheel  replay this session',
          'Tab  autocomplete (/h → /help, /ho → /home)',
          'Shift+Tab  toggle Act/Plan',
          'Esc  return to /home',
        ])
        return
      }

      navigate(known)
    },
    [addOutput, copyInstall, navigate],
  )

  const submitCommand = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const submitted = command
    setCommand('')
    void executeCommand(submitted)
  }

  const onCommandKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Tab' || event.shiftKey) return
    event.preventDefault()
    const completed = autocompleteCommand(command)
    if (completed) setCommand(completed)
  }

  const onTouchStart = (event: TouchEvent<HTMLDivElement>) => {
    touchStart.current = event.changedTouches[0]?.clientY ?? 0
  }

  const onTouchEnd = (event: TouchEvent<HTMLDivElement>) => {
    const end = event.changedTouches[0]?.clientY ?? touchStart.current
    const delta = touchStart.current - end
    if (Math.abs(delta) > 34) changeStep(delta > 0 ? 1 : -1)
  }

  const transcriptSteps = useMemo(
    () => blocksThroughGroup(visibleStep),
    [visibleStep],
  )

  const displayItems = useMemo(
    () => transcriptDisplayItems(visibleStep),
    [visibleStep],
  )

  const revealedBlocks = transcriptSteps.length
  const tokenCount = 1248 + Math.max(0, revealedBlocks) * 420
  const contextPercent = 6 + Math.max(0, visibleStep) * 18

  return (
    <div
      ref={root}
      className={`terminal-page terminal-page--${variant} terminal-page--with-nav`}
      data-theme={theme}
      onTouchStart={onTouchStart}
      onTouchEnd={onTouchEnd}
      onMouseDown={() => input.current?.focus()}
    >
      <VariantNavbar />

      <main className="tui" aria-label={`Ness Agent ${variant} terminal`}>
        <header className="tui__chrome">
          <div className="tui__chrome-row">
            <div className="tui__mark">
              {booted ? <NessLogo /> : null}
            </div>
            <div className="tui__chrome-main">
              <div className="tui__brandline">
                <strong>NessAgent</strong>
                <span>v0.2.4</span>
                <span className="tui__brandline-status" title="Press Esc to open Home">
                  <kbd>Esc</kbd> Home
                </span>
              </div>
              <dl className="tui__status">
                <div>
                  <dt>Session</dt>
                  <dd>{step.session}</dd>
                  <dt>Mode</dt>
                  <dd>{mode}</dd>
                </div>
                <div>
                  <dt>Project</dt>
                  <dd>{step.project}</dd>
                  <dt>Add-ons</dt>
                  <dd>3 Skills</dd>
                </div>
              </dl>
              <div className="tui__hints">
                <span>Hints:</span>
                <span>↑/↓ select</span>
                <i>·</i>
                <span>Tab complete</span>
                <i>·</i>
                <span>Shift+Tab toggle Act/Plan</span>
                {COMMAND_HINTS.map((hint) => (
                  <button
                    type="button"
                    key={hint}
                    onClick={() => {
                      if (
                        hint === '/home' ||
                        hint === '/news' ||
                        hint === '/blog' ||
                        hint === '/docs'
                      ) {
                        navigate(hint)
                      } else {
                        void executeCommand(hint)
                      }
                    }}
                  >
                    {hint}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </header>

        <div
          ref={transcript}
          className="tui__transcript"
          aria-live="polite"
          aria-label="Session transcript"
        >
          <div className="tui__transcript-stack">
            <AnimatePresence initial={false}>
              {displayItems.map((item, itemIndex) => {
                if (item.type === 'merged') {
                  return (
                    <MergedTranscriptStep
                      key={item.group.key}
                      group={item.group}
                      appearDelay={0}
                      onCopy={(installCommand) => void copyInstall(installCommand)}
                      copiedCommand={copiedCommand}
                    />
                  )
                }

                const group = TUI_STEP_GROUPS[visibleStep]
                const withinGroup = group?.indexOf(item.index) ?? -1
                return (
                  <TranscriptStep
                    key={TUI_STEPS[item.index].id}
                    index={item.index}
                    appearDelay={withinGroup >= 0 ? withinGroup * 0.05 : itemIndex * 0.05}
                    onCopy={(installCommand) => void copyInstall(installCommand)}
                    copiedCommand={copiedCommand}
                  />
                )
              })}
              {outputs.map((output) => (
                <CommandBlock output={output} key={output.id} />
              ))}
            </AnimatePresence>
          </div>
        </div>

        <form className="tui__input" onSubmit={submitCommand}>
          <label htmlFor={`${variant}-command`}>
            {mode} &gt;
          </label>
          <input
            ref={input}
            id={`${variant}-command`}
            value={command}
            onChange={(event) => setCommand(event.target.value)}
            onKeyDown={onCommandKeyDown}
            placeholder=" "
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            aria-label="Ness command input"
          />
          <span className="tui__cursor" aria-hidden="true" />
          <button type="submit" tabIndex={-1} aria-label="Run command">
            ↵
          </button>
        </form>

        <footer className="tui__footer">
          <span>↑ {stepIndex}</span>
          <span>↓ {Math.max(0, groupCount - 1 - stepIndex)}</span>
          <span>◉ ${((tokenCount / 1000) * 0.003).toFixed(5)}</span>
          <span className="tui__cwd">~/projects/ness-agent (main)</span>
          <span className="tui__context-label">
            context {tokenCount.toLocaleString()}/1000k used
          </span>
          <span className="tui__gauge" aria-label={`${contextPercent}% context used`}>
            <i style={{ width: `${contextPercent}%` }} />
          </span>
          <span className="tui__model">claude-opus-5</span>
        </footer>
      </main>

      <AnimatePresence>
        {!booted ? (
          <BootOverlay variant={variant} onComplete={completeBoot} />
        ) : null}
      </AnimatePresence>
    </div>
  )
}
