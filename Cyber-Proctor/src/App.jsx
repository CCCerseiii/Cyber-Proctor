import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import './App.css'

const PHONE_CLASS = 'cell phone'
const DETECT_MAX_BOXES = 40
const DETECT_MIN_SCORE = 0.22
const GUNSHOT_SRC = `${import.meta.env.BASE_URL}gunshot.mp3`
const OFFICER_PATROL_SRC = `${import.meta.env.BASE_URL}officer_patrol.mp4`
const OFFICER_SHOOT_SRC = `${import.meta.env.BASE_URL}officer_shoot.mp4`
const SILENT_WAV =
  'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA'
const PUNISHMENT_DELAY_MS = 1800
const DEFAULT_COUNTDOWN_SEC = 25 * 60

function isAbortLikeError(err) {
  if (!err) return false
  if (err.name === 'AbortError') return true
  return /aborted|abort/i.test(String(err.message || ''))
}

function formatClockTime(date = new Date()) {
  return date.toLocaleTimeString('en-US', { hour12: false })
}

function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':')
}

function getDefaultCameraPanelPos() {
  if (typeof window === 'undefined') return { top: 78, left: 20 }
  const width = Math.min(250, window.innerWidth - 40)
  return { top: 78, left: window.innerWidth - width - 20 }
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const modelRef = useRef(null)
  const streamRef = useRef(null)
  const rafRef = useRef(null)
  const predictingRef = useRef(false)
  const lastPredictionAtRef = useRef(0)
  const gunshotAudioRef = useRef(null)
  const audioContextRef = useRef(null)
  const prevPhoneRef = useRef(false)
  const executionPendingRef = useRef(false)
  const punishmentDoneRef = useRef(false)
  const disarmUntilPhoneClearRef = useRef(false)
  const officerBgVideoRef = useRef(null)
  const punishmentDelayTimeoutRef = useRef(null)
  const cameraPanelRef = useRef(null)
  const panelDragRef = useRef({
    active: false,
    startX: 0,
    startY: 0,
    startLeft: 0,
    startTop: 0,
  })

  const [focusMode, setFocusMode] = useState('countdown')
  const [cameraPanelPos, setCameraPanelPos] = useState(getDefaultCameraPanelPos)
  const [isDraggingPanel, setIsDraggingPanel] = useState(false)
  const [isFocusing, setIsFocusing] = useState(false)
  const [countdownRemaining, setCountdownRemaining] = useState(DEFAULT_COUNTDOWN_SEC)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [violationCount, setViolationCount] = useState(0)
  const [logs, setLogs] = useState([])

  const [isShaking, setIsShaking] = useState(false)
  const [status, setStatus] = useState('patrol')
  const [cameraReady, setCameraReady] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const [phoneDetected, setPhoneDetected] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [modelError, setModelError] = useState('')
  const [showEndSessionModal, setShowEndSessionModal] = useState(false)

  const clampCameraPanelPos = useCallback((left, top) => {
    const el = cameraPanelRef.current
    const w = el?.offsetWidth ?? Math.min(250, window.innerWidth - 40)
    const h = el?.offsetHeight ?? w * (9 / 16)
    const maxLeft = Math.max(0, window.innerWidth - w)
    const maxTop = Math.max(0, window.innerHeight - h)
    return {
      left: Math.min(Math.max(0, left), maxLeft),
      top: Math.min(Math.max(0, top), maxTop),
    }
  }, [])

  const handleCameraPanelDragStart = (e) => {
    if (e.button !== 0) return
    e.preventDefault()
    panelDragRef.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      startLeft: cameraPanelPos.left,
      startTop: cameraPanelPos.top,
    }
    setIsDraggingPanel(true)
  }

  useEffect(() => {
    const onMove = (e) => {
      if (!panelDragRef.current.active) return
      const dx = e.clientX - panelDragRef.current.startX
      const dy = e.clientY - panelDragRef.current.startY
      setCameraPanelPos(
        clampCameraPanelPos(
          panelDragRef.current.startLeft + dx,
          panelDragRef.current.startTop + dy,
        ),
      )
    }
    const onUp = () => {
      if (!panelDragRef.current.active) return
      panelDragRef.current.active = false
      setIsDraggingPanel(false)
      setCameraPanelPos((pos) => clampCameraPanelPos(pos.left, pos.top))
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [clampCameraPanelPos])

  useEffect(() => {
    const onResize = () => {
      setCameraPanelPos((pos) => clampCameraPanelPos(pos.left, pos.top))
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [clampCameraPanelPos])

  useEffect(() => {
    if (!isFocusing) return
    const id = requestAnimationFrame(() => {
      setCameraPanelPos((pos) => clampCameraPanelPos(pos.left, pos.top))
    })
    return () => cancelAnimationFrame(id)
  }, [isFocusing, cameraReady, modelReady, clampCameraPanelPos])

  const clearPunishmentDelayTimer = useCallback(() => {
    if (punishmentDelayTimeoutRef.current) {
      clearTimeout(punishmentDelayTimeoutRef.current)
      punishmentDelayTimeoutRef.current = null
    }
  }, [])

  const resetPunishmentState = useCallback(() => {
    clearPunishmentDelayTimer()
    executionPendingRef.current = false
    punishmentDoneRef.current = false
    disarmUntilPhoneClearRef.current = false
    prevPhoneRef.current = false
    setIsShaking(false)
    setStatus('patrol')
    const audio = gunshotAudioRef.current
    if (audio) {
      audio.pause()
      audio.currentTime = 0
    }
  }, [clearPunishmentDelayTimer])

  const stopFocusSession = useCallback(() => {
    resetPunishmentState()
    setIsFocusing(false)
    setPhoneDetected(false)
  }, [resetPunishmentState])

  const resetTimersAfterSession = useCallback(() => {
    if (focusMode === 'countdown') {
      setCountdownRemaining(DEFAULT_COUNTDOWN_SEC)
    } else {
      setElapsedSeconds(0)
    }
  }, [focusMode])

  const finalizeEndFocus = useCallback(() => {
    if (!isFocusing) return
    stopFocusSession()
    resetTimersAfterSession()
  }, [isFocusing, stopFocusSession, resetTimersAfterSession])

  const handleRequestEndFocus = () => {
    if (!isFocusing) return
    setShowEndSessionModal(true)
  }

  const handleCancelEndFocus = () => {
    setShowEndSessionModal(false)
  }

  const handleConfirmEndFocus = () => {
    setShowEndSessionModal(false)
    finalizeEndFocus()
  }

  const handleStartFocus = () => {
    if (isFocusing) return

    const ACtx = window.AudioContext || window.webkitAudioContext
    if (ACtx) {
      const ctx = audioContextRef.current ?? new ACtx()
      audioContextRef.current = ctx
      void ctx.resume()
    } else {
      const silent = new Audio(SILENT_WAV)
      silent.volume = 0
      void silent.play().then(() => silent.pause()).catch(() => {})
    }

    const audio = new Audio(GUNSHOT_SRC)
    audio.preload = 'auto'
    gunshotAudioRef.current = audio

    setViolationCount(0)
    setLogs([])
    if (focusMode === 'countdown') {
      setCountdownRemaining(DEFAULT_COUNTDOWN_SEC)
    } else {
      setElapsedSeconds(0)
    }
    setIsFocusing(true)
  }

  useEffect(() => {
    if (!isFocusing) return

    let cancelled = false
    const video = videoRef.current
    const canvas = canvasRef.current

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        })

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        streamRef.current = stream
        const videoEl = videoRef.current
        if (videoEl) {
          videoEl.srcObject = stream
          try {
            await videoEl.play()
          } catch (playErr) {
            if (cancelled || isAbortLikeError(playErr)) {
              stream.getTracks().forEach((track) => track.stop())
              streamRef.current = null
              return
            }
            setCameraError(`Unable to play camera feed: ${playErr.message}`)
            stream.getTracks().forEach((track) => track.stop())
            streamRef.current = null
            return
          }
        }

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          streamRef.current = null
          return
        }

        setCameraError('')
        setCameraReady(true)
      } catch (cameraErr) {
        if (cancelled || isAbortLikeError(cameraErr)) return
        setCameraError(`Unable to access camera: ${cameraErr.message}`)
      }
    }

    const setupModel = async () => {
      try {
        await tf.ready()
        const model = await cocoSsd.load({ base: 'mobilenet_v2' })
        if (cancelled) {
          model.dispose()
          return
        }
        modelRef.current = model
        setModelError('')
        setModelReady(true)
      } catch (modelErr) {
        if (!cancelled) setModelError(`Failed to load model: ${modelErr.message}`)
      }
    }

    setupCamera()
    setupModel()

    return () => {
      cancelled = true
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
        streamRef.current = null
      }
      if (video) video.srcObject = null
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      if (modelRef.current) {
        modelRef.current.dispose()
        modelRef.current = null
      }
      setCameraReady(false)
      setModelReady(false)
      setPhoneDetected(false)
      setCameraError('')
      setModelError('')
    }
  }, [isFocusing])

  useEffect(() => {
    if (!isFocusing) return

    const tick = window.setInterval(() => {
      if (focusMode === 'countdown') {
        setCountdownRemaining((prev) => {
          if (prev <= 1) {
            queueMicrotask(() => finalizeEndFocus())
            return 0
          }
          return prev - 1
        })
      } else {
        setElapsedSeconds((prev) => prev + 1)
      }
    }, 1000)

    return () => clearInterval(tick)
  }, [isFocusing, focusMode, finalizeEndFocus])

  useEffect(() => {
    if (!isFocusing || !cameraReady || !modelReady) return

    const detect = async () => {
      const videoEl = videoRef.current
      const canvasEl = canvasRef.current
      const model = modelRef.current
      if (!videoEl || !canvasEl || !model) {
        rafRef.current = requestAnimationFrame(detect)
        return
      }
      if (videoEl.readyState < 2 || predictingRef.current) {
        rafRef.current = requestAnimationFrame(detect)
        return
      }

      const now = performance.now()
      if (now - lastPredictionAtRef.current < 120) {
        rafRef.current = requestAnimationFrame(detect)
        return
      }

      predictingRef.current = true
      lastPredictionAtRef.current = now

      try {
        const predictions = await model.detect(videoEl, DETECT_MAX_BOXES, DETECT_MIN_SCORE)
        const phones = predictions.filter((item) => item.class === PHONE_CLASS)
        setPhoneDetected(phones.length > 0)

        canvasEl.width = videoEl.videoWidth
        canvasEl.height = videoEl.videoHeight
        const ctx = canvasEl.getContext('2d')
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height)
        ctx.lineWidth = 3
        ctx.font = '16px Arial'

        phones.forEach((prediction) => {
          const [x, y, width, height] = prediction.bbox
          ctx.strokeStyle = '#ff4d4f'
          ctx.fillStyle = '#ff4d4f'
          ctx.strokeRect(x, y, width, height)
          ctx.fillRect(x, y - 24, 80, 24)
          ctx.fillStyle = '#fff'
          ctx.fillText('Phone', x + 18, y - 7)
        })
      } catch (detectError) {
        setModelError(`Detection interrupted: ${detectError.message}`)
      } finally {
        predictingRef.current = false
        rafRef.current = requestAnimationFrame(detect)
      }
    }

    rafRef.current = requestAnimationFrame(detect)
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [isFocusing, cameraReady, modelReady])

  useEffect(() => {
    if (!isFocusing || !cameraReady || !modelReady) {
      clearPunishmentDelayTimer()
      executionPendingRef.current = false
      punishmentDoneRef.current = false
      disarmUntilPhoneClearRef.current = false
      prevPhoneRef.current = false
      queueMicrotask(() => setIsShaking(false))
      return
    }

    if (!phoneDetected) {
      disarmUntilPhoneClearRef.current = false
    }

    const risingEdge = phoneDetected && !prevPhoneRef.current
    prevPhoneRef.current = phoneDetected

    if (risingEdge) {
      const entry = `${formatClockTime()} — Phone usage detected`
      setLogs((prev) => [entry, ...prev].slice(0, 50))
    }

    const canArm =
      risingEdge &&
      !executionPendingRef.current &&
      status === 'patrol' &&
      !disarmUntilPhoneClearRef.current

    if (!canArm) return

    executionPendingRef.current = true
    punishmentDoneRef.current = false
    setIsShaking(false)
    setStatus('shoot')

    clearPunishmentDelayTimer()
    punishmentDelayTimeoutRef.current = window.setTimeout(() => {
      punishmentDelayTimeoutRef.current = null
      executionPendingRef.current = false
      punishmentDoneRef.current = true
      disarmUntilPhoneClearRef.current = true

      setViolationCount((c) => c + 1)
      setIsShaking(true)
      const audio = gunshotAudioRef.current
      if (audio) {
        audio.volume = 1
        audio.currentTime = 0
        void audio.play().catch(() => {})
      }
    }, PUNISHMENT_DELAY_MS)
  }, [
    phoneDetected,
    isFocusing,
    cameraReady,
    modelReady,
    status,
    clearPunishmentDelayTimer,
  ])

  useEffect(() => () => clearPunishmentDelayTimer(), [clearPunishmentDelayTimer])

  useEffect(() => {
    const v = officerBgVideoRef.current
    if (!v) return
    void v.play().catch(() => {})
  }, [status])

  const handleOfficerShootEnded = () => {
    if (!punishmentDoneRef.current) {
      const v = officerBgVideoRef.current
      if (v) {
        v.currentTime = 0
        void v.play().catch(() => {})
      }
      return
    }
    setStatus('patrol')
    setIsShaking(false)
    punishmentDoneRef.current = false
  }

  const timerDisplay =
    focusMode === 'countdown'
      ? formatDuration(countdownRemaining)
      : formatDuration(elapsedSeconds)

  const aiActive = isFocusing && cameraReady && modelReady
  const showCornerMonitoring = aiActive && !phoneDetected && !cameraError && !modelError
  const showPhoneBanner = phoneDetected && aiActive
  const mainClassName = [
    'app',
    showPhoneBanner ? 'app--phone-alert' : '',
    isShaking ? 'shake-effect' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <main className={mainClassName}>
      <video
        ref={officerBgVideoRef}
        className="officer-bg-video"
        src={status === 'patrol' ? OFFICER_PATROL_SRC : OFFICER_SHOOT_SRC}
        loop={status === 'patrol'}
        muted
        playsInline
        preload="auto"
        aria-hidden={true}
        onEnded={status === 'shoot' ? handleOfficerShootEnded : undefined}
      />

      <div className="app-content">
        {showPhoneBanner && (
          <div className="phone-top-banner" role="alert" aria-live="assertive">
            PUT DOWN YOUR PHONE!
          </div>
        )}

        <aside className="cyber-dashboard" aria-label="Focus statistics dashboard">
          <div className="dashboard-header">
            <span className="dashboard-badge">CYBER-PROCTOR</span>
            <h2 className="dashboard-title">Focus Dashboard</h2>
          </div>

          <div className="dashboard-timer-block">
            <span className="dashboard-label">Current Timer</span>
            <div className="dashboard-timer">{timerDisplay}</div>
            <span className="dashboard-mode-hint">
              {focusMode === 'countdown'
                ? 'Pomodoro Mode · 25 min default'
                : 'Stopwatch Mode · Count Up'}
            </span>
          </div>

          <div className="dashboard-stats">
            <div className="stat-card">
              <span className="stat-label">Total Violations</span>
              <span className="stat-value stat-value--alert">{violationCount}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">AI Detection</span>
              <span className={`stat-value ${aiActive ? 'stat-value--ok' : ''}`}>
                {aiActive ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>

          <div className="dashboard-logs">
            <h3 className="logs-title">Violation Logs</h3>
            <ul className="logs-list">
              {logs.length === 0 ? (
                <li className="logs-empty">No violations recorded</li>
              ) : (
                logs.map((entry, index) => (
                  <li key={`${entry}-${index}`} className="logs-item">
                    {entry}
                  </li>
                ))
              )}
            </ul>
          </div>
        </aside>

        <header className="top-bar">
          <h1>Cyber-Proctor</h1>
          <p>AI-Proctored Focus Sessions</p>

          <div className="focus-mode-row" role="group" aria-label="Focus mode">
            <label className={`mode-option${focusMode === 'countdown' ? ' mode-option--active' : ''}`}>
              <input
                type="radio"
                name="focusMode"
                value="countdown"
                checked={focusMode === 'countdown'}
                disabled={isFocusing}
                onChange={() => setFocusMode('countdown')}
              />
              Pomodoro Mode (25 Min)
            </label>
            <label className={`mode-option${focusMode === 'stopwatch' ? ' mode-option--active' : ''}`}>
              <input
                type="radio"
                name="focusMode"
                value="stopwatch"
                checked={focusMode === 'stopwatch'}
                disabled={isFocusing}
                onChange={() => setFocusMode('stopwatch')}
              />
              Stopwatch Mode
            </label>
          </div>

          <div className="start-row">
            {!isFocusing ? (
              <button type="button" className="start-monitoring-btn" onClick={handleStartFocus}>
                Start Focus
              </button>
            ) : (
              <>
                <button type="button" className="monitoring-active-btn" disabled>
                  Focus In Progress
                </button>
                <button
                  type="button"
                  className="stop-monitoring-btn"
                  onClick={handleRequestEndFocus}
                >
                  End Focus
                </button>
              </>
            )}
            {!isFocusing && (
              <span className="start-hint">
                Camera and phone detection start after you click Start Focus. Pomodoro sessions
                end automatically when the timer reaches zero.
              </span>
            )}
          </div>
        </header>

        <section className="stage">
          {isFocusing && (
            <div
              ref={cameraPanelRef}
              className={`camera-panel${isDraggingPanel ? ' camera-panel--dragging' : ''}`}
              style={{
                left: cameraPanelPos.left,
                top: cameraPanelPos.top,
              }}
            >
              <div
                className="camera-panel-drag-handle"
                onMouseDown={handleCameraPanelDragStart}
                role="presentation"
                title="Drag to move monitor window"
              >
                <span className="camera-panel-drag-title">LIVE MONITOR</span>
                <span className="camera-panel-drag-hint">
                  Camera: {cameraReady ? 'Ready' : 'Loading'} · Model:{' '}
                  {modelReady ? 'Ready' : 'Loading'}
                </span>
              </div>
              <div className="camera-panel-body">
                <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
                <canvas ref={canvasRef} className="overlay" />
              </div>
            </div>
          )}
        </section>

        <footer className="info">
          <span>Focus: {isFocusing ? 'Active' : 'Not started'}</span>
          <span>Camera: {isFocusing ? (cameraReady ? 'Ready' : 'Loading') : '—'}</span>
          <span>Model: {isFocusing ? (modelReady ? 'Ready' : 'Loading') : '—'}</span>
        </footer>

        {showCornerMonitoring && (
          <div className="corner-monitor-msg" role="status" aria-live="polite">
            <p className="monitoring-hint">Monitoring in progress…</p>
          </div>
        )}

        {showEndSessionModal && (
          <div
            className="modal-overlay"
            role="dialog"
            aria-modal="true"
            aria-labelledby="end-session-title"
            onClick={handleCancelEndFocus}
          >
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
              <h2 id="end-session-title" className="modal-title">
                End Focus Session
              </h2>
              <p className="modal-message">
                Are you sure you want to end the current session?
              </p>
              <div className="modal-actions">
                <button
                  type="button"
                  className="modal-btn modal-btn--cancel"
                  onClick={handleCancelEndFocus}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="modal-btn modal-btn--confirm"
                  onClick={handleConfirmEndFocus}
                >
                  End Session
                </button>
              </div>
            </div>
          </div>
        )}

        {(cameraError || modelError) && (
          <div className="error-banner" role="alert">
            {[cameraError, modelError].filter(Boolean).join(' ')}
          </div>
        )}
      </div>
    </main>
  )
}

export default App
