import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import './App.css'

const PHONE_CLASS = 'cell phone'
const DETECT_MAX_BOXES = 40
const DETECT_MIN_SCORE = 0.22
const GUNSHOT_SRC = `${import.meta.env.BASE_URL}gunshot.mp3`

function assetUrl(path) {
  const normalized = path.startsWith('/') ? path.slice(1) : path
  return `${import.meta.env.BASE_URL}${normalized}`
}
const SILENT_WAV =
  'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA'
const PUNISHMENT_DELAY_MS = 1800
const DEFAULT_COUNTDOWN_SEC = 25 * 60

const VISUAL_SCENES = {
  supervisor: {
    name: 'Cyber Supervisor (Default)',
    patrolVideoSrc: '/officer_patrol.mp4',
    punishmentVideoSrc: '/officer_shoot.mp4',
  },
  fireplace: {
    name: 'Cozy Fireplace',
    videoSrc: '/fireplace.mp4',
  },
  rain: {
    name: 'Window Rain',
    videoSrc: '/rain.mp4',
  },
  rainforest: {
    name: 'Cyber Rainforest',
    videoSrc: '/rainforest.mp4',
  },
  classroom: {
    name: 'Study Classroom',
    videoSrc: '/classroom.mp4',
  },
  library: {
    name: 'Quiet Library',
    videoSrc: '/library.mp4',
  },
}

const SUPERVISOR_PATROL_VIDEO_SRC = assetUrl(
  VISUAL_SCENES.supervisor.patrolVideoSrc,
)
const SUPERVISOR_SHOOT_VIDEO_SRC = assetUrl(
  VISUAL_SCENES.supervisor.punishmentVideoSrc,
)

const AMBIENT_SOUNDS = {
  none: { name: 'None', audioSrc: null },
  white: { name: 'Pure White Noise', audioSrc: '/white-noise.mp3' },
  fireplace: {
    name: 'Fireplace Crackling',
    audioSrc: '/fireplace-noise.mp3',
  },
  rain: { name: 'Deep Rain Comfort', audioSrc: '/rain-noise.mp3' },
}

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

const CUSTOM_DURATION_FALLBACK_MIN = 1

function getCustomPreviewSeconds(minutes) {
  if (minutes === '' || minutes === null || minutes === undefined) return 0
  const n = parseInt(String(minutes), 10)
  if (Number.isNaN(n) || n <= 0) return 0
  return Math.min(999, n) * 60
}

function resolveCustomDurationMinutes(
  minutes,
  fallbackMinutes = CUSTOM_DURATION_FALLBACK_MIN,
) {
  if (minutes === '' || minutes === null || minutes === undefined) {
    return fallbackMinutes
  }
  const n = parseInt(String(minutes), 10)
  if (Number.isNaN(n) || n < 1) return fallbackMinutes
  return Math.min(999, n)
}

function resolveCustomDurationSeconds(
  minutes,
  fallbackMinutes = CUSTOM_DURATION_FALLBACK_MIN,
) {
  return resolveCustomDurationMinutes(minutes, fallbackMinutes) * 60
}

function isCustomMinutesInvalidForStart(minutes) {
  if (minutes === '' || minutes === null || minutes === undefined) return true
  const n = parseInt(String(minutes), 10)
  return Number.isNaN(n) || n < 1
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
  const sceneVideoRef = useRef(null)
  const patrolVideoRef = useRef(null)
  const punishmentVideoRef = useRef(null)
  const patrolAutoplayMutedForAutoplayRef = useRef(false)
  const supervisorPatrolAudioUnlockedRef = useRef(false)
  const ambientAudioRef = useRef(null)
  const punishmentDelayTimeoutRef = useRef(null)
  const cameraPanelRef = useRef(null)
  const panelDragRef = useRef({
    active: false,
    startX: 0,
    startY: 0,
    startLeft: 0,
    startTop: 0,
  })

  const [focusMode, setFocusMode] = useState('pomodoro')
  const [customMinutes, setCustomMinutes] = useState('25')
  const [cameraPanelPos, setCameraPanelPos] = useState(getDefaultCameraPanelPos)
  const [isDraggingPanel, setIsDraggingPanel] = useState(false)
  const [isFocusing, setIsFocusing] = useState(false)
  const [countdownRemaining, setCountdownRemaining] = useState(DEFAULT_COUNTDOWN_SEC)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [totalViolations, setTotalViolations] = useState(0)
  const [logs, setLogs] = useState([])
  const [visualScene, setVisualScene] = useState('supervisor')
  const [ambientSound, setAmbientSound] = useState('none')
  const [supervisorPatrolMuted, setSupervisorPatrolMuted] = useState(true)

  const [isShaking, setIsShaking] = useState(false)
  const [status, setStatus] = useState('patrol')
  const [cameraReady, setCameraReady] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const [phoneDetected, setPhoneDetected] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [modelError, setModelError] = useState('')
  const [showEndSessionModal, setShowEndSessionModal] = useState(false)
  const [showCompletionModal, setShowCompletionModal] = useState(false)

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

  const playSupervisorPatrol = useCallback(async () => {
    const v = patrolVideoRef.current
    if (!v) return

    const playMuted = async () => {
      patrolAutoplayMutedForAutoplayRef.current = true
      setSupervisorPatrolMuted(true)
      v.muted = true
      try {
        await v.play()
      } catch {
        // Browser may still block playback until user gesture.
      }
    }

    if (supervisorPatrolAudioUnlockedRef.current) {
      patrolAutoplayMutedForAutoplayRef.current = false
      setSupervisorPatrolMuted(false)
      v.muted = false
      try {
        await v.play()
      } catch {
        await playMuted()
      }
      return
    }

    v.muted = true
    setSupervisorPatrolMuted(true)
    try {
      await v.play()
    } catch {
      // Retry once after a frame (some browsers need muted attribute flushed).
      requestAnimationFrame(() => void playMuted())
    }
  }, [])

  const patrolVideoCallbackRef = useCallback(
    (node) => {
      patrolVideoRef.current = node
      if (node) void playSupervisorPatrol()
    },
    [playSupervisorPatrol],
  )

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

  const pauseAmbientAudio = useCallback(() => {
    if (ambientAudioRef.current) {
      ambientAudioRef.current.pause()
      ambientAudioRef.current.currentTime = 0
    }
  }, [])

  const stopFocusSession = useCallback(() => {
    resetPunishmentState()
    setIsFocusing(false)
    setPhoneDetected(false)
  }, [resetPunishmentState])

  const resetTimersAfterSession = useCallback(() => {
    if (focusMode === 'pomodoro') {
      setCountdownRemaining(DEFAULT_COUNTDOWN_SEC)
    } else if (focusMode === 'custom') {
      setCountdownRemaining(getCustomPreviewSeconds(customMinutes))
    } else {
      setElapsedSeconds(0)
    }
  }, [focusMode, customMinutes])

  const finalizeEndFocus = useCallback(() => {
    if (!isFocusing) return
    pauseAmbientAudio()
    stopFocusSession()
    resetTimersAfterSession()
  }, [isFocusing, pauseAmbientAudio, stopFocusSession, resetTimersAfterSession])

  const handleCountdownComplete = useCallback(() => {
    if (!isFocusing) return
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    pauseAmbientAudio()
    stopFocusSession()
    resetTimersAfterSession()
    setShowCompletionModal(true)
  }, [isFocusing, pauseAmbientAudio, stopFocusSession, resetTimersAfterSession])

  const handleDismissCompletionModal = () => {
    setShowCompletionModal(false)
  }

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

  const handleFocusModeChange = (mode) => {
    setFocusMode(mode)
    if (isFocusing) return
    if (mode === 'pomodoro') {
      setCountdownRemaining(DEFAULT_COUNTDOWN_SEC)
    } else if (mode === 'custom') {
      setCountdownRemaining(getCustomPreviewSeconds(customMinutes))
    }
  }

  const handleCustomMinutesChange = (e) => {
    const raw = e.target.value
    if (raw === '') {
      setCustomMinutes('')
      return
    }
    if (!/^\d+$/.test(raw)) return
    const n = parseInt(raw, 10)
    if (n > 999) return
    setCustomMinutes(raw)
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

    setTotalViolations(0)
    setLogs([])
    setShowCompletionModal(false)
    if (focusMode === 'pomodoro') {
      setCountdownRemaining(DEFAULT_COUNTDOWN_SEC)
    } else if (focusMode === 'custom') {
      const resolvedMinutes = resolveCustomDurationMinutes(customMinutes)
      if (isCustomMinutesInvalidForStart(customMinutes)) {
        setCustomMinutes(String(resolvedMinutes))
      }
      setCountdownRemaining(resolveCustomDurationSeconds(customMinutes))
    } else {
      setElapsedSeconds(0)
    }

    supervisorPatrolAudioUnlockedRef.current = true
    patrolAutoplayMutedForAutoplayRef.current = false
    setSupervisorPatrolMuted(false)
    const patrol = patrolVideoRef.current
    if (patrol) {
      patrol.muted = false
      void patrol.play().catch(() => {})
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
      if (focusMode === 'pomodoro' || focusMode === 'custom') {
        setCountdownRemaining((prev) => {
          if (prev <= 1) {
            queueMicrotask(() => handleCountdownComplete())
            return 0
          }
          return prev - 1
        })
      } else {
        setElapsedSeconds((prev) => prev + 1)
      }
    }, 1000)

    return () => clearInterval(tick)
  }, [isFocusing, focusMode, handleCountdownComplete])

  useEffect(() => {
    if (!isFocusing || !cameraReady || !modelReady || showCompletionModal) return

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
  }, [isFocusing, cameraReady, modelReady, showCompletionModal])

  useEffect(() => {
    if (!isFocusing || !cameraReady || !modelReady || showCompletionModal) {
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
      setTotalViolations((c) => c + 1)
    }

    const isSupervisorScene = visualScene === 'supervisor'
    const canArmSupervisorPunishment =
      risingEdge &&
      isSupervisorScene &&
      !executionPendingRef.current &&
      status === 'patrol' &&
      !disarmUntilPhoneClearRef.current

    if (!canArmSupervisorPunishment) return

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
    visualScene,
    showCompletionModal,
    clearPunishmentDelayTimer,
  ])

  useEffect(() => () => clearPunishmentDelayTimer(), [clearPunishmentDelayTimer])

  const handleVisualSceneChange = (nextScene) => {
    if (nextScene !== 'supervisor' && status === 'shoot') {
      resetPunishmentState()
    }
    setVisualScene(nextScene)
  }

  useEffect(() => {
    const sound = AMBIENT_SOUNDS[ambientSound]
    if (!sound?.audioSrc || !isFocusing || showCompletionModal) {
      if (ambientAudioRef.current) {
        ambientAudioRef.current.pause()
        ambientAudioRef.current = null
      }
      return
    }

    const audio = new Audio(assetUrl(sound.audioSrc))
    audio.loop = true
    audio.volume = 0.55
    ambientAudioRef.current = audio
    void audio.play().catch(() => {})

    return () => {
      audio.pause()
      ambientAudioRef.current = null
    }
  }, [ambientSound, isFocusing, showCompletionModal])

  const isSupervisorScene = visualScene === 'supervisor'
  const showSupervisorPunishment =
    isSupervisorScene && status === 'shoot' && isFocusing

  useEffect(() => {
    if (visualScene !== 'supervisor') {
      patrolAutoplayMutedForAutoplayRef.current = false
      if (!supervisorPatrolAudioUnlockedRef.current) {
        setSupervisorPatrolMuted(true)
      }
      return
    }

    void playSupervisorPatrol()

    const v = patrolVideoRef.current
    if (!v) return

    const retry = () => void playSupervisorPatrol()
    v.addEventListener('loadeddata', retry)
    v.addEventListener('canplay', retry)
    return () => {
      v.removeEventListener('loadeddata', retry)
      v.removeEventListener('canplay', retry)
    }
  }, [visualScene, playSupervisorPatrol])

  useEffect(() => {
    if (visualScene === 'supervisor') return
    const v = sceneVideoRef.current
    if (!v) return
    void v.load()
    void v.play().catch(() => {})
  }, [visualScene])

  useEffect(() => {
    if (!showSupervisorPunishment) return
    const v = punishmentVideoRef.current
    if (!v) return
    v.currentTime = 0
    void v.play().catch(() => {})
  }, [showSupervisorPunishment])

  const handleSupervisorShootEnded = () => {
    if (!punishmentDoneRef.current) {
      const v = punishmentVideoRef.current
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

  const isCountdownMode = focusMode === 'pomodoro' || focusMode === 'custom'
  const timerDisplay = isCountdownMode
    ? formatDuration(
        !isFocusing && focusMode === 'custom'
          ? getCustomPreviewSeconds(customMinutes)
          : countdownRemaining,
      )
    : formatDuration(elapsedSeconds)

  const timerModeHint =
    focusMode === 'pomodoro'
      ? 'Pomodoro Mode · 25 min default'
      : focusMode === 'custom'
        ? customMinutes === ''
          ? 'Custom Mode · Set minutes'
          : `Custom Mode · ${customMinutes} min`
        : 'Stopwatch Mode · Count Up'

  const aiActive = isFocusing && cameraReady && modelReady

  const showCornerMonitoring = aiActive && !phoneDetected && !cameraError && !modelError
  const showPhoneBanner = phoneDetected && aiActive && isSupervisorScene
  const mainClassName = [
    'app',
    showPhoneBanner ? 'app--phone-alert' : '',
    isShaking ? 'shake-effect' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <main className={mainClassName}>
      <div
        className={[
          'scene-bg-video',
          showSupervisorPunishment ? 'scene-bg-video--punishment' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        aria-hidden={true}
      >
        {isSupervisorScene ? (
          <>
            <video
              ref={patrolVideoCallbackRef}
              className="officer-bg-video officer-bg-video--patrol"
              src={SUPERVISOR_PATROL_VIDEO_SRC}
              loop
              autoPlay
              playsInline
              preload="auto"
              muted={supervisorPatrolMuted}
              onLoadedData={() => void playSupervisorPatrol()}
              onCanPlay={() => void playSupervisorPatrol()}
            />
            {showSupervisorPunishment && (
              <video
                ref={punishmentVideoRef}
                className="officer-bg-video officer-bg-video--punishment"
                src={SUPERVISOR_SHOOT_VIDEO_SRC}
                autoPlay
                playsInline
                preload="auto"
                muted={false}
                onEnded={handleSupervisorShootEnded}
              />
            )}
          </>
        ) : (
          <video
            ref={sceneVideoRef}
            className="officer-bg-video"
            src={assetUrl(VISUAL_SCENES[visualScene].videoSrc)}
            loop
            autoPlay
            playsInline
            preload="auto"
            muted
          />
        )}
      </div>

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
            <span className="dashboard-mode-hint">{timerModeHint}</span>
          </div>

          <div className="dashboard-stats">
            <div className="stat-card">
              <span className="stat-label">Total Violations</span>
              <span className="stat-value stat-value--alert">{totalViolations}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">AI Detection</span>
              <span className={`stat-value ${aiActive ? 'stat-value--ok' : ''}`}>
                {aiActive ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>

          <div className="dashboard-env">
            <h3 className="env-title">Environment</h3>
            <label className="env-field">
              <span className="env-label">Visual Scene</span>
              <div className="env-select-wrap">
                <select
                  className="env-select"
                  value={visualScene}
                  disabled={isFocusing}
                  onChange={(e) => handleVisualSceneChange(e.target.value)}
                >
                  {Object.entries(VISUAL_SCENES).map(([key, scene]) => (
                    <option key={key} value={key}>
                      {scene.name}
                    </option>
                  ))}
                </select>
              </div>
            </label>
            <label className="env-field">
              <span className="env-label">Ambient Sound</span>
              <div className="env-select-wrap">
                <select
                  className="env-select"
                  value={ambientSound}
                  disabled={isFocusing}
                  onChange={(e) => setAmbientSound(e.target.value)}
                >
                  {Object.entries(AMBIENT_SOUNDS).map(([key, sound]) => (
                    <option key={key} value={key}>
                      {sound.name}
                    </option>
                  ))}
                </select>
              </div>
            </label>
            {!isSupervisorScene && (
              <p className="env-note">
                Non-supervisor scenes log violations silently (no shoot FX).
              </p>
            )}
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
            <label className={`mode-option${focusMode === 'pomodoro' ? ' mode-option--active' : ''}`}>
              <input
                type="radio"
                name="focusMode"
                value="pomodoro"
                checked={focusMode === 'pomodoro'}
                disabled={isFocusing}
                onChange={() => handleFocusModeChange('pomodoro')}
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
                onChange={() => handleFocusModeChange('stopwatch')}
              />
              Stopwatch Mode
            </label>
            <label className={`mode-option${focusMode === 'custom' ? ' mode-option--active' : ''}`}>
              <input
                type="radio"
                name="focusMode"
                value="custom"
                checked={focusMode === 'custom'}
                disabled={isFocusing}
                onChange={() => handleFocusModeChange('custom')}
              />
              Custom Mode
            </label>
          </div>

          {focusMode === 'custom' && (
            <div className="custom-mode-field">
              <label className="custom-mode-label" htmlFor="custom-minutes">
                Duration
              </label>
              <div className="custom-minutes-wrap">
                <input
                  id="custom-minutes"
                  type="number"
                  className="custom-minutes-input"
                  min={0}
                  max={999}
                  step={1}
                  placeholder="Minutes"
                  value={customMinutes}
                  disabled={isFocusing}
                  onChange={handleCustomMinutesChange}
                  aria-label="Custom focus duration in minutes"
                />
                <span className="custom-minutes-suffix" aria-hidden="true">
                  min
                </span>
              </div>
            </div>
          )}

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
                Camera and phone detection start after you click Start Focus. Pomodoro and
                custom countdown sessions end automatically when the timer reaches zero.
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

        {showCompletionModal && (
          <div
            className="modal-overlay modal-overlay--completion"
            role="dialog"
            aria-modal="true"
            aria-labelledby="completion-modal-title"
          >
            <div
              className="modal-card modal-card--completion"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 id="completion-modal-title" className="modal-title modal-title--completion">
                Time&apos;s Up! ⏱️
              </h2>
              <p className="modal-message">
                Great job! You have successfully completed your focus session. It&apos;s time to
                take a well-deserved break!
              </p>
              <div className="modal-actions modal-actions--completion">
                <button
                  type="button"
                  className="modal-btn modal-btn--completion"
                  onClick={handleDismissCompletionModal}
                >
                  Got it, Rest Now
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
