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
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':')
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

  const [focusMode, setFocusMode] = useState('countdown')
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

  const handleEndFocus = useCallback(() => {
    if (!isFocusing) return
    resetPunishmentState()
    setIsFocusing(false)
    setPhoneDetected(false)
  }, [isFocusing, resetPunishmentState])

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
            setCameraError(`无法播放摄像头画面：${playErr.message}`)
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
        setCameraError(`无法访问摄像头：${cameraErr.message}`)
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
        if (!cancelled) setModelError(`模型加载失败：${modelErr.message}`)
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
            queueMicrotask(() => handleEndFocus())
            return 0
          }
          return prev - 1
        })
      } else {
        setElapsedSeconds((prev) => prev + 1)
      }
    }, 1000)

    return () => clearInterval(tick)
  }, [isFocusing, focusMode, handleEndFocus])

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
          ctx.fillText('手机', x + 18, y - 7)
        })
      } catch (detectError) {
        setModelError(`检测中断：${detectError.message}`)
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
      const entry = `${formatClockTime()} 拿起手机违规`
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

        <aside className="cyber-dashboard" aria-label="专注统计仪表盘">
          <div className="dashboard-header">
            <span className="dashboard-badge">CYBER-PROCTOR</span>
            <h2 className="dashboard-title">专注仪表盘</h2>
          </div>

          <div className="dashboard-timer-block">
            <span className="dashboard-label">当前计时</span>
            <div className="dashboard-timer">{timerDisplay}</div>
            <span className="dashboard-mode-hint">
              {focusMode === 'countdown' ? '倒计时专注 · 默认 25 分钟' : '正计时专注'}
            </span>
          </div>

          <div className="dashboard-stats">
            <div className="stat-card">
              <span className="stat-label">违规次数</span>
              <span className="stat-value stat-value--alert">{violationCount}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">AI 检测</span>
              <span className={`stat-value ${aiActive ? 'stat-value--ok' : ''}`}>
                {aiActive ? '运行中' : '未启动'}
              </span>
            </div>
          </div>

          <div className="dashboard-logs">
            <h3 className="logs-title">违规日志</h3>
            <ul className="logs-list">
              {logs.length === 0 ? (
                <li className="logs-empty">暂无违规记录</li>
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
          <p>AI 监督学习 · 专注模式</p>

          <div className="focus-mode-row" role="group" aria-label="专注模式">
            <label className={`mode-option${focusMode === 'countdown' ? ' mode-option--active' : ''}`}>
              <input
                type="radio"
                name="focusMode"
                value="countdown"
                checked={focusMode === 'countdown'}
                disabled={isFocusing}
                onChange={() => setFocusMode('countdown')}
              />
              倒计时专注（25 分钟）
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
              正计时专注
            </label>
          </div>

          <div className="start-row">
            {!isFocusing ? (
              <button type="button" className="start-monitoring-btn" onClick={handleStartFocus}>
                开始专注
              </button>
            ) : (
              <>
                <button type="button" className="monitoring-active-btn" disabled>
                  专注进行中
                </button>
                <button type="button" className="stop-monitoring-btn" onClick={handleEndFocus}>
                  结束专注
                </button>
              </>
            )}
            {!isFocusing && (
              <span className="start-hint">
                点击「开始专注」后才会启动摄像头与手机检测；倒计时归零将自动结束
              </span>
            )}
          </div>
        </header>

        <section className="stage">
          {isFocusing && (
            <div className="camera-panel">
              <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
              <canvas ref={canvasRef} className="overlay" />
            </div>
          )}

          <aside className="supervisor">
            <div className="avatar" aria-hidden="true">
              🕵️
            </div>
            <div>
              <h2>监督者</h2>
              <p>
                {!isFocusing
                  ? '选择专注模式后点击「开始专注」。'
                  : modelReady && cameraReady
                    ? '模型与摄像头已就绪，持续巡查中。'
                    : '正在加载 AI 模型与摄像头...'}
              </p>
              {isFocusing && !cameraReady && !cameraError && <p>正在请求摄像头...</p>}
            </div>
          </aside>
        </section>

        <footer className="info">
          <span>专注: {isFocusing ? '进行中' : '未开始'}</span>
          <span>Camera: {isFocusing ? (cameraReady ? 'Ready' : 'Loading') : '—'}</span>
          <span>Model: {isFocusing ? (modelReady ? 'Ready' : 'Loading') : '—'}</span>
        </footer>

        {showCornerMonitoring && (
          <div className="corner-monitor-msg" role="status" aria-live="polite">
            <p className="monitoring-hint">正在监控中...</p>
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
