import { useEffect, useRef, useState } from 'react'
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

function isAbortLikeError(err) {
  if (!err) return false
  if (err.name === 'AbortError') return true
  return /aborted|abort/i.test(String(err.message || ''))
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
  const prevPhoneDetectedRef = useRef(false)
  const prevPhoneForOfficerRef = useRef(false)
  const officerBgVideoRef = useRef(null)

  const [isMonitoring, setIsMonitoring] = useState(false)
  const [status, setStatus] = useState('patrol')
  const [cameraReady, setCameraReady] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const [phoneDetected, setPhoneDetected] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [modelError, setModelError] = useState('')

  const handleStartMonitoring = () => {
    if (isMonitoring) return

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

    prevPhoneDetectedRef.current = false
    prevPhoneForOfficerRef.current = false
    setIsMonitoring(true)
  }

  const handleStopMonitoring = () => {
    if (!isMonitoring) return
    prevPhoneDetectedRef.current = false
    prevPhoneForOfficerRef.current = false
    setStatus('patrol')
    const audio = gunshotAudioRef.current
    if (audio) {
      audio.pause()
      audio.currentTime = 0
    }
    setIsMonitoring(false)
    setPhoneDetected(false)
  }

  useEffect(() => {
    if (!isMonitoring) return

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

    setupCamera()

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
      setCameraReady(false)
      setPhoneDetected(false)
      setCameraError('')
    }
  }, [isMonitoring])

  useEffect(() => {
    let cancelled = false

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

    setupModel()

    return () => {
      cancelled = true
      if (modelRef.current) {
        modelRef.current.dispose()
        modelRef.current = null
      }
      setModelReady(false)
      setModelError('')
    }
  }, [])

  useEffect(() => {
    if (!cameraReady || !modelReady) return

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
  }, [cameraReady, modelReady])

  useEffect(() => {
    if (!isMonitoring || !cameraReady || !modelReady) {
      prevPhoneDetectedRef.current = phoneDetected
      return
    }

    const risingEdge = phoneDetected && !prevPhoneDetectedRef.current
    if (!risingEdge) {
      prevPhoneDetectedRef.current = phoneDetected
      return
    }

    prevPhoneDetectedRef.current = phoneDetected
    const audio = gunshotAudioRef.current
    if (!audio) return
    audio.currentTime = 0
    audio.play().catch(() => {})
  }, [phoneDetected, isMonitoring, cameraReady, modelReady])

  useEffect(() => {
    if (!phoneDetected) {
      prevPhoneForOfficerRef.current = false
      return
    }
    const edge = !prevPhoneForOfficerRef.current
    prevPhoneForOfficerRef.current = true
    if (edge) setStatus('shoot')
  }, [phoneDetected])

  useEffect(() => {
    const v = officerBgVideoRef.current
    if (!v) return
    void v.play().catch(() => {})
  }, [status])

  const handleOfficerShootEnded = () => setStatus('patrol')

  const showCornerMonitoring =
    isMonitoring && cameraReady && modelReady && !phoneDetected && !cameraError && !modelError
  const showPhoneBanner = phoneDetected && cameraReady && modelReady
  const mainClassName = ['app', showPhoneBanner ? 'app--phone-alert' : '', status === 'shoot' ? 'shake-effect' : '']
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

        <header className="top-bar">
          <h1>Cyber-Proctor</h1>
          <p>AI 监督学习演示版（手机检测）</p>
          <div className="start-row">
            {!isMonitoring ? (
              <button
                type="button"
                className="start-monitoring-btn"
                onClick={handleStartMonitoring}
              >
                开始监控
              </button>
            ) : (
              <>
                <button type="button" className="monitoring-active-btn" disabled>
                  正在监控
                </button>
                <button
                  type="button"
                  className="stop-monitoring-btn"
                  onClick={handleStopMonitoring}
                >
                  停止监控
                </button>
              </>
            )}
            {!isMonitoring && (
              <span className="start-hint">请先点击「开始监控」授权摄像头；提示音仅在检测到手机时播放</span>
            )}
          </div>
        </header>

        <section className="stage">
          <div className="camera-panel">
            <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
            <canvas ref={canvasRef} className="overlay" />
          </div>

          <aside className="supervisor">
            <div className="avatar" aria-hidden="true">
              🕵️
            </div>
            <div>
              <h2>监督者</h2>
              <p>
                {!isMonitoring
                  ? '点击「开始监控」后才会开启摄像头。'
                  : modelReady
                    ? '模型已就绪，持续巡查中。'
                    : '正在加载 AI 模型...'}
              </p>
              {isMonitoring && !cameraReady && !cameraError && <p>正在请求摄像头...</p>}
            </div>
          </aside>
        </section>

        <footer className="info">
          <span>Camera: {!isMonitoring ? '未启动' : cameraReady ? 'Ready' : 'Loading'}</span>
          <span>Model: {modelReady ? 'Ready' : 'Loading'}</span>
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
