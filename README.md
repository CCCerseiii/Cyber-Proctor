# 👁️ Cyber-Proctor

> An immersive, AI-proctored focus dashboard designed to help people—especially individuals with **ADHD**—sustain deep attention through interactive computer vision, customized sensory isolation, and tech-noir aesthetics.

---

## 💡 Inspiration & Motivation

Standard productivity apps often fail those who struggle with executive dysfunction or ADHD, offering little immediate feedback to combat distractions. **Cyber-Proctor** re-imagines focus sessions by combining client-side AI monitoring with immersive gamification. 

By delivering instant multi-sensory feedback (visual shifts and tactical audio cues) upon distraction, it provides the external guardrails needed to re-engage the brain, breaking the cycle of casual phone-scrolling while wrapped in a highly tailored, satisfying visual experience.

---

## ✨ Current Features

* **📱 Local AI Phone Detection (`COCO-SSD`)**: Uses an on-device TensorFlow.js model to scan your camera feed and detect `cell phone` usage in real-time. 
* **🔒 Privacy-First Sandbox**: 100% of the computer vision modeling runs completely within your local browser. **No camera data or video feeds are ever uploaded to any server.** * **🎧 Decoupled Sensory Environments**: Toggle independently between high-fidelity looping video backgrounds (Cyber Supervisor, Window Rain, Fireplace) and ambient white noise choices to create your ideal focus sanctuary.
* **⏱️ Flexible Clock Tri-Mode**: Supports standard 25-min Pomodoro countdowns, traditional stopwatches, and fluid, fully adjustable custom countdown inputs.
* **🚨 Responsive Completion Modal**: Blocks the screen on session end with a premium glassmorphic overlay, forcing a manual close to guarantee a well-deserved, mindful break.

---

## 🛠️ Tech Stack

* **Frontend**: React.js, Vite, HTML5 Audio/Video API
* **AI Engine**: `@tensorflow/tfjs`, `@tensorflow-models/coco-ssd`
* **Styling**: Cyberpunk-inspired Glassmorphism via advanced CSS3 backdrop filters

---

## 🗺️ Future Roadmap (Upcoming Highlights)

We are constantly working to turn Cyber-Proctor into the ultimate focusing universe. Here is what's coming next:

* [ ] **AI-Driven Animated Supervisors**: Allow users to upload a single photo of any supervisor (a mentor, celebrity, or historical figure) and use client-side generative AI to animate their expressions/movements.
* [ ] **Custom Media Uploads**: Let users drag and drop their own ambient looping videos or custom background tracks.
* [ ] **Gamified Scenario Punishments**: Expand the reward/punishment animations (e.g., a classroom theme where a teacher turns around and aggressively taps the blackboard upon phone detection).
* [ ] **Camera Toggle Freedom**: Allow users to optionally hide the live webcam preview widget into a compact background toggle without interrupting the AI proctoring backend.

---

## 🚀 Quick Start (Local Setup)

1. Clone the repository:
   ```bash
   git clone [https://github.com/CCCerseiii/Cyber-Proctor.git](https://github.com/CCCerseiii/Cyber-Proctor.git)