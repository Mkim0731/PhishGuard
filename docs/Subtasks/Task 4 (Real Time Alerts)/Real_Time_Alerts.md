# Real-Time Alerts

## Goal
Notify users instantly when a phishing email is detected via Chrome extension integration.

## Existing Technologies & Methodologies

### 1. Chrome Extensions (Manifest V3)
- Allows you to inject scripts into Gmail UI and display alerts.

**Strengths:**
- Tight integration with browser UI.
- Lightweight and efficient.

**Weaknesses:**
- Requires permission setup.
- Gmail DOM structure is fragile and may change.

**Docs:** [Chrome Extension Docs](https://developer.chrome.com/docs/extensions/mv3/)

---

### 2. JS Alerts / React Popups
- Simple frontend techniques to display phishing warnings.

**Strengths:**
- Easy to implement and customize.
- No external dependencies.

**Weaknesses:**
- Alerts can be intrusive and break UX flow.

**Starter Kit:** [Chrome Extension Starter](https://github.com/abhijithvijayan/awesome-chrome-extensions)

---

### 3. Email Security Gateways (e.g., Proofpoint, Mimecast)
- Trigger real-time banners or warnings on suspicious messages.

**Strengths:**
- Enterprise-grade threat detection with centralized alerting.

**Weaknesses:**
- Costly, not user-configurable, adds latency in message delivery.

**Starter Kit:** [Chrome Extension Starter](https://github.com/abhijithvijayan/awesome-chrome-extensions)

---

### 4. Push Notification APIs (Web Push / Firebase)
- Notify users outside the inbox (browser/mobile) when a phishing attempt is flagged.

**Strengths:**
- Asynchronous alerting, works beyond browser tab scope.

**Weaknesses:**
- Requires notification permissions and backend infrastructure.

**WebPush API:** https://developer.mozilla.org/en-US/docs/Web/API/Push_API

**Github:** https://github.com/firebase/quickstart-js/tree/master/messaging

