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
