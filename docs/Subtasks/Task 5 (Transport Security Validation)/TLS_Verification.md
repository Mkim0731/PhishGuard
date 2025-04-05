# Gmail Encryption & TLS Verification

## Goal
Verify whether emails are transmitted securely via TLS by inspecting headers or sending test messages.

## Existing Technologies & Methodologies

### 1. Header Parsing
- Analyze `Received:` headers for `TLS` or `ESMTPS` indicators.

**Strengths:**
- Works passively on received emails.
- Lightweight and simple.

**Weaknesses:**
- Header format can vary.
- TLS presence ≠ strong encryption.

**Tutorial:** [Email Header Parsing](https://www.geeksforgeeks.org/how-to-extract-email-headers-in-python/)
**Docs:** [Python smtplib](https://docs.python.org/3/library/smtplib.html)
---

### 2. `smtplib` TLS Tests
- Actively verify TLS by sending test emails with TLS required.

**Strengths:**
- Confirms if a server supports STARTTLS.

**Weaknesses:**
- Only verifies outbound, not inbound encryption.

**Docs:** [Python smtplib](https://docs.python.org/3/library/smtplib.html)

---

### 3. Gmail TLS Status
- Check Google's Transparency Report for TLS adoption metrics.

**Link:** [Google Safer Email Report](https://transparencyreport.google.com/safer-email/overview)
