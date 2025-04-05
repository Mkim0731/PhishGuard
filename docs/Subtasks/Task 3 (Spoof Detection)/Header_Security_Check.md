# Outlook Security Check (SPF, DKIM, DMARC)

## Goal
Examine email headers to detect spoofing using standard security protocols.

## Existing Technologies & Methodologies

### 1. Python Email Module: Parses raw headers (SPF, DKIM, DMARC)
- Parses raw email headers including SPF, DKIM, and DMARC entries.

**Strengths:**
- Built-in Python tool.
- Works across platforms.

**Weaknesses:**
- Low-level; complex for detailed parsing.

**Docs:** [Python Email Module](https://docs.python.org/3/library/email.html)

---

### 2. `mailparser` Python Library
- High-level .eml and header parser.

**Strengths:**
- High-level and easy to use.
- Compatible with `pandas`.
- Simple and efficient.

**Weaknesses:**
- Limited support for non-standard headers.

**GitHub:** [mailparser](https://github.com/SpamScope/mailparser)

---

### 3. `dkimpy` and `pyspf`
- Libraries to validate DKIM signatures and SPF DNS records.
- Validates DKIM signatures and SPF records via DNS.

**Strengths:**
- Protocol-compliant.
- Offers detailed validation feedback.

**Weaknesses:**
- Requires DNS access and error handling.

**GitHub:** [dkimpy](https://github.com/kamailio/dkimpy) | [pyspf](https://github.com/sdgathman/pyspf)


### 4. Header Parsing (TLS): 
- Checks Received headers for TLS/ESMTPS.
- 
- **Strengths:**
Passive and lightweight.

**Weaknesses:**
- Header format inconsistencies; not encryption-proof.
