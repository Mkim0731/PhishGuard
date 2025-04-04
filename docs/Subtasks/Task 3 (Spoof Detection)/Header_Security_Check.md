# Outlook Security Check (SPF, DKIM, DMARC)

## Goal
Examine email headers to detect spoofing using standard security protocols.

## Existing Technologies & Methodologies

### 1. Python `email` module
- Parses raw email headers including SPF, DKIM, and DMARC entries.

**Strengths:**
- Built-in Python tool.
- Works across platforms.

**Weaknesses:**
- Verbose and low-level for complex parsing.

**Docs:** [Python Email Module](https://docs.python.org/3/library/email.html)

---

### 2. `mailparser` Python Library
- Simplifies parsing of raw `.eml` messages and headers.

**Strengths:**
- High-level and easy to use.
- Compatible with `pandas`.

**Weaknesses:**
- May not support all edge cases or custom headers.

**GitHub:** [mailparser](https://github.com/SpamScope/mailparser)

---

### 3. `dkimpy` and `pyspf`
- Libraries to validate DKIM signatures and SPF DNS records.

**Strengths:**
- Protocol-compliant.
- Offers detailed validation feedback.

**Weaknesses:**
- Requires exception handling and DNS access.

**GitHub:** [dkimpy](https://github.com/kamailio/dkimpy) | [pyspf](https://github.com/sdgathman/pyspf)
