"""
ShopEase AI — Security Hardening Module
Provides:
1. Input length validation & capping
2. Per-session sliding window rate limiting
3. PII detection and redaction (emails, phone numbers, credit card numbers)
4. Transactional confirmation guard before place/cancel/modify
"""

import re
import time
from threading import Lock

MAX_INPUT_LENGTH = 1000
RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60

# PII Detection regular expressions
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
# Matches 10-digit formats with separators, optional country code: e.g. +1-555-123-4567, 555.123.4567, 555-123-4567
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
# Matches 13-19 digit card patterns (standard credit/debit: Visa, Mastercard, Amex, etc.) with optional spaces or hyphens
CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{1,7}\b|\b\d{13,19}\b')


def validate_input_length(text: str, max_length: int = MAX_INPUT_LENGTH) -> tuple[bool, str]:
    """Validate that user input does not exceed max_length.
    
    Returns:
        (True, sanitized_text) if valid.
        (False, error_message) if text exceeds max_length.
    """
    if text is None:
        return True, ""
    
    if len(text) > max_length:
        return False, (
            f"Input exceeds maximum allowed limit of {max_length} characters "
            f"(received {len(text)} characters). Please shorten your request."
        )
    return True, text


class SessionRateLimiter:
    """Thread-safe sliding-window rate limiter per session ID."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: float = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = {}
        self._lock = Lock()
    
    def check_rate_limit(self, session_id: str = "default") -> tuple[bool, str]:
        """Check if request is allowed for session_id.
        
        Returns:
            (True, "") if within rate limit.
            (False, error_message) if rate limit exceeded.
        """
        now = time.time()
        with self._lock:
            timestamps = self._history.get(session_id, [])
            # Evict timestamps older than sliding window
            cutoff = now - self.window_seconds
            timestamps = [t for t in timestamps if t > cutoff]
            
            if len(timestamps) >= self.max_requests:
                earliest = timestamps[0]
                retry_after = int(self.window_seconds - (now - earliest)) + 1
                return False, (
                    f"Rate limit exceeded: maximum {self.max_requests} requests per minute. "
                    f"Please wait {retry_after} second(s) before sending another message."
                )
            
            timestamps.append(now)
            self._history[session_id] = timestamps
            return True, ""
    
    def reset(self, session_id: str | None = None):
        """Reset rate limit history for a session or globally."""
        with self._lock:
            if session_id:
                self._history.pop(session_id, None)
            else:
                self._history.clear()


# Default global rate limiter instance
global_rate_limiter = SessionRateLimiter()


def redact_pii(text: str) -> tuple[str, list[dict]]:
    """Redact sensitive personally identifiable information (PII) before text reaches LLM & logs.
    
    Masks:
    - Email addresses -> [EMAIL REDACTED]
    - Phone numbers -> [PHONE REDACTED]
    - Credit/Debit cards -> [CARD REDACTED]
    
    Returns:
        (redacted_text, detected_pii_list)
    """
    if not text:
        return text, []
    
    detected = []
    
    # 1. Redact credit card numbers first (so numbers aren't caught by phone pattern)
    def card_sub(m):
        raw = m.group(0)
        digits_only = re.sub(r'\D', '', raw)
        if 13 <= len(digits_only) <= 19:
            detected.append({"type": "credit_card", "matched": raw})
            return "[CARD REDACTED]"
        return raw

    redacted = CARD_REGEX.sub(card_sub, text)
    
    # 2. Redact email addresses
    def email_sub(m):
        raw = m.group(0)
        detected.append({"type": "email", "matched": raw})
        return "[EMAIL REDACTED]"

    redacted = EMAIL_REGEX.sub(email_sub, redacted)
    
    # 3. Redact phone numbers
    def phone_sub(m):
        raw = m.group(0)
        digits_only = re.sub(r'\D', '', raw)
        if 10 <= len(digits_only) <= 15:
            detected.append({"type": "phone", "matched": raw})
            return "[PHONE REDACTED]"
        return raw

    redacted = PHONE_REGEX.sub(phone_sub, redacted)
    
    return redacted, detected


# Tools that perform stateful database modifications
TRANSACTIONAL_WRITE_TOOLS = {"place_order", "cancel_order", "modify_order"}

CONFIRMATION_KEYWORDS = {
    "confirm", "confirmed", "yes", "proceed", "go ahead", "approve", "approved",
    "please cancel", "please modify", "please place", "i confirm", "sure", "ok", "okay"
}


def check_transactional_confirmation(
    action: str,
    action_input: dict,
    user_message: str = "",
    history: list[dict] | None = None,
) -> tuple[bool, str]:
    """Verify that user has explicitly confirmed a transactional write before executing.
    
    Args:
        action: The tool action name.
        action_input: Arguments to the tool.
        user_message: Current user message text.
        history: Previous chat turns.
        
    Returns:
        (True, "") if allowed (action is read-only or explicit confirmation is present).
        (False, challenge_message) if confirmation turn is required.
    """
    if action not in TRANSACTIONAL_WRITE_TOOLS:
        return True, ""
    
    msg_lower = (user_message or "").lower()
    
    # Check if user message contains explicit confirmation phrase
    has_confirm = any(re.search(rf'\b{re.escape(kw)}\b', msg_lower) for kw in CONFIRMATION_KEYWORDS)
    if has_confirm:
        return True, ""
    
    # Check if previous assistant turn requested confirmation
    if history:
        for turn in reversed(history):
            content = turn.get("content", "").lower()
            if "please confirm" in content or "would you like to proceed" in content:
                if any(w in msg_lower for w in ["yes", "proceed", "sure", "ok", "yep", "do it"]):
                    return True, ""
    
    # Challenge required
    if action == "place_order":
        item = action_input.get("product_id", "item")
        qty = action_input.get("qty", 1)
        name = action_input.get("customer_name", "Customer")
        return False, (
            f"⚠️ **Order Confirmation Required**: Are you sure you want to place an order for "
            f"**{qty}x {item}** under name **{name}**? "
            f"Please reply **'Yes, confirm'** to proceed with the transaction."
        )
    elif action == "cancel_order":
        order_id = action_input.get("order_id", "your order")
        return False, (
            f"⚠️ **Cancellation Confirmation Required**: Are you sure you want to permanently cancel "
            f"order **{order_id}**? Please reply **'Yes, confirm cancellation'** to finalize."
        )
    elif action == "modify_order":
        order_id = action_input.get("order_id", "your order")
        new_qty = action_input.get("new_qty", "")
        return False, (
            f"⚠️ **Modification Confirmation Required**: Are you sure you want to modify order "
            f"**{order_id}** to quantity **{new_qty}**? Please reply **'Yes, confirm'** to apply changes."
        )
    
    return False, f"⚠️ Confirmation required before executing {action}. Please confirm to proceed."
