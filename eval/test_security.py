"""
Unit tests for security hardening module (security.py).
Tests:
1. Input length validation & capping
2. Per-session sliding window rate limiting
3. PII redaction (email, phone, credit card)
4. Transactional confirmation guard before place/cancel/modify
"""

import pytest
import time
from security import (
    validate_input_length,
    SessionRateLimiter,
    redact_pii,
    check_transactional_confirmation,
    MAX_INPUT_LENGTH,
)


class TestInputLengthCap:
    def test_valid_length(self):
        valid_text = "What is the status of my order ORD1001?"
        is_valid, msg = validate_input_length(valid_text)
        assert is_valid is True
        assert msg == valid_text

    def test_exceeded_length(self):
        long_text = "a" * (MAX_INPUT_LENGTH + 50)
        is_valid, msg = validate_input_length(long_text)
        assert is_valid is False
        assert "exceeds maximum allowed limit" in msg
        assert str(MAX_INPUT_LENGTH) in msg


class TestRateLimiter:
    def test_within_rate_limit(self):
        limiter = SessionRateLimiter(max_requests=5, window_seconds=10)
        for _ in range(5):
            ok, msg = limiter.check_rate_limit("user-1")
            assert ok is True
            assert msg == ""

    def test_exceed_rate_limit(self):
        limiter = SessionRateLimiter(max_requests=3, window_seconds=10)
        for _ in range(3):
            ok, _ = limiter.check_rate_limit("user-1")
            assert ok is True
        # 4th request must be rejected
        ok, msg = limiter.check_rate_limit("user-1")
        assert ok is False
        assert "Rate limit exceeded" in msg

    def test_independent_sessions(self):
        limiter = SessionRateLimiter(max_requests=2, window_seconds=10)
        limiter.check_rate_limit("session-a")
        limiter.check_rate_limit("session-a")
        # session-a is capped
        ok_a, _ = limiter.check_rate_limit("session-a")
        assert ok_a is False
        # session-b should still be allowed
        ok_b, _ = limiter.check_rate_limit("session-b")
        assert ok_b is True


class TestPIIRedaction:
    def test_email_redaction(self):
        text = "My email is customer.support@example.com, please update my account."
        redacted, detected = redact_pii(text)
        assert "[EMAIL REDACTED]" in redacted
        assert "customer.support@example.com" not in redacted
        assert len(detected) == 1
        assert detected[0]["type"] == "email"

    def test_phone_redaction(self):
        text = "Call me back at 555-123-4567 or (555) 987-6543 today."
        redacted, detected = redact_pii(text)
        assert "[PHONE REDACTED]" in redacted
        assert "555-123-4567" not in redacted
        assert "987-6543" not in redacted
        assert any(d["type"] == "phone" for d in detected)

    def test_card_redaction(self):
        text = "Charge my card 4111 2222 3333 4444 for order ORD1001."
        redacted, detected = redact_pii(text)
        assert "[CARD REDACTED]" in redacted
        assert "4111 2222 3333 4444" not in redacted
        assert any(d["type"] == "credit_card" for d in detected)

    def test_no_pii_unchanged(self):
        text = "Can I return the Denim Jacket from order ORD1008?"
        redacted, detected = redact_pii(text)
        assert redacted == text
        assert len(detected) == 0


class TestTransactionalConfirmation:
    def test_readonly_tool_no_confirmation_needed(self):
        ok, msg = check_transactional_confirmation(
            action="get_order_status",
            action_input={"order_id": "ORD1001", "customer_name": "Alice Johnson"},
            user_message="Where is ORD1001?",
        )
        assert ok is True
        assert msg == ""

    def test_cancel_order_requires_confirmation_when_not_confirmed(self):
        ok, msg = check_transactional_confirmation(
            action="cancel_order",
            action_input={"order_id": "ORD1003", "customer_name": "Carlos Gomez"},
            user_message="I might want to cancel ORD1003",
        )
        assert ok is False
        assert "Cancellation Confirmation Required" in msg

    def test_cancel_order_allowed_when_explicitly_confirmed(self):
        ok, msg = check_transactional_confirmation(
            action="cancel_order",
            action_input={"order_id": "ORD1003", "customer_name": "Carlos Gomez"},
            user_message="Yes, please cancel ORD1003, I confirm.",
        )
        assert ok is True
        assert msg == ""

    def test_place_order_requires_confirmation_without_keyword(self):
        ok, msg = check_transactional_confirmation(
            action="place_order",
            action_input={"product_id": "PROD003", "qty": 1, "customer_name": "Alex"},
            user_message="I want 1 headphones",
        )
        assert ok is False
        assert "Order Confirmation Required" in msg

    def test_place_order_allowed_with_confirm_keyword(self):
        ok, msg = check_transactional_confirmation(
            action="place_order",
            action_input={"product_id": "PROD003", "qty": 1, "customer_name": "Alex"},
            user_message="Please confirm and place my order for PROD003.",
        )
        assert ok is True
        assert msg == ""
