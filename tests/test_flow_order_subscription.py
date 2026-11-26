import json
import os
import time
from hmac import digest
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app


class StripeSubscriptionAccessTests(TestCase):
    def setUp(self):
        self.client = TestClient(app.app)
        self.addCleanup(self.client.close)
        self.signing_secret = "whsec_local_mock"
        self.enterContext(patch.dict(os.environ, {
            "STRIPE_SUBSCRIPTION_WEBHOOK_SECRET": self.signing_secret,
            "FLOW_WEBHOOK_SECRET": "flow-mock",
            "FLOW_SUBSCRIPTIONS_API_TOKEN": "subscriptions-mock",
            "FLOW_LOCATION_ID": "test-location",
            "SUPABASE_URL": "https://supabase.example.test",
            "SUPABASE_SERVICE_ROLE_KEY": "sb_secret_mock",
        }))
        self.rpc = self.enterContext(patch.object(app.requests, "post"))
        self.rpc.return_value = Mock(status_code=200)
        self.rpc.return_value.json.return_value = True
        self.subscription = {
            "object": "subscription", "id": "sub_Verified1", "status": "canceled",
            "livemode": True, "ended_at": int(time.time()) - 1,
        }
        self.event = {
            "object": "event", "type": "customer.subscription.deleted", "livemode": True,
            "data": {"object": self.subscription},
        }

    def stripe_delivery(self, event, *, timestamp=None, secret=None):
        timestamp = str(int(time.time()) if timestamp is None else timestamp)
        body = json.dumps(event, separators=(",", ":")).encode()
        signature = digest(
            (self.signing_secret if secret is None else secret).encode(),
            timestamp.encode() + b"." + body,
            "sha256",
        ).hex()
        return self.client.post("/stripe/subscription-ended", content=body, headers={
            "Content-Type": "application/json", "Stripe-Signature": f"t={timestamp},v1={signature}",
        })

    def test_effective_end_records_only_subscription_id_and_timestamp(self):
        result = self.stripe_delivery(self.event)
        self.assertEqual(result.status_code, 200)
        self.rpc.assert_called_once()
        request = self.rpc.call_args
        self.assertEqual(request.args[0], "https://supabase.example.test/rest/v1/rpc/record_tutor_subscription_access")
        self.assertEqual(request.kwargs["headers"], {"apikey": "sb_secret_mock"})
        self.assertEqual(request.kwargs["json"]["p_subscription_id"], "sub_Verified1")
        self.assertEqual(set(request.kwargs["json"]), {"p_subscription_id", "p_ended_at"})

    def test_scheduled_cancellation_and_test_mode_leave_access_alone(self):
        scheduled = {**self.event, "type": "customer.subscription.updated", "data": {"object": {
            **self.subscription, "cancel_at_period_end": True, "status": "active", "ended_at": None,
        }}}
        for event in (scheduled, {**self.event, "livemode": False}):
            with self.subTest(event=event):
                self.assertEqual(self.stripe_delivery(event).json(), {"ok": True, "ignored": True})
        self.rpc.assert_not_called()

    def test_invalid_or_stale_signatures_and_unconfirmed_end_do_not_write(self):
        self.assertEqual(self.stripe_delivery(self.event, secret="wrong").status_code, 400)
        self.assertEqual(self.stripe_delivery(self.event, timestamp=int(time.time()) - 360).status_code, 400)
        for changes in ({"ended_at": None}, {"ended_at": int(time.time()) + 3600}, {"status": "active"}):
            event = {**self.event, "data": {"object": {**self.subscription, **changes}}}
            self.assertEqual(self.stripe_delivery(event).status_code, 400)
        self.rpc.assert_not_called()

    def test_supabase_failure_returns_retryable_status(self):
        self.rpc.return_value.status_code = 500
        self.assertEqual(self.stripe_delivery(self.event).status_code, 503)

    def test_paid_webhook_cannot_bypass_ended_subscription_with_email_upsert(self):
        with patch.object(app, "verify_flow_payment", return_value={
            "subscriptionId": "sub_Verified1", "contactId": "checkout-contact",
        }), patch.object(app, "supabase_authorized_users_request") as legacy_write:
            self.rpc.return_value.json.return_value = False
            result = self.client.post("/flow/order-paid", headers={"X-Textbook-Tutor-Secret": "flow-mock"}, json={
                "email": "buyer@example.com", "event_type": "order_submitted", "payment_transaction_id": "tx-1",
            })
        self.assertEqual(result.status_code, 403)
        self.assertEqual(self.rpc.call_args.kwargs["json"], {
            "p_subscription_id": "sub_Verified1", "p_email": "buyer@example.com", "p_source": "stripe",
            "p_contact_id": "checkout-contact",
        })
        legacy_write.assert_not_called()

    def test_promo_webhook_binds_the_verified_order_subscription(self):
        row = {
            "altId": "test-location", "altType": "location", "entityType": "order", "entityId": "order-1",
            "contactId": "checkout-contact", "liveMode": True, "paymentProviderType": "stripe",
            "status": "active", "subscriptionId": "sub_Verified1",
        }
        listing = Mock(status_code=200)
        listing.json.return_value = {"data": [row], "totalCount": 1}
        with patch.object(app, "verify_flow_promo", return_value="checkout-contact"), \
                patch.object(app.requests, "get", return_value=listing) as lookup:
            result = self.client.post("/flow/order-promo", headers={"X-Textbook-Tutor-Secret": "flow-mock"}, json={
                "email": "buyer@example.com", "event_type": "promo_redeemed", "order_id": "order-1",
            })
        self.assertEqual(result.status_code, 200)
        self.assertEqual(lookup.call_args.kwargs["params"]["entityId"], "order-1")
        self.assertEqual(lookup.call_args.kwargs["params"]["contactId"], "checkout-contact")
        self.assertEqual(self.rpc.call_args.kwargs["json"]["p_source"], "promo")
        self.assertEqual(self.rpc.call_args.kwargs["json"]["p_subscription_id"], "sub_Verified1")

    def test_promo_lookup_rejects_ambiguous_or_unrelated_subscriptions(self):
        with patch.object(app.requests, "get") as lookup:
            lookup.return_value.status_code = 200
            for page in ({"data": [{}, {}], "totalCount": 2}, {"data": [{"entityId": "other"}], "totalCount": 1}):
                lookup.return_value.json.return_value = page
                with self.assertRaises(app.HTTPException):
                    app.get_flow_promo_subscription_id("order-1", "checkout-contact")
        self.rpc.assert_not_called()
