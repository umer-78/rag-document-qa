# Customer Support Runbook

## Response targets

First response within 4 working hours for standard plans and 1 hour for
enterprise. Resolution target is 2 working days for standard issues. If a ticket
will miss its target, tell the customer before it does, not afterwards.

## Triage

Every ticket gets a severity within 30 minutes: blocking, degraded or question.
Blocking tickets are escalated to the on-call engineer straight away through the
on-call tool, not by direct message.

## Common issues

### Login loops

Usually a stale session cookie. Ask the customer to sign out on all devices from
account settings, then sign in again in a private window. If it persists, check
the identity provider status page before escalating.

### Missing data in reports

Reports are built from the previous night's snapshot, so data entered today
appears tomorrow. Live figures are on the dashboard, not in reports. If
yesterday's data is missing, check the nightly job in the admin panel.

### Export failures

Exports over 100,000 rows time out on the standard plan. Suggest filtering by
date range, or the API for bulk pulls.

## Refunds

Support can approve refunds up to USD 200 without escalation. Above that, a team
lead approves. Refunds are processed to the original payment method within 5
working days.

## Escalation path

Support → team lead → on-call engineer → head of engineering. Escalate by
severity, not by how loud the customer is, and record the reason in the ticket.

## Tone

Plain language, no blame, no jargon. Say what happened, what you are doing, and
when the customer will hear from you next. Never speculate about causes in
writing before the investigation is done.
