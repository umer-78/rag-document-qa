# Engineering Onboarding

## Your first day

You will be given access to the code host, the cloud console, the ticket tracker
and the on-call tool. If something is missing, ask in #it-help rather than
waiting. Your buddy runs a 30-minute walkthrough of the architecture.

## Development environment

The platform runs on Python 3.11 and Node 20. Clone the monorepo, run
`make bootstrap`, then `make test`. The whole suite should finish in under four
minutes on a laptop; if it does not, say so, because slow tests get skipped and
skipped tests stop working.

## Branching and review

Trunk-based: short-lived branches off `main`, merged with a pull request. Every
pull request needs one approval, a green build, and a description that says what
changed and how it was checked. Pull requests over 400 changed lines should be
split.

## Testing expectations

New code ships with tests. Bug fixes ship with a test that fails before the fix.
The build blocks on unit tests, linting and a type check. Flaky tests are quarantined
within a day and fixed within a week, not muted and forgotten.

## Deployment

`main` deploys to staging automatically. Production deploys are manual, from a
green staging build, and are announced in #deploys. Any engineer may deploy; the
person who deploys watches the dashboards for 15 minutes afterwards.

## On-call

The on-call rotation is one week, handed over on Wednesday at 14:00. On-call
covers paging alerts only; routine work pauses. Compensation is one extra day
off per week of on-call. Being paged more than twice in a night is a bug in the
alerting, and is reviewed at the weekly operations meeting.

## Incidents

Severity 1 means customers cannot use the product; severity 2 is serious
degradation. Declare an incident early, in #incidents, with a channel topic that
names the incident commander. Every severity 1 gets a written review within five
working days, focused on causes rather than people.
