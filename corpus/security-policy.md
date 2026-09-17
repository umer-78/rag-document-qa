# Information Security Policy

## Passwords and MFA

Passwords must be at least 14 characters and unique to each service. The company
password manager (1Password) is mandatory for all work accounts. Multi-factor
authentication is required on email, the cloud console, the code host and the
VPN; SMS codes are not accepted as a second factor.

## Device security

Full-disk encryption must be on. Screen lock after 5 minutes. Laptops receive
security updates within 7 days of release, and the operating system must be on a
version that still receives updates.

## Access control

Access follows least privilege and is reviewed every quarter. Production
database access is limited to the on-call engineer and the platform team, and
every session is logged. Access is removed on the last working day, not later.

## Secrets

Secrets never go into the repository, a ticket, or chat. They live in the cloud
secret manager. A secret that has been exposed is rotated within 24 hours and an
incident is opened, even if it was exposed only internally.

## Reporting an incident

Report anything suspicious to security@northwind.example or in #security. There
is no penalty for a false alarm, and no penalty for reporting a mistake you made
yourself. The security team acknowledges within 1 hour during working hours and
within 4 hours otherwise.

## Phishing

Suspicious email is forwarded as an attachment to phishing@northwind.example.
The team runs simulated phishing twice a year; results are only ever reported as
an aggregate, never per person.

## Data classification

Public, Internal, Confidential and Restricted. Customer data is Confidential at
minimum. Restricted data (payment details, government IDs) must not be copied to
a laptop and may only be processed inside the production environment.

## Vulnerability handling

Critical vulnerabilities are patched within 7 days, high within 14, medium
within 30. A missed deadline needs a written exception approved by the head of
engineering.
