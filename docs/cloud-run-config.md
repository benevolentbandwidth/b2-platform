# Cloud Run config reference

What each service needs configured, and why. This is a reference for humans;
`deploy/cloud_run_config.json` is the machine-readable version that
`scripts/verify_cloud_run_config.py` checks on every deploy (see
`cloudbuild.yaml`). Keep the two in sync when either changes.

Last verified directly against both live services: 2026-09-24.

## Both services, project `b2-platform`, region `us-east1`

| Name | Kind | Required | Why |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | plain env, value `benevolent-bandwidth` | yes | Drives both Vertex AI and Firestore. Without it the router fails on the first inbound message. |
| `VERTEX_LOCATION` | plain env, value `us-east1` | yes | Code defaults to `us-central1` if unset — silently wrong for this org. |
| `WEBHOOK_SECRET` | Secret Manager reference, secret `webhook-secret`, version `latest` | yes | `/message` returns 503 if this is unset at all, and 401 on every call if it's wrong. Never set as a plain value — the check fails the build if it finds one. |
| `GOOGLE_DRIVE_FOLDER_ID` | plain env | not yet set on either service | Without it, the GiveLight handoff silently no-ops on an accepted case — no error, just nothing delivered. Flagged as a warning, not a build failure, until someone decides on prod vs. a separate staging folder. |

`WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` are deliberately absent from
both services as plain env vars. `src/gcp_secrets.py` falls back to the
`WHATSAPP_ACCESS_TOKEN` / `WHATSAPP_PHONE_NUMBER_ID` secrets in
`benevolent-bandwidth`'s Secret Manager, readable by both services' shared
runtime identity, `b2-platform-app-service@b2-platform.iam.gserviceaccount.com`.
Confirmed working end to end on 2026-09-24 against `b2-whatsapp-adapter`. If
either env var is ever set directly, it wins over the Secret Manager value.

## Where the two services differ, and whether that's decided

| | `b2-whatsapp-adapter` | `b2-whatsapp-adapter-staging` |
|---|---|---|
| Built from | `AnkitGole007/b2-platform` main | `benevolentbandwidth/b2-platform` main |
| Max instances | 1 | 100 |
| `webhook-secret` value | same secret, same value | same secret, same value — **not yet a deliberate decision.** Anyone with staging's secret can authenticate to prod's webhook and vice versa. Worth a decision, not an accident. |

## Running the check by hand

```
python scripts/verify_cloud_run_config.py --service b2-whatsapp-adapter --region us-east1 --project b2-platform
```

Needs `gcloud auth login` (or ADC) with read access to the service. Exits 1
and prints exactly what's missing if a required var or secret binding is
absent, or if the secret is bound as a plain value instead of a Secret
Manager reference. Warnings print but don't fail the run.

## Wiring it into a deploy

`cloudbuild.yaml` in the repo root runs this as the first step of the build,
before the image is even built — a service missing required config fails
fast without spending build minutes. It is not yet the active config on any
of the three Cloud Build triggers; see the comment at the top of that file
for how to point a trigger at it.
