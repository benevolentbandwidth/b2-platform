# GiveLight handoff

Every case where the claimant sent a document is uploaded to GiveLight as two
files: the original document and a JSON payload. There is no review queue on
the B2 side — GiveLight's reviewers are the human review — so cases that were
not automatically accepted are uploaded too, marked as needing review.

B2 uploads each case to a Cloud Storage staging bucket. GiveLight's own Apps
Script ([`integrations/givelight-drive-automation`](../../integrations/givelight-drive-automation/README.md))
moves each complete case into GiveLight's private Drive folder, emails
GiveLight, and deletes the case from the bucket. B2 never has access to
GiveLight's Drive, and GiveLight's authorization never leaves its script.

B2 does not upload to Drive itself because Google gives service accounts no
Drive storage: a service account cannot create files in a My Drive folder, only
in a Shared Drive. Files the script creates belong to GiveLight's account.

## Reading a case

- **Two kinds of case**, by `decision` in the JSON and in the file name:
  - `death-certificate-accepted-<reference>` — cleared automatic acceptance.
  - `death-certificate-needs-review-<reference>` — needs a person to look at it.
    `review_reasons` says why in plain terms (for example, the claimant's
    account and the certificate disagree, or a check could not run), and
    `authenticity` lists which fraud checks ran, flagged or errored.
- **`case_reference`** (e.g. `DC-7F3A-9C21-B4E8`) is also given to the claimant,
  so a case can be found by searching Drive for the number they quote.
- **A JSON file means a complete case.** The document is uploaded first and the
  JSON last, so a JSON only appears once its document is already in the bucket.
  The JSON names its document in `image_file`; a document with no matching
  JSON is an interrupted upload (logged as `handoff.incomplete`) and can be
  ignored. The claimant is asked to resend when an upload fails.
- **The document is exactly what the claimant sent**, byte for byte. (TIFFs are
  converted to PNG only for Gemini's analysis, never for this upload.)
- `contact_phone` is the claimant's WhatsApp number, international format
  without a leading `+`, so GiveLight can contact the family.
- `config_versions` records the `scoring` and `detector` settings versions that
  produced the verdict.
- Rarely, a network retry can upload the same file twice. Copies share a name
  and belong to the same case.

## Google Cloud setup

1. Create the staging bucket (one per service, e.g. one for staging and one for
   production) with public access prevention, uniform bucket-level access, soft
   delete off, and a lifecycle rule deleting objects older than 7 days. See the
   automation README for why each matters.
2. Grant the Cloud Run runtime service account
   `b2-platform-app-service@b2-platform.iam.gserviceaccount.com`
   **Storage Object Creator** on the bucket, and nothing else. B2 uploads
   create-only (`ifGenerationMatch=0`), so it can add a case but never read,
   replace or delete one. It authenticates through Application Default
   Credentials; do not create or send a service-account JSON key.
3. Grant GiveLight's automation account **Storage Object Admin** on the bucket.
4. Set the bucket name on the service as `GIVELIGHT_HANDOFF_BUCKET`.
5. Have GiveLight set up the Apps Script, then verify that one case lands in
   the bucket and then in GiveLight's folder.

The uploader requests only the
`https://www.googleapis.com/auth/devstorage.read_write` OAuth scope; the
bucket's IAM limits it to creating objects. Store configuration in the
deployment platform; never commit credentials or secret values.

## WhatsApp configuration

The current repository requires:

- `WHATSAPP_TOKEN`: a production system-user access token with
  `whatsapp_business_messaging`, used to retrieve incoming media.
- `WHATSAPP_GRAPH_VERSION`: an optional, non-secret Graph API version override.
- `WEBHOOK_SECRET`: the internal shared secret between the central B2 webhook
  and this service.

If this service later integrates directly with Meta, it will additionally need
the Meta App ID and App Secret, WhatsApp Business Account (WABA) ID, Phone
Number ID, and a separately generated webhook verification token. Grant
`whatsapp_business_management` when account or template administration is
needed. Share secret values only through the approved secret-management
channel; do not send them in chat or commit them to the repository.
