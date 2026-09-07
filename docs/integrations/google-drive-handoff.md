# Google Drive handoff

Accepted death-certificate cases are uploaded to GiveLight as two files with a
shared filename stem: the JSON verification payload and the original image (or
PDF). The stem is an opaque `death-certificate-<uuid>` identifier and does not
contain a contact-derived value or submission timestamp. The destination should
be a folder in a Google Shared Drive.

## Google Cloud and GiveLight setup

1. Enable the Google Drive API in the Google Cloud project that runs the B2
   WhatsApp adapter.
2. Use the Cloud Run runtime service account through Application Default
   Credentials (ADC). Do not create or send a service-account JSON key.
3. Ask GiveLight to share the destination Shared Drive folder with the runtime
   service-account email and grant it permission to add files.
4. Copy the folder ID from its Drive URL and set it on the service as
   `GOOGLE_DRIVE_FOLDER_ID`.
5. Deploy a revision and verify that one accepted case creates both the JSON
   payload and original document in the destination folder.

The uploader requests only the
`https://www.googleapis.com/auth/drive.file` OAuth scope. Store configuration in
the deployment platform; never commit credentials or secret values.

## Optional GiveLight private-folder automation

If B2 should upload only to a shared staging folder, GiveLight can own a
standalone Apps Script that copies complete JSON/document pairs into a private
GiveLight folder and then sends a notification email. This keeps the private
folder and email authority under GiveLight's Google account; the B2 service
account needs access only to the staging folder.

See the
[`integrations/givelight-drive-automation`](../../integrations/givelight-drive-automation/README.md)
setup guide and reference implementation. The automation polls once per minute,
validates and processes batches of up to 10 pairs, copies both files before
sending email, and trashes the staging copies only after notification succeeds.
Resumable Script Properties prevent ordinary retries from repeating completed
steps, though an interruption between an external Drive or email operation and
its state write can still cause a duplicate.

Run the script as a dedicated GiveLight automation account with access only to
the staging and destination folders. Restrict staging writers to the B2 runtime
service account and designated GiveLight administrators, and grant the
automation account permission to trash staging files. Filename, size, and JSON
schema checks do not provide cryptographic provenance; the reference workflow
does not implement a signed manifest or HMAC.

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
