# GiveLight Drive automation

This standalone Google Apps Script lets GiveLight move completed handoffs across
an access boundary without granting B2 access to GiveLight's private Drive
folder. The B2 service uploads a JSON file and its matching document to a shared
staging folder. A GiveLight-owned script checks that folder every minute, copies
complete pairs into a private destination folder, and emails a notification.

Use a dedicated GiveLight automation account to create and authorize the
script. That account must have access only to the staging and destination
folders needed by this workflow. It must be able to read and trash files in the
staging folder and create files in the private destination folder. Restrict
staging-folder writers to the B2 runtime service account and designated
GiveLight administrators. Notification emails are sent as the authorizing
GiveLight account. Apps Script email and runtime quotas apply, and a handoff may
take up to approximately one minute to process.

## Create the Apps Script project

1. Sign in to the dedicated GiveLight automation account that will own the
   automation.
2. Go to [script.google.com](https://script.google.com/) and select **New
   project**.
3. Give the project a recognizable name, such as `GiveLight Drive Handoff`.
4. Replace the contents of `Code.gs` with this directory's
   [`Code.gs`](./Code.gs).
5. Open **Project Settings**, enable **Show "appsscript.json" manifest file in
   editor**, and replace that file's contents with this directory's
   [`appsscript.json`](./appsscript.json).

The manifest enables the Advanced Drive service v3 used for both My Drive and
Shared Drive folders. If the script is linked to a standard Google Cloud
project instead of its default Apps Script Cloud project, also enable the
Google Drive API in that standard project.

## Configure Script Properties

In **Project Settings**, under **Script Properties**, add:

- `STAGING_FOLDER_ID`: the ID of the folder shared with the B2 service account.
- `DESTINATION_FOLDER_ID`: the ID of GiveLight's private destination folder.
- `NOTIFICATION_EMAILS`: one or more comma-separated notification recipients.
- `NOTIFICATION_SUBJECT` (optional): the email subject. It defaults to
  `GiveLight Drive handoff completed`.

A folder ID is the value after `/folders/` in a Google Drive folder URL. Do not
put credentials or document data in Script Properties.

## Authorize and install the trigger

1. In the Apps Script editor, select `installTrigger` from the function menu.
2. Click **Run**.
3. Review and approve the requested Drive, external-request, email, and trigger
   permissions for the GiveLight account. The external request is used only to
   download the verification JSON from the Google Drive API for validation.
4. Open **Triggers** in the left sidebar and confirm that `processHandoffs` has
   a time-driven trigger that runs every minute.

Running `installTrigger` again replaces any existing `processHandoffs` triggers
in this project, so it is safe to use when repairing the schedule.

## Test the automation

1. Put two test files in the staging folder with the same filename stem, for
   example `death-certificate-1234.json` and `death-certificate-1234.pdf`. The
   JSON must be valid UTF-8 JSON with nonempty `schema_version` and
   `submitted_at` string fields. Supported document extensions are `.jpg`,
   `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.pdf`, and `.bin`, matched
   case-insensitively.
2. Select `processHandoffs` in the Apps Script editor and click **Run**, or wait
   for the scheduled trigger.
3. Confirm that both files appear in the destination folder, the configured
   recipients receive an email, and the staging copies move to trash.

Incomplete pairs remain in staging for a later run. A candidate must have a
`death-certificate-` filename stem, exactly one JSON file and one supported
document, JSON no larger than 1 MiB, and a document no larger than 20 MiB. The
script parses the JSON only to check the two required fields; it does not log or
email its contents. These checks reduce accidental or malformed input, but they
do not cryptographically prove that B2 created the files. This version does not
use a signed manifest or HMAC, so staging write access must remain restricted.

The script processes at most 10 handoffs per invocation. Failures are isolated
per handoff so one bad pair does not prevent later pairs from running. Script
Properties record the source IDs, destination copy IDs, notification status,
and cleanup status after each completed step. A later run resumes pending work
before discovering new pairs, avoids repeating steps already recorded as
complete, and can finish cleanup even after source files leave the staging
folder. When pending work exists, each run attempts at most five pending
handoffs so at least half of the batch remains available for newly discovered
work. Failed pending handoffs record only a retry timestamp and are rotated
oldest-first on later runs, preventing one persistent failure from monopolizing
the retry capacity. Complete pairs enter this isolated state before validation,
so malformed pairs also rotate instead of repeatedly occupying new-work slots.
A permanently invalid pair remains pending until an administrator fixes its
source files or removes its `HANDOFF_STATE_...` Script Property and the invalid
staging files.

The operation order is copy both files, send the email, then trash both staging
files. The state is removed only after both source files are trashed. The
automation account therefore requires permission to trash staging files. As
with any external call followed by a state write, there is a small crash window
after `MailApp` accepts an email but before `notificationSent` is saved; an
interruption in that window can produce a duplicate notification. Similar
interruptions immediately after a Drive copy can produce a duplicate copy, so
the workflow does not claim perfect exactly-once processing.

The notification contains only the processing timestamp, a statement that the
verification JSON and source document were copied, and the destination folder
link. It does not include filenames, JSON contents, extracted fields, or the
source document as an attachment. B2 generates opaque UUID-based filenames so
folder metadata does not expose contact-derived identifiers or submission
timestamps.

## Disable or remove the automation

Open **Triggers** in the Apps Script project's left sidebar, locate the
`processHandoffs` trigger, open its actions menu, and select **Delete trigger**.
Deleting the Apps Script project also removes its triggers. Removing the trigger
does not change files that have already been copied or trashed.
