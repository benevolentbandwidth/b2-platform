# GiveLight Drive automation

This standalone Google Apps Script moves completed handoffs into GiveLight's
private Drive folder without giving B2 any access to GiveLight's Drive. The B2
service uploads a JSON file and its matching document to a Cloud Storage
staging bucket. A GiveLight-owned script checks that bucket every minute, saves
complete pairs into a private destination folder, emails a notification, and
deletes the pair from the bucket.

The files in the destination folder are created by the script's account, so
they are owned by GiveLight and use its Drive storage. This is why B2 does not
upload to Drive directly: Google gives service accounts no Drive storage of
their own, so a service account cannot create files in a My Drive folder.

Use a dedicated GiveLight automation account to create and authorize the
script. That account needs access only to the staging bucket and the
destination folder. Notification emails are sent as the authorizing GiveLight
account. Apps Script email, URL fetch and runtime quotas apply, and a handoff
may take up to approximately one minute to process.

## Staging bucket

The B2 team creates the bucket and sets it on the B2 service as
`GIVELIGHT_HANDOFF_BUCKET`. It must:

- grant the B2 runtime service account **Storage Object Creator** only. B2
  uploads create-only, so it can add a case but never read, replace or delete
  one;
- grant the GiveLight automation account **Storage Object Admin**, to list,
  read and delete the staging objects;
- enforce public access prevention and uniform bucket-level access;
- have soft delete turned off, so deleted cases are not kept;
- delete objects older than 7 days, so nothing stays if the script stops, and a
  document whose JSON never arrived is cleaned up.

Grant no other access. The bucket is the only thing the GiveLight account can
reach in the B2 Cloud project.

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

The manifest enables the Advanced Drive service v3, used for both My Drive and
Shared Drive destination folders, and requests Cloud Storage access for the
staging bucket. If the first run fails with "Cloud Storage JSON API has not been
used in project …", link the script to a standard Google Cloud project with the
Cloud Storage JSON API and Google Drive API enabled (**Project Settings** →
**Google Cloud Platform (GCP) Project**).

## Configure Script Properties

In **Project Settings**, under **Script Properties**, add:

- `STAGING_BUCKET`: the name of the Cloud Storage staging bucket.
- `DESTINATION_FOLDER_ID`: the ID of GiveLight's private destination folder.
- `NOTIFICATION_EMAILS`: one or more comma-separated notification recipients.
- `NOTIFICATION_SUBJECT` (optional): the email subject. It defaults to
  `GiveLight Drive handoff completed`.

A folder ID is the value after `/folders/` in a Google Drive folder URL. Do not
put credentials or document data in Script Properties.

## Authorize and install the trigger

1. In the Apps Script editor, select `installTrigger` from the function menu.
2. Click **Run**.
3. Review and approve the requested Drive, Cloud Storage, external-request,
   email, and trigger permissions for the GiveLight account. External requests
   are used only to call the Cloud Storage API for the staging bucket.
4. Open **Triggers** in the left sidebar and confirm that `processHandoffs` has
   a time-driven trigger that runs every minute.

Running `installTrigger` again replaces any existing `processHandoffs` triggers
in this project, so it is safe to use when repairing the schedule.

The authorization belongs to the GiveLight account and stays with this script:
B2 never receives a token. Deleting the trigger or the project, or removing the
account's access in its Google Account security settings, stops it at once.

## Test the automation

1. Put two test objects in the staging bucket with the same name stem, for
   example `death-certificate-1234.json` and `death-certificate-1234.pdf`. The
   JSON must be valid UTF-8 JSON with nonempty `schema_version` and
   `submitted_at` string fields. Supported document extensions are `.jpg`,
   `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.pdf`, and `.bin`, matched
   case-insensitively.
2. Select `processHandoffs` in the Apps Script editor and click **Run**, or wait
   for the scheduled trigger.
3. Confirm that both files appear in the destination folder, the configured
   recipients receive an email, and both objects are gone from the bucket.

Incomplete pairs remain in the bucket for a later run until the bucket's age
rule deletes them. A candidate must have a `death-certificate-` name stem,
exactly one JSON object and one supported document, JSON no larger than 1 MiB,
and a document no larger than 20 MiB. The script parses the JSON only to check
the two required fields; it does not log or email its contents. These checks
reduce accidental or malformed input, but they do not cryptographically prove
that B2 created the objects. This version does not use a signed manifest or
HMAC, so bucket write access must remain restricted to the B2 service account.

The script processes at most 10 handoffs per invocation. Failures are isolated
per handoff so one bad pair does not prevent later pairs from running. Each
staging object is pinned by name and generation, and Script Properties record
the destination file IDs, notification status, and cleanup status after each
completed step. A later run resumes pending work before discovering new pairs
and avoids repeating steps already recorded as complete. When pending work
exists, each run attempts at most five pending handoffs so at least half of the
batch remains available for newly discovered work. Failed pending handoffs
record only a retry timestamp and are rotated oldest-first on later runs,
preventing one persistent failure from monopolizing the retry capacity.
Complete pairs enter this isolated state before validation, so malformed pairs
also rotate instead of repeatedly occupying new-work slots. A permanently
invalid pair remains pending until an administrator fixes or removes its
staging objects and removes its `HANDOFF_STATE_...` Script Property.

The operation order is save both files to Drive, send the email, then delete
both staging objects. The state is removed only after both objects are deleted.
As with any external call followed by a state write, there is a small crash
window after `MailApp` accepts an email but before `notificationSent` is saved;
an interruption in that window can produce a duplicate notification. Similar
interruptions immediately after a Drive create can produce a duplicate file, so
the workflow does not claim perfect exactly-once processing.

The notification contains only the processing timestamp, a statement that the
verification JSON and source document were saved, and the destination folder
link. It does not include filenames, JSON contents, extracted fields, or the
source document as an attachment.

## Disable or remove the automation

Open **Triggers** in the Apps Script project's left sidebar, locate the
`processHandoffs` trigger, open its actions menu, and select **Delete trigger**.
Deleting the Apps Script project also removes its triggers. Removing the trigger
does not change files that have already been saved or deleted; cases still in
the bucket are deleted by its age rule.
