const DOCUMENT_EXTENSIONS = [
  "jpg",
  "jpeg",
  "png",
  "webp",
  "heic",
  "heif",
  "pdf",
  "bin",
];
const HANDOFF_STEM_PREFIX = "death-certificate-";
const HANDOFF_STATE_PREFIX = "HANDOFF_STATE_";
const MAX_HANDOFFS_PER_RUN = 10;
const MAX_JSON_BYTES = 1024 * 1024;
const MAX_DOCUMENT_BYTES = 20 * 1024 * 1024;
const STORAGE_API = "https://storage.googleapis.com/storage/v1/b/";

/**
 * Saves complete handoff pairs from the staging bucket into GiveLight's private
 * folder, sends a notification, and then deletes the staging objects.
 */
function processHandoffs() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(0)) {
    return;
  }

  try {
    const properties = PropertiesService.getScriptProperties();
    const stagingBucket = getRequiredProperty_(properties, "STAGING_BUCKET");
    const destinationFolderId = getRequiredProperty_(
      properties,
      "DESTINATION_FOLDER_ID",
    );
    const notificationEmails = getRequiredProperty_(
      properties,
      "NOTIFICATION_EMAILS",
    );
    const notificationSubject =
      properties.getProperty("NOTIFICATION_SUBJECT") ||
      "GiveLight Drive handoff completed";
    assertFolderAccessible_(destinationFolderId);
    const config = {
      stagingBucket: stagingBucket,
      destinationFolderId: destinationFolderId,
      notificationEmails: notificationEmails,
      notificationSubject: notificationSubject,
      properties: properties,
    };

    const pendingStates = loadPendingStates_(properties);
    const knownJsonNames = {};
    pendingStates.forEach(function (state) {
      knownJsonNames[state.sourceJsonName] = true;
    });

    const pendingLimit = Math.floor(MAX_HANDOFFS_PER_RUN / 2);
    const pendingToAttempt = pendingStates.slice(0, pendingLimit);
    pendingToAttempt.forEach(function (state) {
      processStateSafely_(state, config);
    });

    const available = MAX_HANDOFFS_PER_RUN - pendingToAttempt.length;
    findCompleteHandoffs_(stagingBucket, knownJsonNames)
      .slice(0, available)
      .forEach(function (handoff) {
        const state = createState_(handoff);
        try {
          saveState_(properties, state);
        } catch (error) {
          console.error("A new handoff's retry state could not be saved.");
          return;
        }
        processStateSafely_(state, config);
      });
  } finally {
    lock.releaseLock();
  }
}

/**
 * Replaces this project's existing processHandoffs triggers with a trigger
 * that checks for completed handoffs every minute.
 */
function installTrigger() {
  ScriptApp.getProjectTriggers().forEach(function (trigger) {
    if (trigger.getHandlerFunction() === "processHandoffs") {
      ScriptApp.deleteTrigger(trigger);
    }
  });

  ScriptApp.newTrigger("processHandoffs").timeBased().everyMinutes(1).create();
}

function processStateSafely_(state, config) {
  try {
    processState_(state, config);
  } catch (error) {
    state.lastAttemptAt = new Date().toISOString();
    try {
      saveState_(config.properties, state);
    } catch (stateError) {
      console.error("A failed handoff's retry state could not be saved.");
    }
    console.error("A pending handoff could not be resumed; it will be retried.");
  }
}

function processState_(state, config) {
  const bucket = config.stagingBucket;
  if (!state.validated) {
    const sourceJson = getObjectMetadata_(
      bucket,
      state.sourceJsonName,
      state.sourceJsonGeneration,
    );
    const sourceDocument = getObjectMetadata_(
      bucket,
      state.sourceDocumentName,
      state.sourceDocumentGeneration,
    );
    validateState_(state, bucket, sourceJson, sourceDocument);
    state.validated = true;
    saveState_(config.properties, state);
  }

  // Files are created by this script's account, so they are owned by
  // GiveLight and use its storage.
  if (!state.destinationJsonFileId) {
    state.destinationJsonFileId = saveObjectToFolder_(
      bucket,
      state.sourceJsonName,
      state.sourceJsonGeneration,
      config.destinationFolderId,
    );
    saveState_(config.properties, state);
  }

  if (!state.destinationDocumentFileId) {
    state.destinationDocumentFileId = saveObjectToFolder_(
      bucket,
      state.sourceDocumentName,
      state.sourceDocumentGeneration,
      config.destinationFolderId,
    );
    saveState_(config.properties, state);
  }

  if (!state.notificationSent) {
    sendNotification_(config);
    state.notificationSent = true;
    saveState_(config.properties, state);
  }

  if (!state.sourceJsonDeleted) {
    deleteObject_(bucket, state.sourceJsonName, state.sourceJsonGeneration);
    state.sourceJsonDeleted = true;
    saveState_(config.properties, state);
  }

  if (!state.sourceDocumentDeleted) {
    deleteObject_(
      bucket,
      state.sourceDocumentName,
      state.sourceDocumentGeneration,
    );
    state.sourceDocumentDeleted = true;
    saveState_(config.properties, state);
  }

  config.properties.deleteProperty(stateKey_(state.sourceJsonName));
}

function sendNotification_(config) {
  const timestamp = Utilities.formatDate(
    new Date(),
    Session.getScriptTimeZone(),
    "yyyy-MM-dd HH:mm:ss z",
  );
  const destinationFolderUrl =
    "https://drive.google.com/drive/folders/" + config.destinationFolderId;

  MailApp.sendEmail({
    to: config.notificationEmails,
    subject: config.notificationSubject,
    body: [
      "A GiveLight Drive handoff completed.",
      "",
      "Timestamp: " + timestamp,
      "Files saved: 2 (verification JSON and source document)",
      "Destination folder: " + destinationFolderUrl,
    ].join("\n"),
  });
}

function createState_(handoff) {
  // An object is pinned by name and generation, so a later object reusing the
  // name is never mistaken for this one.
  return {
    stem: handoff.stem,
    sourceJsonName: handoff.jsonObject.name,
    sourceJsonGeneration: handoff.jsonObject.generation,
    sourceDocumentName: handoff.documentObject.name,
    sourceDocumentGeneration: handoff.documentObject.generation,
    validated: false,
    destinationJsonFileId: null,
    destinationDocumentFileId: null,
    notificationSent: false,
    sourceJsonDeleted: false,
    sourceDocumentDeleted: false,
    lastAttemptAt: null,
  };
}

function loadPendingStates_(properties) {
  const allProperties = properties.getProperties();
  const states = [];
  Object.keys(allProperties)
    .filter(function (key) {
      return key.indexOf(HANDOFF_STATE_PREFIX) === 0;
    })
    .sort()
    .forEach(function (key) {
      try {
        const state = JSON.parse(allProperties[key]);
        if (
          !state ||
          !state.stem ||
          !state.sourceJsonName ||
          !state.sourceJsonGeneration ||
          !state.sourceDocumentName ||
          !state.sourceDocumentGeneration ||
          typeof state.validated !== "boolean" ||
          key !== stateKey_(state.sourceJsonName)
        ) {
          throw new Error("Invalid handoff state.");
        }
        states.push(state);
      } catch (error) {
        console.error("A saved handoff state is invalid and was skipped.");
      }
    });
  return states.sort(function (left, right) {
    if (!left.lastAttemptAt && right.lastAttemptAt) {
      return -1;
    }
    if (left.lastAttemptAt && !right.lastAttemptAt) {
      return 1;
    }
    return String(left.lastAttemptAt || "").localeCompare(
      String(right.lastAttemptAt || ""),
    );
  });
}

function saveState_(properties, state) {
  properties.setProperty(
    stateKey_(state.sourceJsonName),
    JSON.stringify(state),
  );
}

function stateKey_(sourceJsonName) {
  return HANDOFF_STATE_PREFIX + sourceJsonName;
}

function validateState_(state, bucket, jsonObject, documentObject) {
  const jsonName = splitFileName_(jsonObject.name);
  const documentName = splitFileName_(documentObject.name);
  if (
    state.stem.indexOf(HANDOFF_STEM_PREFIX) !== 0 ||
    jsonName.stem !== state.stem ||
    documentName.stem !== state.stem ||
    jsonName.extension !== "json" ||
    DOCUMENT_EXTENSIONS.indexOf(documentName.extension) === -1
  ) {
    throw new Error("Unexpected handoff filename.");
  }
  if (Number(jsonObject.size) > MAX_JSON_BYTES) {
    throw new Error("Verification JSON is too large.");
  }
  if (Number(documentObject.size) > MAX_DOCUMENT_BYTES) {
    throw new Error("Source document is too large.");
  }

  const payload = JSON.parse(
    downloadObject_(bucket, jsonObject.name, jsonObject.generation)
      .getDataAsString("UTF-8"),
  );
  if (
    !payload ||
    typeof payload.schema_version !== "string" ||
    !payload.schema_version.trim() ||
    typeof payload.submitted_at !== "string" ||
    !payload.submitted_at.trim()
  ) {
    throw new Error("Verification JSON is missing required fields.");
  }
}

function splitFileName_(name) {
  const extensionMatch = name.match(/\.([^.]+)$/);
  if (!extensionMatch) {
    return { stem: name, extension: "" };
  }
  return {
    stem: name.slice(0, -extensionMatch[0].length),
    extension: extensionMatch[1].toLowerCase(),
  };
}

function findCompleteHandoffs_(bucket, excludedJsonNames) {
  const jsonObjectsByStem = {};
  const documentObjectsByStem = {};
  const objects = listObjects_(bucket);

  objects.forEach(function (object) {
    const name = object.name;
    const extensionMatch = name.match(/\.([^.]+)$/);

    if (!extensionMatch) {
      return;
    }

    const extension = extensionMatch[1].toLowerCase();
    const stem = name.slice(0, -extensionMatch[0].length);
    if (stem.indexOf(HANDOFF_STEM_PREFIX) !== 0) {
      return;
    }

    if (extension === "json") {
      addFileByStem_(jsonObjectsByStem, stem, object);
    } else if (DOCUMENT_EXTENSIONS.indexOf(extension) !== -1) {
      addFileByStem_(documentObjectsByStem, stem, object);
    }
  });

  return Object.keys(jsonObjectsByStem)
    .filter(function (stem) {
      return (
        jsonObjectsByStem[stem].length === 1 &&
        !excludedJsonNames[jsonObjectsByStem[stem][0].name] &&
        documentObjectsByStem[stem] &&
        documentObjectsByStem[stem].length === 1
      );
    })
    .sort()
    .map(function (stem) {
      return {
        stem: stem,
        jsonObject: jsonObjectsByStem[stem][0],
        documentObject: documentObjectsByStem[stem][0],
      };
    });
}

function listObjects_(bucket) {
  const objects = [];
  let pageToken;

  do {
    let url =
      STORAGE_API +
      encodeURIComponent(bucket) +
      "/o?fields=" +
      encodeURIComponent("nextPageToken,items(name,size,generation)");
    if (pageToken) {
      url += "&pageToken=" + encodeURIComponent(pageToken);
    }

    const response = JSON.parse(storageFetch_(url, "get").getContentText());
    Array.prototype.push.apply(objects, response.items || []);
    pageToken = response.nextPageToken;
  } while (pageToken);

  return objects;
}

function objectUrl_(bucket, name) {
  return (
    STORAGE_API + encodeURIComponent(bucket) + "/o/" + encodeURIComponent(name)
  );
}

function getObjectMetadata_(bucket, name, generation) {
  const url =
    objectUrl_(bucket, name) +
    "?generation=" +
    encodeURIComponent(generation) +
    "&fields=name,size,generation";
  return JSON.parse(storageFetch_(url, "get").getContentText());
}

function downloadObject_(bucket, name, generation) {
  const url =
    objectUrl_(bucket, name) +
    "?alt=media&generation=" +
    encodeURIComponent(generation);
  return storageFetch_(url, "get").getBlob().setName(name);
}

function saveObjectToFolder_(bucket, name, generation, folderId) {
  const blob = downloadObject_(bucket, name, generation);
  const file = Drive.Files.create(
    { name: name, parents: [folderId], mimeType: blob.getContentType() },
    blob,
    { supportsAllDrives: true, fields: "id" },
  );
  return file.id;
}

function deleteObject_(bucket, name, generation) {
  const url =
    objectUrl_(bucket, name) + "?generation=" + encodeURIComponent(generation);
  // Already gone (a previous run deleted it, or the bucket's age rule did).
  storageFetch_(url, "delete", [404]);
}

/**
 * Calls the Cloud Storage JSON API as this script's account. Throws on any
 * status other than 2xx or one listed in allowedStatuses.
 */
function storageFetch_(url, method, allowedStatuses) {
  const response = UrlFetchApp.fetch(url, {
    method: method,
    headers: { Authorization: "Bearer " + ScriptApp.getOAuthToken() },
    muteHttpExceptions: true,
  });
  const status = response.getResponseCode();
  if (
    (status < 200 || status >= 300) &&
    (allowedStatuses || []).indexOf(status) === -1
  ) {
    // The status alone: error bodies can echo object names.
    throw new Error("Cloud Storage request failed with status " + status + ".");
  }
  return response;
}

function assertFolderAccessible_(folderId) {
  const folder = Drive.Files.get(folderId, {
    supportsAllDrives: true,
    fields: "id,mimeType",
  });
  if (folder.mimeType !== "application/vnd.google-apps.folder") {
    throw new Error("Configured Drive ID is not a folder.");
  }
}

function addFileByStem_(filesByStem, stem, file) {
  if (!filesByStem[stem]) {
    filesByStem[stem] = [];
  }
  filesByStem[stem].push(file);
}

function getRequiredProperty_(properties, name) {
  const value = properties.getProperty(name);
  if (!value || !value.trim()) {
    throw new Error("Missing required Script Property: " + name);
  }
  return value.trim();
}
