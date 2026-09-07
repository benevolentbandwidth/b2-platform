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

/**
 * Copies complete handoff pairs into GiveLight's private folder, sends a
 * notification, and then removes the staging copies.
 */
function processHandoffs() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(0)) {
    return;
  }

  try {
    const properties = PropertiesService.getScriptProperties();
    const stagingFolderId = getRequiredProperty_(properties, "STAGING_FOLDER_ID");
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
    assertFolderAccessible_(stagingFolderId);
    assertFolderAccessible_(destinationFolderId);
    const config = {
      destinationFolderId: destinationFolderId,
      notificationEmails: notificationEmails,
      notificationSubject: notificationSubject,
      properties: properties,
    };

    const pendingStates = loadPendingStates_(properties);
    const knownJsonIds = {};
    pendingStates.forEach(function (state) {
      knownJsonIds[state.sourceJsonId] = true;
    });

    const pendingLimit = Math.floor(MAX_HANDOFFS_PER_RUN / 2);
    const pendingToAttempt = pendingStates.slice(0, pendingLimit);
    pendingToAttempt.forEach(function (state) {
      processStateSafely_(state, config);
    });

    const available = MAX_HANDOFFS_PER_RUN - pendingToAttempt.length;
    findCompleteHandoffs_(stagingFolderId, knownJsonIds)
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
  if (!state.validated) {
    const sourceJson = getFileMetadata_(state.sourceJsonId);
    const sourceDocument = getFileMetadata_(state.sourceDocumentId);
    validateState_(state, sourceJson, sourceDocument);
    state.validated = true;
    saveState_(config.properties, state);
  }

  if (!state.destinationJsonCopyId) {
    const sourceJson = getFileMetadata_(state.sourceJsonId);
    const jsonCopy = Drive.Files.copy(
      {
        name: sourceJson.name,
        parents: [config.destinationFolderId],
      },
      sourceJson.id,
      { supportsAllDrives: true, fields: "id" },
    );
    state.destinationJsonCopyId = jsonCopy.id;
    saveState_(config.properties, state);
  }

  if (!state.destinationDocumentCopyId) {
    const sourceDocument = getFileMetadata_(state.sourceDocumentId);
    const documentCopy = Drive.Files.copy(
      {
        name: sourceDocument.name,
        parents: [config.destinationFolderId],
      },
      sourceDocument.id,
      { supportsAllDrives: true, fields: "id" },
    );
    state.destinationDocumentCopyId = documentCopy.id;
    saveState_(config.properties, state);
  }

  if (!state.notificationSent) {
    sendNotification_(config);
    state.notificationSent = true;
    saveState_(config.properties, state);
  }

  if (!state.sourceJsonTrashed) {
    trashFile_(state.sourceJsonId);
    state.sourceJsonTrashed = true;
    saveState_(config.properties, state);
  }

  if (!state.sourceDocumentTrashed) {
    trashFile_(state.sourceDocumentId);
    state.sourceDocumentTrashed = true;
    saveState_(config.properties, state);
  }

  config.properties.deleteProperty(stateKey_(state.sourceJsonId));
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
      "Files copied: 2 (verification JSON and source document)",
      "Destination folder: " + destinationFolderUrl,
    ].join("\n"),
  });
}

function createState_(handoff) {
  return {
    stem: handoff.stem,
    sourceJsonId: handoff.jsonFile.id,
    sourceDocumentId: handoff.documentFile.id,
    validated: false,
    destinationJsonCopyId: null,
    destinationDocumentCopyId: null,
    notificationSent: false,
    sourceJsonTrashed: false,
    sourceDocumentTrashed: false,
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
          !state.sourceJsonId ||
          !state.sourceDocumentId ||
          typeof state.validated !== "boolean" ||
          key !== stateKey_(state.sourceJsonId)
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
    stateKey_(state.sourceJsonId),
    JSON.stringify(state),
  );
}

function stateKey_(sourceJsonId) {
  return HANDOFF_STATE_PREFIX + sourceJsonId;
}

function validateState_(state, jsonFile, documentFile) {
  const jsonName = splitFileName_(jsonFile.name);
  const documentName = splitFileName_(documentFile.name);
  if (
    state.stem.indexOf(HANDOFF_STEM_PREFIX) !== 0 ||
    jsonName.stem !== state.stem ||
    documentName.stem !== state.stem ||
    jsonName.extension !== "json" ||
    DOCUMENT_EXTENSIONS.indexOf(documentName.extension) === -1
  ) {
    throw new Error("Unexpected handoff filename.");
  }
  if (Number(jsonFile.size) > MAX_JSON_BYTES) {
    throw new Error("Verification JSON is too large.");
  }
  if (Number(documentFile.size) > MAX_DOCUMENT_BYTES) {
    throw new Error("Source document is too large.");
  }

  const payload = JSON.parse(downloadFileText_(jsonFile.id));
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

function findCompleteHandoffs_(folderId, excludedJsonIds) {
  const jsonFilesByStem = {};
  const documentFilesByStem = {};
  const files = listFilesInFolder_(folderId);

  files.forEach(function (file) {
    const name = file.name;
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
      addFileByStem_(jsonFilesByStem, stem, file);
    } else if (DOCUMENT_EXTENSIONS.indexOf(extension) !== -1) {
      addFileByStem_(documentFilesByStem, stem, file);
    }
  });

  return Object.keys(jsonFilesByStem)
    .filter(function (stem) {
      return (
        jsonFilesByStem[stem].length === 1 &&
        !excludedJsonIds[jsonFilesByStem[stem][0].id] &&
        documentFilesByStem[stem] &&
        documentFilesByStem[stem].length === 1
      );
    })
    .sort()
    .map(function (stem) {
      return {
        stem: stem,
        jsonFile: jsonFilesByStem[stem][0],
        documentFile: documentFilesByStem[stem][0],
      };
    });
}

function listFilesInFolder_(folderId) {
  const files = [];
  let pageToken;

  do {
    const options = {
      q: "'" + folderId + "' in parents and trashed = false",
      spaces: "drive",
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
      fields: "nextPageToken,files(id,name,size)",
    };
    if (pageToken) {
      options.pageToken = pageToken;
    }

    const response = Drive.Files.list(options);
    Array.prototype.push.apply(files, response.files || []);
    pageToken = response.nextPageToken;
  } while (pageToken);

  return files;
}

function getFileMetadata_(fileId) {
  return Drive.Files.get(fileId, {
    supportsAllDrives: true,
    fields: "id,name,size",
  });
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

function downloadFileText_(fileId) {
  const url =
    "https://www.googleapis.com/drive/v3/files/" +
    encodeURIComponent(fileId) +
    "?alt=media&supportsAllDrives=true";
  return UrlFetchApp.fetch(url, {
    headers: {
      Authorization: "Bearer " + ScriptApp.getOAuthToken(),
    },
  }).getContentText("UTF-8");
}

function trashFile_(fileId) {
  Drive.Files.update(
    { trashed: true },
    fileId,
    null,
    { supportsAllDrives: true, fields: "id,trashed" },
  );
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
