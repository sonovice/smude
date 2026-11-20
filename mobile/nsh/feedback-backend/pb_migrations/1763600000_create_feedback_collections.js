/// <reference path="../pb_data/types.d.ts" />

// TestFlight-Feedback der Smude-App: Scan-Berichte und Preisumfrage.
// Lesen, Ändern und Löschen nur für Superuser; Anlegen nur mit dem App-Header
// `X-Smude-Token` (PocketBase normalisiert ihn zu `x_smude_token`). Das Token
// hält nur Gelegenheits-Posts ab und ist kein echtes Geheimnis.

const CREATE_RULE = '@request.headers.x_smude_token = "c415095c1254bc2ebea4fbff1e9629dd"';
const KB = 1024;
const MB = 1024 * KB;

const autodateFields = [
  { name: "created", type: "autodate", onCreate: true, onUpdate: false },
  { name: "updated", type: "autodate", onCreate: true, onUpdate: true },
];

const superuserOnly = {
  listRule: null,
  viewRule: null,
  createRule: CREATE_RULE,
  updateRule: null,
  deleteRule: null,
};

migrate(
  (app) => {
    app.save(
      new Collection({
        type: "base",
        name: "reports",
        ...superuserOnly,
        fields: [
          { name: "install", type: "text", required: true, max: 64 },
          { name: "status", type: "text", max: 32 },
          { name: "error", type: "text", max: 2000 },
          { name: "app", type: "json", maxSize: 64 * KB },
          { name: "device", type: "json", maxSize: 64 * KB },
          { name: "page", type: "json", maxSize: 256 * KB },
          { name: "capture", type: "json", maxSize: 256 * KB },
          { name: "original", type: "file", maxSelect: 1, maxSize: 80 * MB, mimeTypes: [] },
          { name: "result", type: "file", maxSelect: 1, maxSize: 80 * MB, mimeTypes: [] },
          ...autodateFields,
        ],
      }),
    );

    app.save(
      new Collection({
        type: "base",
        name: "price_votes",
        ...superuserOnly,
        fields: [
          { name: "install", type: "text", required: true, max: 64 },
          { name: "price", type: "number" },
          { name: "answered", type: "bool" },
          { name: "active_seconds", type: "number" },
          { name: "language", type: "text", max: 8 },
          { name: "app", type: "json", maxSize: 64 * KB },
          { name: "device", type: "json", maxSize: 64 * KB },
          ...autodateFields,
        ],
      }),
    );
  },
  (app) => {
    for (const name of ["price_votes", "reports"]) {
      app.delete(app.findCollectionByNameOrId(name));
    }
  },
);
