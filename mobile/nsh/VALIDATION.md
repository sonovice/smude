# Validierung – 8. November 2026

## Build 20 – Titel, Sammlungen und Sprachen (9. November)

Robuste Grundlinien-Schätzung statt Buchstabenmittelpunkten; die neue Regression mit wechselnder Zeichenhöhe scheiterte vor dem Fix und besteht jetzt. Die lokal verfügbare erste Seite von „Test 2“ wurde vor/nachher vollständig verarbeitet; beide Titelzeilen sind gerader. Das zweite aktuelle Original ist lokal nicht vorhanden und bleibt ungeprüft. Bestehende harte Musikschutz-, Render- und Normalisierungstests bestehen.

Core: 150 Bibliotheks-, 8 Anwendungs-, 7 Mesh-, 12 GCS- und 2 Textschutztests bestanden. Mobile: 35 Bibliotheks-, 15 UI-, 3 Kamera- und 7 Exporttests bestanden; vorhandene Opt-in-Tests unverändert. Clippy/Format bestanden. Geprüft sind unter anderem drei Rasterspalten mit größeren Vorschauen und ohne feste Textleerräume, Sprachwechsel bei erhaltenen Sammlungsdaten, Auswahl/Undo/Rotation, native Dialogaktionen und die vorschaubehaltende v6-Cache-Migration.

iOS-Gerätepaket Build 20 signiert und Signatur geprüft; Simulatorpaket gebaut/installiert. Native XCTest-Sprachwahl für EN, DE, PT-BR, PT-PT und die Präferenzliste FR→DE besteht. Der native Rastertest bestätigt drei Spalten und den Umbruch nach der dritten Karte. Der Viewer-Test prüft die kompakten Aktionen und den nativen Mehr-Dialog; die plattformeigene Popover-Darstellung wird außerhalb angetippt, wenn sie keinen sichtbaren Abbrechen-Button zeigt. Das Plus hat im nativen Header keinen NSH-Test-Identifier; der Test wurde auf den tatsächlichen lokalisierten UIKit-Button korrigiert. Android Release, Lint und JVM-Tests bestanden; signierte APK `outputs/releases/smude-0.1.0-20-arm64.apk`. Auf dem Android-Emulator wurden EN-GB, DE und PT-BR über OS-App-Sprachen ausgewählt; reale Hierarchien zeigen die passenden Übersetzungen und regionalen Datumsformate.

Die physischen Geräte sind aktuell beide nicht erreichbar: iPhone-Zugriff zweimal CoreDevice 4016, iPad-Installation läuft in einen Timeout. Daher **kein erfolgreicher Geräte-Update-Nachweis** für Build 20. Signiertes Gerätepaket bleibt unter `.nsh/device20.app` für die spätere Installation erhalten. Auf dem iPad wurden keine UI-/Funktionstests durchgeführt. Keine Bibliothek auf einem physischen Gerät verändert. Kein Commit/Push/TestFlight-Upload.

## Build 19 – Fortschrittsanzeige (9. November)

NSH Progress in Viewer und Seitenübersicht folgt optionalen Core-Ereignissen für Vorbereitung, Seitenerkennung, Analyse, Entzerrung, Rendern, Aufbereitung und Speichern. Zwischen echten Grenzen werden Zeitanteile geschätzt und nach erfolgreichen Durchläufen lokal kalibriert, getrennt nach Modell und Bildgröße. Doppelseiten teilen das Budget; erneute Analysen verschieben den angezeigten Fortschritt nicht rückwärts. Cachetreffer und Fehlschläge trainieren die Zeiten nicht. Der Balken bleibt bis zur Ergebnisfreigabe unter 100 %. Keine Änderung am Rendercache oder an Bildoperationen.

Core: 149 Bibliothekstests und 8 Anwendungstests bestanden, darunter der echte Zweitseiten-Ereignisablauf. Mobile: 33 Bibliotheks-, 14 UI-, 3 Kamera-Koordinaten- und 7 Exporttests bestanden; vorhandene externe/Hardwaretests bleiben opt-in. Der neue UI-Test prüft NSHs Progressbar-Rolle und Wertebereich sowie das Verwerfen veralteter Seitenrevisionen. Core/Mobile Clippy und Formatprüfungen bestanden; Desktop-Check bestanden. Android Release, Lint, JVM-Tests, APK-Signatur und Buildnummer 19 geprüft.

Derselbe signierte Build 19 ist auf iPhone und iPad installiert; iPhone erfolgreich gestartet. Auf dem iPad wurden ausdrücklich nur Installation und kein UI-/Funktionstest durchgeführt. Für weitere Updates ist das iPad ebenfalls Installationsziel, ohne zusätzliche Tests, solange nicht angefordert. Vorhandene Projekte wurden nicht für Tests verändert. Kein Commit, Push oder TestFlight-Upload.

## Rust

- `cargo check --all-targets`: bestanden.
- `cargo clippy --lib --tests -- -D warnings`: bestanden.
- `cargo test --lib --test app`: 13 bestanden (5 Geometrie-/Domänentests, 8 UI-Abläufe).
- `cargo check --lib --target aarch64-apple-ios-sim`: bestanden.
- Eigenständige Rust-UI, keine TSX-Anwendungsdateien.

## iOS

Release-Simulatorpaket durch `native/ios/package.sh` gebaut und installiert. XCTest gegen die tatsächliche UIKit-App auf iPhone-18-Pro-Max-Simulator, iOS 27:

1. Randwischen abbrechen, danach vollständig ausführen: bestanden.
2. Projekte → Seiten → Viewer → Kamera → Export, native Zurücknavigation: bestanden.
3. Im Viewer nach links/rechts blättern; danach vom Rand zurück ins Projekt: bestanden.

Ergebnis: **3/3 bestanden**. Artefakt `.nsh/navigation-final.xcresult`, Bilder in `/tmp/smude-nsh-final-screens/`. XCTest prüft Gestenresultate und Zustandsgrenzen; keine pixelgenaue Referenz für jeden Animationsframe. Die Übergänge stammen vom nativen UINavigationController.

Simulator hat keine physische Kamera. Kamera-/Modellcode und Swift-Brücke sind im Paket enthalten; reale Live-Konturen, Aufnahmequalität, Landscape-Wechsel und Berechtigungs-Wiederkehr müssen auf einem physischen Gerät zusätzlich geprüft werden. Ein iOS-15–25-Gerät ist nicht getestet; der iOS-26-spezifische Content-Pop-Guard ist verfügbarkeitsgeschützt.

## Android

Custom Gradle-Host: Debug-APK und R8-Release-APK gebaut, Java-Compilation und Android Lint bestanden (0 Fehler). NSH-Rust-Root/JNI-Symbole sowie die erforderliche NDK-C++-Laufzeit im Paket geprüft.

Arm64-Emulator mit API 35:

- Projekte → Projekt → Viewer und Export geöffnet.
- Mittiges Wischen im Viewer blättert von Seite 1 auf 2.
- Linke Randgeste zeigt während des Ziehens die vorherige Ansicht; Zurückziehen an den Rand bricht ab und erhält Seite 2.
- Vollständige Randgeste entfernt genau einen Navigationseintrag.
- Rechte Randgeste verlässt die Kamera; anschließend keine aktiven Camera-Service-Clients.
- Zurück auf Projektebene verlässt die App zum vorherigen OS-Task.
- CameraX zeigt die virtuelle Kameraszene; ohne gültige Seite bleibt der Auslöser deaktiviert.

Screenshots: `/tmp/smude-nsh-android-visual/`, einschließlich `back-progress.png`. Kein physisches Android-Gerät war per ADB verbunden. Reale Papierkontur-Ausrichtung, Schärfe, Aufnahmequalität und ältere API-Stufen bleiben Hardwaretests. Ein zuletzt ergänzter 180°-Display-Rotationslistener und der API-28–32-Back-Button-Fallback wurden kompiliert; diese zusätzlichen Pfade wurden nicht als erfolgreich getestete Hardware-/Altgerätefälle ausgegeben.

## Funktionsintegration (neuerer Prüfstand)

Der frühere UI-Entwurf ist jetzt mit der gemeinsamen Core-Pipeline, dauerhaften Medien, nativen Importen und tatsächlichen Exportdateien verbunden. Die früheren Navigationstests oben bleiben als historischer Gestennachweis erhalten.

- 32 reguläre Rust-Tests bestanden (16 Library/State, 8 UI, 2 Kameraadapter, 6 Export), Clippy ohne Warnungen.
- Zehn Workflow-Regressionen prüfen insbesondere Doppelseiten-Reihenfolge, Capture-Layout, getrennte Eingangs-/Ergebnisrotation, alte Jobs, gelöschte Seiten und fehlgeschlagene Neuaufnahmen.
- Echter vollständiger Lauf mit `input_fullsize.jpg`: GCS 3086×4208 und Mesh 3028×4134, keine Warnungen. Wiederholung lädt identische Ergebnisdateien aus dem Cache; Originalbytes unverändert. Debug-Zeiten sind keine Release-Performancewerte.
- Reale Zwei-Seiten-Exporte unabhängig mit Poppler geprüft: Graustufen 1.435.323 Bytes, echte 1-Bit-PDF 113.916 Bytes (Standardgröße). Rasterwerte, Seitenreihenfolge und Drehung geprüft; alle PDF-Seiten gerendert und angesehen.
- Android API35: neues Projekt mit sichtbarer Tastatur, echter PhotoPicker-Import, GCS, Mesh-Wechsel, PDF-Share-Chooser, 1-Bit-Kleinexport und Wiederöffnen nach Prozessende bestanden. Fünf Instrumentierungs- und drei Kamera-Stabilitätstests bestanden. Kein Empfänger gewählt, nichts versendet.
- iOS native Services: HEIC sowie EXIF-Orientierungen 1–8, volle Bildgröße, atomare Importübergabe und arm64-iOS15.1-Kompilierung geprüft.
- Kamerastabilität: 1200 ms maximale Frame-Altersgrenze; kurze ungültige Messungen höchstens 600 ms überbrücken, ohne das Alter der gültigen Erkennung zu erneuern. Bewegung, Fehler und Lifecyclewechsel löschen die Freigabe sofort.
- Klassifikator: Gewichte und Ausgaben bei identischem Originaltensor bitgleich. Eine vorhandene vollständige Buch-Mockup-Vorlage liefert Auto-Double mit zwei sichtbaren und akzeptierten Konturen. Das Mockup ist KI-generiert, kein Nachweis realer Buchkameraqualität. Die 154 bisherigen Fotos liefern keine Double-Vorhersage; bekannte Beispiele zeigen nur eine vollständige Seite plus angeschnittene Nachbarseite.

Physische Android-Kamera, vollständiges reales Buch als positiver Doppelseitenfall und große Batch-Projekte bleiben zusätzliche Hardwareprüfungen. Das System-Teilen bestätigt lediglich die geöffnete Freigabe, keine Zustellung.

## Installation auf dem iPhone

Der signierte Geräte-Release wurde am 8. November 2026 auf SiWa (iPhone 16 Pro) erfolgreich als `com.notavium.smude` installiert. Rust, native Kamera und Bundle verwenden jetzt konsistent mindestens iOS 15.1 (ONNX-Runtime-Anforderung). Der automatische Start wurde vom gesperrten Gerät abgewiesen; das ist noch kein Laufzeit-/Kameranachweis auf dem physischen Gerät. Die vorherige Smude-App blieb erhalten.

## Neuer Funktionsbuild auf iOS

Der vollständige XCTest-Ablauf besteht auf dem iOS27-Simulator: neues Projekt mit fokussierter Tastatur → tatsächlicher PhotoPicker → GCS → Drehung → Mesh → PDF → UIActivityViewController. Das erzeugte PDF wurde zusätzlich aus dem App-Container gelesen, mit Poppler geprüft und gerendert: eine 4134×3029-Graustufenseite, 8 Bit, 90° entsprechend der Viewer-Drehung. Ergebnisbundle: `/tmp/smude-ios-functions-4.xcresult`.

Ein dabei reproduzierter iOS-Headerfehler ist behoben: eigene pressbare Views in der Titelleiste leiteten den Druck nicht zuverlässig weiter. Die App verwendet für Export, Projektmenü und Kamera-Fertig nun NSHs native iOS-Bar-Items, mit derselben Rust-Aktion. Der zuvor fehlgeschlagene vollständige Test besteht damit.

Der neue signierte Funktionsbuild ist erfolgreich auf dem physischen iPhone SiWa installiert; vorhandene App-Daten werden beibehalten. Androids abschließendes APK wurde ebenfalls neu gebaut. Der Installationserfolg ist kein zusätzlicher Hardware-Kameranachweis.

## Projektgesten

Der Projekt-Header enthält jetzt den nativen Teilen-Button. Langes Drücken auf eine Seite beginnt die Auswahl; Ziehen verschiebt die ausgewählte Seite, ohne die Reihenfolge der übrigen Seiten zu vertauschen. Teilen verwendet die Auswahl oder alle Seiten. Löschen mit Undo und zugängliche Vor/Zurück-Aktionen stehen im Auswahlmodus; der große Exportbutton entfällt. Langes Drücken auf den Projekttitel öffnet Umbenennen/Löschen.

Acht aktualisierte Rust-UI-Tests und Clippy bestanden. `ProjectGesturesTests` im iOS27-Simulator prüft die tatsächliche Long-Press-Geste, Drag von Seite1 auf Position2, Auswahl-Teilen, Löschen, Undo und Teilen aller Seiten; bestanden. Ergebnis: `/tmp/smude-project-gestures-1.xcresult`.

Android API35 wurde mit denselben vier Seiten geprüft: Long-Press aktiviert die Auswahl, Ziehen speichert die Reihenfolge `[999002,999001,999003,999004]`, der obere Teilen-Button öffnet die Auswahl `2`. Ein Gestenabbruch durch Ändern der Z-Reihenfolge beim Pan-Start wurde behoben: die ausgewählte Karte wird bereits vor dem Ziehen angehoben. Die endgültige iOS-Regression besteht ebenfalls (`/tmp/smude-project-gestures-final2.xcresult`). Temporäre Diagnoseinstrumentierung entfernt.

## Auswahl, Live-Reihenfolge und Toast

Auswahl fügt keine Texte oder Zeilen über den Bildern ein. Absolute Auswahlkreise erscheinen auf allen Karten; ausgewählte Seiten und Projekte haben Haken und gelben Rahmen. Projekt-Long-Press zeigt das native iOS-Löschsymbol beziehungsweise einen gezeichneten Papierkorb auf Android. Zusätzliche Fertig-Haken unter Bildern entfernt, Abstand zur Beschriftung erhöht.

Die Live-Reihenfolge benutzt ausschließlich native Transforms der unveränderten Kartenhierarchie. Während des Ziehens werden die beiden nativen Positionswerte gelesen; die anderen Karten wechseln mit 160ms-Animationen den Platz. Erst Loslassen schreibt die Reihenfolge, Abbruch setzt die Vorschau zurück. Reduce Motion wird berücksichtigt. Ein Android-Screenshot während der laufenden Geste zeigt bereits Seite2 links und die gezogene Seite1 rechts (`/tmp/smude-selection-mid-drag.png`).

Root berücksichtigt die untere und seitlichen Safe Areas, die Aktionsleisten fügen die Safe Area nicht doppelt hinzu. Der Undo-Toast ist ein absolutes Overlay über dem Layout, bleibt oberhalb des Home-Indikators und verschwindet nach sieben Sekunden; neue Meldungen ersetzen den alten Timer.

17 reguläre Library-/State-Tests und 9 UI-Tests bestanden, Clippy ohne Warnungen. Native iOS-Tests prüfen unveränderte Kartenposition bei Auswahl, sichtbare Kreise/Selected-Zustand, Ziehen, Teilen, Löschen/Undo, Projekt-Long-Press und tatsächlichen Toast-Ablauf. Beide Tests bestanden in `/tmp/smude-selection-native-1.xcresult`; abschließender Stand zusätzlich in `/tmp/smude-selection-native-final.xcresult`. Der signierte Build ist auf SiWa installiert.

## TestFlight-Erstupload

Smude `0.1.0 (3)` wurde am 8. November 2026 über den persönlichen Account `waloschek@pm.me`, Team `76CP95AA8F`, erfolgreich zu App Store Connect hochgeladen. Die letzte bestätigte Servermeldung lautet „Upload succeeded“ und „Uploaded package is processing“. Der App-Eintrag wurde über Xcode als „Smude“ mit Bundle-ID `com.notavium.smude` angelegt. Externe Testergruppe und Beta-Freigabe sind noch nicht eingerichtet; dafür fehlt die Browser-Anmeldung.

Die native Paketierung enthält jetzt das aus der Benutzervorlage vorbereitete, opake 1024×1024-App-Icon, von actool erzeugte iPhone-/iPad-Icons und Asset-Katalog. Xcode-Plattformmetadaten, arm64-Geräteanforderung, iPad-Orientierungen und PrivacyInfo.xcprivacy sind im Paket enthalten. Der Standalone-Packager ließ zuvor Metadaten aus; die erste Servervalidierung verlangte deshalb explizit arm64. Build3 bestand diese Prüfung und wurde vollständig übertragen.

Nachweis: `/tmp/smude-testflight-upload3.log`; Archiv und Export unter `.nsh/testflight/`. Keine öffentliche Store-Veröffentlichung und keine Tester-Einladungen durchgeführt.


## Build 5: camera orientation, overlay, capture resolution and mobile controls

Physical camera orientation is now independent of interface/display rotation on
iOS and Android. Model input rotates to upright; contour coordinates return to
the unchanged preview. Orientation changes invalidate pending inference and the
capture latch. Unknown/flat poses preserve the last known cardinal orientation.
The shared Rust overlay interpolates small updates over 220 ms, matches contour
vertices and page order, and resets on lost pages, changed dimensions/count or
large moves. It never changes capture eligibility.

Both native capture paths explicitly request their largest offered photo size
and prioritize quality. iOS sets output and per-photo maxPhotoDimensions (legacy
high-resolution opt-in before iOS 16). Android includes slow high-resolution
CameraX sizes and writes the whole in-memory JPEG, preserving pixel dimensions
and setting EXIF without the preview crop. An existing physical-iPhone capture
was verified as 3024×4032 before this change. A newly captured maximum-resolution
photo and a rotation-locked real double-page capture remain physical acceptance
checks; requested resolution is not proof of delivered sensor resolution.

The native header now creates projects with a plus; the large bottom action is
removed. Toasts dismiss on tap without undoing the action or cancelling work.

Verification: mobile Library and UI suites, Clippy and formatting; Swift
orientation and existing stability helpers; Android orientation unit tests and
an API35 instrumentation test verifying unchanged JPEG samples/dimensions at
all four EXIF rotations. Real-model photo replay at 90/180/270 degrees preserves
classification, validity and exact preview contour mapping. Native iOS header/
toast create-delete-dismiss flow is checked separately in HeaderToastTests.


### Build 6: actual rotation-locked flat-book reproduction

The user's rotation-locked screenshot reproduced a remaining failure in Build 5:
no device cardinal pose was available while pointing down at the book. The model
classified the sideways spread as two_page_landscape; fixed left/right splitting
produced overlapping contours and rejected a duplicate page. This was not a UI
shutter bug: an upright 90-degree input accepts both pages with unchanged guards.

The live adapter now tries the two orthogonal rotations only for this explicit
failed sideways-double case. It accepts only complete, non-overlapping Double
results classified two_page_portrait with score >=0.5, preferring the stronger
classification. Coordinates return to the original preview. The native adapters
reuse the accepted analysis orientation for subsequent frames and for the photo,
including when gravity is ambiguous. A cached hint never grants capture: fresh
frames must still satisfy the existing stability and age checks. iOS retains the
hint across expiry of a slow search, but clears it on real orientation/lifecycle
changes, avoiding repeated expensive acquisition.

A test using a crop of the exact user screenshot failed before and passes after
this change (SMUDE_CAMERA_FLAT_SPREAD). Replay also verifies 90/180/270-degree
images and exact reverse contour coordinates. This source contains the old
yellow overlay and is therefore a bug reproduction, not a boundary-accuracy
benchmark or a raw sensor fixture. The next physical retest is still required.


## Build 8 — gray paper clouds (2026-11-08)

Shared paper enhancement uses a finer illumination field, guarded by a broader field to preserve filled heads and thick strokes. Two regression tests cover gray clouds crossed by faint print and large gray heads. Core library: 131 passed, two ignored; CLI output regression passed. Mobile library: 21 passed, one ignored. Core Clippy and changed-file formatting pass. Three same-geometry photograph controls were inspected; no formerly <180 pixel becomes pure white in those controls, which is not a ground-truth ink recall metric.

iPhone build 8 installed successfully. Final launch was blocked by the device lock; Test page 1794175004057757 is queued Pending (revision 2) and retains its original/output references until reprocessing on the next launch. Pre-change library backups are under /tmp/smude-iphone-library-before-paper8.json. The original JPEG is 6048 × 8064 pixels. No TestFlight upload. Android build 8 includes the same shared correction; signed APK is generated locally.


## Build 9 — stronger printed black (2026-11-09)

The shared tone curve adds a dark-population median black reference, while anchoring the stronger contrast below a shoulder of192 so bright output coverage remains unchanged. New regressions cover dark outliers washing out the main print, monotonicity, unchanged bright tones and non-binary gray output. Core library133 passed, two ignored; CLI output regression passed; mobile library21 passed, one ignored. Clippy, changed-file formatting, iOS signing and Android release/lint pass.

iPhone build9 installed and launched successfully. Existing Test page1794175004057757 completed reprocessing as Ready, revision3, output renders/mesh-v8-1794175004057755-Mesh-0-Single/page-1/rectified-clean.png. Both actual build8/build9 phone images were copied back for comparison: equal4578×6654 dimensions, no brighter pixels, all previous values>=193 unchanged, visibly stronger filled noteheads. Original source and previous cached outputs remain intact. Signed Android APK: outputs/releases/smude-0.1.0-9-arm64.apk. No TestFlight upload or commit.


## Build 10 — restored grayscale rim cleanup (2026-11-09)

Restored contour-relative gray rim shape classification in the shared final output, retaining measured music/text guards and avoiding binary page-mask erosion or Sauvola transfer. New final rendering seam regression covers inner rims, faint ink and a compact folio.134 core tests pass, two ignored; application4 and CLI1 pass; mobile21 pass, one ignored. Hardware Metal distance test, clippy and formatting pass.

iPhone build10 installed, launched and reprocessed Test page1794175004057757: Ready, revision4, cache mesh-v9. Actual phone comparison with build9: unchanged4578×6654 size,330207 pixels whitened, no darkened pixels or central interior changes. Original and previous cache retained. Output outputs/research/document-enhancement/restored-rims/iphone-build10-clean.png; signed Android APK outputs/releases/smude-0.1.0-10-arm64.apk. Android release/lint/signature pass. No TestFlight upload or commit.


## Build 11 — full page geometry and NSH feedback (2026-11-09)

Batch page analysis now uses the complete ONNX page geometry head (625 inverse-grid points plus ratio). The compact head remains in live camera detection. The learned inverse map is composed before Mesh/GCS/staff residuals and rendered from the original once. Blank-of-music pages receive identity musical residuals, without invented observations. Actual ONNX tests cover a text-only single page, a forced text-only spread with two reversible page outputs, and compact/full boundary and ratio parity. The tested learned grid is bilinear; it does not claim to reproduce the original app's spline renderer.

The measured title support now retains correction through both ends; feathering happens outside it. Music guards stay exactly unchanged in the title experiment. Photo, text-only, double-page and GCS-residual command-line replays succeeded. Final photo/title artifacts are under outputs/research/full-page-model/final-title. Shared paper/black/rim corrections from builds8–10 are retained.

Feedback uses nsh-base0.4 toast(), toast_stack(), ToastManager and ToastMotion directly in Rust. No extra native presenter, Material dependency or Coordinator host remains. Info is3s by default; Undo6s, adjusted by OS accessibility preferences, paused during interaction/background and persistent for assistive action access. Native service bridges supply preferences only. Messages clear on tap or swipe, and Undo has a separate accessible target. Screen progress remains contextual. Deletion Undo merges removed items into current library rather than rolling back intervening worker results.

Core138 passed, two ignored; application7 and CLI1 passed. Mobile library23 passed, one ignored; UI11 passed; export6 passed, two ignored. Core/mobile Clippy and formatting pass. Minimal onnx-pages feature build passes with dead-code warnings. Android release, lint and5 JVM tests pass; final signed APK outputs/releases/smude-0.1.0-11-arm64.apk. iOS simulator native test passes project creation, deletion, safe-area clearance, separate accessible Undo, repeat deletion and tap-dismiss. Android release installed/launched on the arm64 emulator; native create/delete/Undo and repeat-delete/tap-dismiss were exercised through the actual Android accessibility hierarchy and pointer input; the scratch project was removed and the existing project retained. No physical Android was connected.

iPhone build11 successfully installed and signature verified. Launch was denied because the phone was locked. Its freshly fetched library contains the Test project with zero pages; no deleted page was restored or library rewritten for testing. Core/CLI photo replay is verified; a fresh physical build11 photo remains untested. No TestFlight upload, commit or push.


## Build12 — textured paper, rim guards, capture orientation and RGB covers

See outputs/research/phone-batch12/REPORT.md for the real10page batch replay and regression evidence. Accepted preview rotation now survives capture/import into Page.input_rotation on iOS and Android. Hardware is set to the preview's base orientation, avoiding double application. All16Swift interface/turn cases and durable queue/legacy metadata tests pass; a new physical sideways capture remains to be checked.

All10 corrected results were uploaded to the user's existing Test project, preserving IDs/order/originals and old caches. Page8/9input_rotation is90°, all images remain Ready. Color cover and corrected page8 were copied back before publication and verified byte-identical. iPhone build12 installed; if locked, opening the app shows the already-rendered results without processing. Android signed APK12, release/lint/unit tests pass. New mixed RGB/Gray PDF test uses Poppler to check actual lossless RGB samples. No TestFlight upload or commit.

Final publication: library copied back equals the staged manifest; all10 pages are Ready, IDs/order/source references unchanged. iPhone build12 launched successfully after publication.


## Build13 — local capture sharpness, page-relative alignment and title isolation

Shared core computes regional focus diagnostics from sparse transverse original-pixel staff profiles; no extra model or page resize. The photographed page2 is flagged at its upper third; sharp lower supports do not suppress it. Blank/no-staff regions remain unassessed. Synthetic local-defocus/sharp/blank controls pass. This is a first heuristic calibrated on the available batch, not independently validated blur accuracy.

The Rust camera UI adds a passive smoothed page-relative alignment target from opposing model boundary edge scales. Preview rotations map its vector back into preview axes. A label deadband prevents jitter; it never changes shutter eligibility. Curvature can affect the cue; it is not a calibrated gravity/tilt angle. Regional blur warnings appear on thumbnails, in the viewer with Retake/Keep, and for a newly completed soft capture during a camera batch. Retake preserves the previous page until replacement succeeds.

A global worst-derivative text scale was suppressing page3's measured title. Local signed Lipschitz envelopes now preserve exact music zeros and positive Jacobians without suppressing distant text. Real measurement fixture plus existing rendered/normalized music identity regressions pass. Core143tests pass, two ignored, application7, CLI2, text-safety2. Mobile24library/12UI/7export pass with documented opt-in ignores. Native iOS warning→retake→back→keep test passes. Core/mobile Clippy, format and GUI test compilation pass; Android release/lint/5JVM pass, emulator installed/launched without new fatal logs. Build13 iPhone signed/installed. New physical camera captures/guide usability still require hands-on testing.

All10 original test pages were re-rendered; current GCS choice on2 was noticed in a fresh library and preserved through a separate GCS render. Original JPEGs, IDs, order, rotations and previous caches are retained. New cache metadata carries regional quality through the normal pipeline. Uploaded2/3 PNGs are copied back and hash-checked before publishing new library references. Signed Android APK: outputs/releases/smude-0.1.0-13-arm64.apk. No commit/push/TestFlight upload.

Final delivery: iPhone build13 installed and launched. All10 target pages updated; originals/order/latest model and display-rotation choices retained. Freshly created other project/pages were preserved unchanged. Uploaded2/3 images are byte-identical to local outputs; final library publication read back and verified. Camera guide still needs hands-on assessment during a fresh capture.

Final publication: both Maria outputs copied back and verified byte-identical, both page IDs Ready with errors cleared; latest manifest fetched/merged/read back, all other entries unchanged. iPhone build14 installed. The first launch attempt timed out; corrected results and manifest are already verified on the device. Original source paths/IDs/order preserved. No commit/push/TestFlight upload.

Launch retry: device disconnected immediately after connecting. Installation and image/manifest verification had already succeeded; reopen the app manually to load the published Ready results. Android14 signed release also installed/launched on the arm64 emulator.

Build 20 delivery update: installation succeeded on both physical devices (iPhone and iPad). Per the user’s latest instruction, installation only: no app launch, UI/functional test, second-photo inspection or library modification was performed. The earlier connectivity failure is resolved for installation.


## Build 21 — capture responsiveness, independent workers and collection/viewer layout

Camera capture now durably reserves one or two stable page records before asking the native camera to expose. Native accepted count/rotation reconciles that reservation. Saving completes through a root-lifetime service event even after Done/back or a retake has closed the camera screen. Failed retakes restore the previous page; rejected unacquired shots remove their placeholders. App restart recovers atomically saved JPEGs, and expired unfinished requests become retryable failures. Raw captured previews become visible as soon as the JPEG exists, before source import or dewarping completes.

iOS writes the full-resolution camera JPEG atomically without decoding, rotating and recompressing it. EXIF normalization is deferred to normal processing. Supported responsive capture releases acquisition after the sensor exposure; the fallback releases when the compressed photo arrives. Android copies and closes ImageProxy before using a separate bounded save executor. Both adapters retain in-flight saves after preview destruction, bound pending captures, and require a fresh valid detector frame before accepting another shot. Maximum sensor dimensions and quality settings are unchanged.

The app has three independent bounded worker lanes: one import, one full-image processing job and one export. State/manifest merges remain on the NSH executor, using the latest library; native/blocking threads never mutate the manifest. Imports and export preview are regression-tested while a deliberately blocked render is still in flight. New image processing waits while the camera is visible; an already running job is not cancelled. This is not a pool of simultaneous full-resolution image-processing jobs, and no batch throughput gain from such a pool is claimed.

Collection title moves to the native header, duplicate body text and the device/count row disappear, import actions gain icons, the viewer uses an exclusive Original/Processed toggle, and pagination shares a centered baseline. iPhone Info.plist supports portrait only; the iPad override retains all four orientations. Android uses portrait below 600dp smallest width and leaves tablets unrestricted. Camera physical-orientation detection remains independent of UI orientation.

Final mobile suite: 43 library tests passed (one opt-in ignored), 17 UI tests, three camera-coordinate tests (four opt-in ignored), seven export tests (two opt-in ignored). Clippy with warnings denied and formatting pass. Native iOS simulator CaptureUXTests passed the collection title, camera Done, comparison toggle, pager alignment and phone portrait policy; the subsequent worker separation changes no view geometry. Pure Swift stability-gate checks pass the fresh-frame requirement. Android release/lint/JVM validation and signature/alignment checks are recorded with the final package.

The isolated iOS simulator file-preparation benchmark uses the same 49MP JPEG with EXIF orientation on three runs. Old decode/rotate/re-encode medians are about 2315ms; raw atomic-save median is about 2.46ms. This excludes sensor/ISP latency, immutable source import, deferred orientation, detection and dewarping, and is not a physical-device or end-to-end speed measurement. Inputs/script, measurements and a scope-labelled chart are under outputs/research/capture21/.

Physical-device policy for this delivery: installation only on iPhone and iPad; no launch, photo fetching, library modification or functional/UI tests. No commit, push or TestFlight upload.

Final delivery: build 21 with all three independent worker lanes successfully installed on both physical iPhone and iPad (devicectl outcome success). Neither device was launched or tested. The signed Android APK outputs/releases/smude-0.1.0-21-arm64.apk passes signature verification and 16KB alignment, release/lint/JVM tasks pass, and it is installed on the emulator. The Android app remains portrait when emulator user rotation is forced to landscape; the emulator rotation settings were restored afterwards. Native iOS UI checks ran only on the simulator.


## TestFlight build 21 — uploaded and submitted

Smude 0.1.0 (21) was uploaded through the existing Xcode account on 9 November 2026. The upload log /tmp/smude-testflight-upload21.log confirms Upload succeeded / EXPORT SUCCEEDED. Archive and distribution records are under .nsh/testflight/Smude-0.1.0-21.xcarchive and upload21/.

App Store Connect completed processing. Beta description, review instructions, no-sign-in requirement, feedback address and user-provided review contact were saved. Build-level export compliance was completed. The external group Family & Friends was created; Apple required creating an internal group first, which remains empty with automatic build distribution disabled. Build 21 was submitted for external Beta App Review and the build list confirms Waiting for Review with Family & Friends attached. No individual testers were invited, no public invitation link was enabled, and automatic tester notifications were disabled for this submission. No public App Store submission, commit or push.

Authoritative build ID: bd91618e-0907-47d5-872c-c46f097501d4. Latest machine-readable delivery status: .nsh/testflight/upload-status21.json.


## Android build 22 — native adaptive launcher icon

The legacy square launcher PNG let launchers add their own white background. Added an adaptive icon with explicit dark background, shared source artwork, safe insets and roundIcon metadata. The unchanged full-resolution source PNG is copied by Gradle into generated resources. The installed-icon regression fails against build 21 and passes against the updated app on the Android emulator, including preservation of gold artwork inside a circular mask and a dark rendered rim. Native output was visually inspected at outputs/research/android-icon22/launcher-icon.png. Release build and lint pass; the signed APK outputs/releases/smude-0.1.0-22-arm64.apk passes signing/alignment verification and is installed on the emulator. The physical Pixel was not connected, so no physical-device installation is claimed. iOS/TestFlight build 21 is unchanged. No commit or push.


## Build 23 — DIN-A output framing

Core framing is the final pixel operation after enhancement/rim reconstruction. It classifies the detector's paper ratio within 4% of DIN A (portrait or landscape), pools original nonwhite samples into a bounded component grid, includes detected staff/bar/text protections, removes confirmed exterior whitespace, and copies retained image rows onto a pure-white exact 70:99 raster with symmetric opposite margins of at least 10mm. It never resamples retained notation. Blank/unknown/non-DIN pages, explicit content crop, no-final-crop/no-a4 and preserved RGB covers do not enter this stage. The existing geometry, observations and camera behavior are unchanged. Added final_framing_seconds profiling and explicit final_frame metadata.

Regression coverage includes retained pixel equality in all diagnostic/clean variants, faint unrecognized headers/folios, portrait/landscape aspect and minimum margins, exclusions, mapped overlays/probes and nonlinear preview roundtrips. Mobile PDF export reads the matching pipeline declaration to set 210×297mm, respects page rotation and compressed export resolution, and retains legacy observed sizing for undeclared images. The mobile cache is page-v7; migration preserves source, prior preview and user model/rotation choices.

Real local replay: outputs/research/a4-23/full-photo/page-1/rectified-clean.png and text-page/page-1/rectified-clean.png. Full music page is 3220×4554, text-only page 1120×1584; both exact A4 ratio and visually inspected, retaining music/header/footer. Full-photo framing stage is 0.015681s on this Mac (one run; excludes fitting/rendering, no mobile speed claim). All existing core/application/CLI tests, new A4 integration, mobile library/UI/camera/export tests, core/mobile Clippy and GUI compilation passed. Device packaging/installation results follow below. No commit, push or TestFlight upload requested for this change.

Final A4 delivery: signed build 23 successfully installed on the user's iPhone, iPad and Maria's connected iPhone, with no launch or functional test on any physical iOS device. Android release/lint/JVM checks pass; signed APK outputs/releases/smude-0.1.0-23-arm64.apk verifies signature and 16KB alignment and is installed on the emulator. No physical Android was connected. The additional requested home/camera/menu/grid UX changes remain deferred until after this A4 delivery, per user priority.

## Build 24 — mobile navigation, camera batches and PDF sharing (2026-11-10)

Viewer actions use a square/rotation arrow and more space below the header. Correction-method selection is removed: new pages use Mesh and repository loading migrates legacy GCS records once, preserving the source, rotation and previous preview while queuing a Mesh refresh where the source exists. Serialized legacy enum values remain readable; the core/CLI algorithms are unaffected.

Sharing is PDF-only and headed Share/Teilen/Compartilhar. It contains an editable filename stem with a separate muted, noninteractive .pdf suffix, three size profiles and the Share action. Selection summaries, format/appearance choices, explanatory text, duplicate filename and export preview are removed. Profiles are actually encoded off-thread: Small 1600px/JPEG75, Standard 2600px/JPEG88, Large full resolution/JPEG95, without upscaling. Exact resulting byte counts drive the displayed human-readable sizes and accessibility labels. Sharing atomically copies the chosen prepared PDF, without encoding again. Preparation is cancelable between pages/profiles; stale results cannot replace a different page selection and cache ownership keeps files alive only while needed. RGB covers and grayscale music retain separate PDF color spaces; A4 physical dimensions survive resolution changes and rotation.

Home exposes Camera next to Plus. Unassigned camera batches persist as draft collections; they can be reviewed, shared, deleted, named as a collection or merged into an existing one. Page IDs, capture groups and in-flight import/render results survive a merge. Empty completed camera sessions are discarded. Collection headers expose Camera and More (Photos, Files, Share); normal bottom import/add controls are removed. Both grids have three phone columns and add columns with tablet width. Reordering uses the actual column count and discards a drop if the column count changed during the drag. Thumbnail container heights follow their width rather than retaining the former 190px blank area.

Native iOS fixes: apply header configuration before reconciling the navigation stack; exclude UIKit's background double-tap recognizer on UINavigationBar from canceling custom header presses (iOS 27). Video inspection showed that these changes alone did not prevent UIKit's title/glass morphing, even with a custom bar item. The explicit app back item therefore delegates to one stack-owned native UIButton outside the bar's transitioning contents. A transparent custom spacer reserves its width; actual native bar geometry positions the persistent button, and presses resolve the current screen callback. Root hides it. The native stack still handles interactive back, including cancellation. Only public UIKit APIs are used.

Verification: 46 library tests, 19 headless UI tests, three camera coordinate tests, eight existing export tests and three prepared-PDF tests passed. The added grid-height assertion passed independently after the final layout adjustment. An explicitly run Poppler test independently decodes both gray and RGB pages from all three JPEG-PDF profiles, verifying dimensions, rotation and cover color. Clippy with warnings denied and formatting pass. Native iOS Header24Tests and Sharing24Tests pass against the final persistent-button implementation: both home header buttons, empty capture/Done, collection actions, viewer spacing/menu, fixed suffix, measured size labels, actual system PDF share sheet, repeated forward/back navigation, canceled edge-back and committed edge-back. A recorded transition strip confirms the back label stays sharp and in place; ordinary button press feedback remains. The compact thumbnail is included in the final tested simulator app. Earlier intermediate back-button approaches were rejected by the visual check, and the invalid standalone FixedSpace-item prototype was corrected before physical delivery.

Android release build, lint and JVM tests passed. Emulator UI verified Camera, permission flow, Done and Plus/name entry; no physical Android was connected. APK is arm64 with the existing signing key and adaptive launcher icon. Physical iOS delivery is installation only, with no launch, camera test, reading or modification of user collections. Delivery results are recorded below. No commit, push or TestFlight upload requested for this change.

Final delivery: build 24 with the persistent native back button successfully installed on the user's iPhone, iPad and Maria's connected iPhone. Installation only on all three; no launch or physical-device test. Final signed Android APK is outputs/releases/smude-0.1.0-24-arm64.apk; signature and 16KB alignment checks pass and it is installed on the emulator. Final native screenshots, transition video/contact strip and UI-test log are retained at outputs/research/mobile-ui24/. The final iOS UI test also confirms both a canceled edge gesture and a completed return to the collection.

## Build 25 — separate native header items and A4 rounding (2026-11-10)

The home and collection headers no longer host two custom icons inside one UIBarButtonItem. A small optional native-component symbol contract lets the iOS adapter turn each pressable header action into a separate SF Symbol UIBarButtonItem with its own background. Targets are cached per action, respect current disabled state and retain the NSH callbacks/accessibility identifiers. Other native single items also opt out of shared backgrounds on supporting UIKit versions. Camera/plus, camera/more and selection trash are covered. Other platforms keep their separate header actions. The custom persistent back button was subsequently removed at the user’s request. Back is the ordinary UIKit item with Minimal display mode (chevron only); native edge transitions remain in UIKit. There is no app-owned back overlay or background.

Native regression first reproduced build 24's shared capsule: the supposed gap had intensity 52 against a header background of 21 (difference 31). Build 25 passes the separation check for home and collection; tests additionally verify each button is visibly rendered, hittable, independent and functional, including collection camera and all More actions. Native images are retained in outputs/research/mobile-ui25. The interim request to halve the A4 margin was withdrawn: MARGIN_MM remains 10.

The existing framing guaranteed a minimum 10mm and centered the content in an exact 70:99 pixel grid. Actual old reports confirmed approximately 11.5mm side margins and 19.3mm vertical margins on the full music page, and 11.6/11.8mm sides with 15.8mm vertical margins on the text page. Additional whitespace has two causes: retaining A4 without stretching the content, and avoidable coarse raster rounding. The user explicitly chose A4/unmodified notation with extra whitespace on the necessary axis. Framing now picks the smallest feasible canvas with A4 aspect rounded to the nearest pixel, enforcing the 10mm minimum on both physical axes. It retains every source content pixel, symmetric opposite padding up to one pixel, mapping translations and true physical A4 PDF pages. Actual four-side margins_mm are now reported for diagnosis. Small declared A4 rasters are accepted by PDF export using a pixel-aware aspect tolerance. Mobile render cache is page-v8, retaining the old preview during refresh.

Three core A4 tests and the full pipeline A4 integration pass, including exact retained pixel samples, faint headers/footer, exclusions, overlays, probes and linked preview. Mobile: 46 library and 19 UI tests pass; eight existing export tests and the added small/pixel-rounded A4 PDF test pass. Core and mobile Clippy with denied warnings pass. Android release/lint/JVM and iOS device builds complete. Only generated superseded app archives and package build artifacts were removed when disk exhaustion interrupted a host link; source images/user collections were untouched. No commit, push or TestFlight upload. Physical device actions are installation only; final outcomes follow below.


## Build 26 — pointer-owned reordering and native-back delivery (2026-11-10)

Page reordering now follows the initiating pointer ID rather than a gesture centroid. Holding near the top/bottom scrolls automatically with a bounded quadratic speed ramp (64pt activation band, up to 700pt/s). A second pointer scrolls the collection independently and pauses edge scrolling; it can finish scrolling after the held pointer drops. The held card's transform includes the native scroll offset in the UI-thread animation graph, so it stays under the original finger. Preview destinations and final drops include the actual scroll displacement; persistence happens only on the held pointer's release. Cancellation restores the unchanged order and stops scrolling. Mouse/headless hosts retain their existing pan fallback; phone/tablet hosts use raw touch identities.

The Android runtime needed one targeted fix: ScrollBox's scroll-change callback previously canceled all raw touches during any scroll while touching, including programmatic movement with native user scrolling disabled. The vendored 0.4.0 Java source adds the enabled guard; Gradle filters the upstream ScrollBox/inner classes from its generated jar and compiles the replacement. Cargo registry and installed SDK sources are unchanged; the upstream Apache license is included.

Native QA exposed a fractional-width wrapping error: three requested columns could wrap into two actual columns (the third page started at y=350pt instead of y=136pt). Flooring each cell's share fixes collection and home grids. Native iOS and Android checks now require the first three cards to share a row, so drag destination geometry matches the visible grid.

Final validation:

- 46 mobile library, 21 UI and nine export tests pass (three opt-in external/real-pipeline tests remain ignored). UI tests replay real component touch routing with distinct pointer IDs, include edge scrolling in both directions, canceled gestures, original-order preservation and drops after scrolling. Existing reorder/export behavior remains covered. Clippy with denied warnings and format check pass.
- iOS simulator: separate native header items and UIKit Minimal/chevron-only Back pass. The 30-page native long-press/edge-scroll/drop test passes; reopening preserves the page number. Home compact-list/three-column-grid persistence passes. Logs: /tmp/smude26-grid-fixed.log; native artifacts and screenshots at outputs/research/mobile-ui26/. Prepare the isolated simulator fixture with native/ios/seed-reordering-fixture.py.
- Android emulator: PageReorderingInstrumentedTest injects actual public MotionEvents with two independent pointers. It verifies a stable held-card center during second-finger scrolling, three visible columns, no early library write, correct final position and no lost pages. It restores the prior emulator library. Release, lint and JVM checks pass. Final APK outputs/releases/smude-0.1.0-26-arm64.apk passes signature verification and 16KB alignment and is installed on the emulator; no physical Android was connected.
- Final build 26 includes the corrected 10mm/A4 framing, separate native header items and restored standard UIKit chevron Back from build 25. There is no custom persistent Back control. Installation only on physical iPhone/iPad/Maria's iPhone; no launch, photo retrieval, library modification or physical UI test. Final device outcomes are in .nsh/*-install-build26-final.json.

No commit, push or TestFlight upload.


## Build 27 — screenshot-review implementation and shutter feedback (2026-11-10)

All eleven findings are mapped to implementation and evidence in outputs/research/mobile-ux27/IMPLEMENTATION.md. Shared PagePreview prevents false Processed/demo fallbacks and labels retained results; collection/selection menus and share headers carry exact counts; empty/search/draft flows expose useful actions. Native menus, destructive semantics, keyboard tap handling and platform-sized hit targets are covered. Viewer uses available height on tablets. Camera distinguishes fatal reasons, exposes appropriate fallback/settings, names retake targets and gives a native-animation dark shutter cue on capture_started without blocking Done or batch capture. EN/DE/PT application copy is updated; no processing/cache/storage migration.

49 library + 26 UI + nine export tests, denied-warning Clippy, formatting, release/lint/JVM pass. Native checks: seven iPhone UX flows, one expanded UIKit header hit target, three iPad flows and four Android tests. Native Android flash opacity is observed rising and settling after a synthetic accepted-capture event; no physical camera claim. iOS simulator has no hardware preview, Android uses its virtual room. New images are in the implementation artifact folder. Test libraries restored and camera grant revoked. Build 27 installed successfully on the user's physical iPhone and iPad, installation only. Maria's iPhone unavailable. Signed/aligned outputs/releases/smude-0.1.0-27-arm64.apk installed on emulator. No commit, push or TestFlight upload.
