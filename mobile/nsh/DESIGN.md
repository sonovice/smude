# Smude Mobile: Bedienmodell und native Anschlüsse

## Aufgaben und Hierarchie

| Ansicht | Inhalt | Hauptaktion | Weitere Aktionen |
|---|---|---|---|
| Projekte | Titel, Vorschaubild, Seitenzahl, Prüfhinweis | Neues Projekt | Suche, Projekt öffnen |
| Projekt | Geordnete Seiten; Status je Seite | Seiten hinzufügen | Auswählen, Export, Umbenennen, Sortieren, Löschen |
| Viewer | Eine Ergebnis-Seite; Original umschaltbar | Seite ansehen / durchwischen | Drehen, Neu aufnehmen, Einzel-Export; Verfahren und Löschen im Menü |
| Aufnahme | Kamerabild, aktuelle Seitenkontur, knappe Anweisung | Auslösen | Fotos importieren, letzte Aufnahme prüfen, Fertig |
| Export | Umfang, Dateiname, Format, Darstellung, Größe | Teilen | Exportvorschau |

Kein dauerhaftes Dashboard und keine zusätzlichen Tabs für dieselben Objekte. Native Zurück-Geste führt entlang dieser Hierarchie. Auswahlmodus ersetzt die normalen Projektaktionen, statt sie zusätzlich anzuhängen. Langes Drücken auf eine Seite öffnet die Auswahl mit Kreisen direkt auf allen Bildern, Haken und gelbem Rahmen bei ausgewählten Seiten. Oberhalb des Rasters werden keine zusätzlichen Zeilen eingefügt. Beim Ziehen gleiten die übrigen Karten in die Vorschau-Reihenfolge; die Medienreihenfolge wird erst beim Loslassen dauerhaft gespeichert. Der Teilen-Button oben verwendet die aktuelle Auswahl oder alle Seiten. Der Projekttitel öffnet beim langen Drücken Umbenennen und Projektlöschen.

## Projekt, Aufnahme, Seite, Variante

- **Projekt** ist ein dauerhafter Arbeitsstand mit geordneter Liste stabiler Seiten-IDs, kein PDF.
- **Aufnahme** enthält das unveränderte, dauerhaft kopierte Original, EXIF-Orientierung und Orientierungskorrektur. Ein Foto kann mehrere Seiten liefern.
- **Seite** verweist auf ihre Aufnahme und einen bestimmten Seitenbereich (single/left/right). Sie hat eine stabile Position und eigene Rotation, Prüfstatus und gewähltes Verfahren.
- **Ergebnisvariante** wird durch Aufnahmeversion, Seitenbereich, Orientierung, Engine-Version und Verfahren identifiziert. Ein verspäteter Job darf eine neuere Aufnahme nie überschreiben. Pro Seite GCS/Mesh cachen; genau die angezeigte Variante exportieren.
- **Export** ist ein unveränderlicher Schnappschuss ausgewählter Seitenversionen in Projektreihenfolge, mit eigenständigen Einstellungen. Keine Originale oder Masterbilder dafür überschreiben.

Produktionsspeicherung im App-Datenverzeichnis: kleine Datenbank oder atomar ersetzte Projektmanifeste; Mediendateien separat. Import erst nach vollständiger Dateikopie bestätigen. Gesicherte Originale niemals im vom System löschbaren Cache. Temporäre Vorschaubilder dürfen regeneriert werden. Löschung mit sieben Sekunden Undo-Frist in einem schwebenden Toast; geteilte Dateien bis zum Abschluss der Freigabe verfügbar halten. Die Implementierung speichert ein atomares JSON-Manifest und eine Wiederherstellungskopie im nativen App-Datenverzeichnis. NSH-Storage wird nur zur Migration früherer Metadaten gelesen.

## Batch und Unterbrechungen

Aufnehmen darf schneller sein als Entzerren. Original sofort sichern, Seitenplätze in Aufnahmereihenfolge reservieren, Hintergrundwarteschlange abarbeiten. Status: wartend → wird verarbeitet → fertig / prüfen / fehlgeschlagen. Nächste Aufnahme blockiert nicht auf Rendering. Fehler pro Seite mit Wiederholen; andere Seiten bleiben benutzbar. Bei Doppelseiten linke/rechte Plätze vor Beginn reservieren, nicht nach Fertigstellungszeit sortieren.

Abbruch einer Neuaufnahme erhält die bisherige Seite. Erst ein erfolgreich gesichertes und akzeptiertes Ergebnis ersetzt sie. Eine einzeln ersetzte Buchhälfte verändert die Geschwisterseite nicht. Retake ist bewusst auf Einzelaufnahme begrenzt; „Doppelseite ersetzen“ wäre eine gesonderte Auswahl.

App-Neustart lädt Projekte und setzt unterbrochene Jobs kontrolliert zurück in die Warteschlange. Kamera bei Hintergrundwechsel pausieren. Bei Speicherfehler keine Speicherung behaupten. Ein laufender Export nennt seine festgelegte Auswahl; unfertige Seiten werden niemals still ausgelassen.

## Viewer und Orientierung

Nur sichtbare Seite und direkte Nachbarn als passende Vorschaubilder laden. Vollauflösung bei Zoom bedarfsgerecht nachladen. Horizontal wischen blättert; bei Zoom verschiebt dieselbe Geste das Bild. Vor/Zurück-Buttons bleiben als Alternative. GCS/Mesh im Seitenmenü, nicht in der Aufnahme oder Projektliste.

Drehung einer fertigen Seite ist reversibel und beeinflusst Anzeige und Export. Falls die ursprüngliche Orientierung schon die Erkennung verhindert hat: Original drehen, betroffene Seiten neu analysieren; bisherige Ergebnisse bis zum Erfolg behalten. Bei gemeinsamen Doppelseiten muss zwischen „diese Seite drehen“ und „Aufnahme neu ausrichten“ unterschieden werden. Die Implementierung trennt Eingangsrotation und Ergebnisrotation; unfertige Originale werden neu analysiert, fertige Ergebnisse verlustfrei in Anzeige und Export gedreht.

## Native Kamera

NSH bietet die View-Integration; Smude liefert die Erkennung. iOS: AVCaptureSession/AVCaptureVideoPreviewLayer. Android: CameraX Preview und ImageAnalysis. Als native NSH-Komponente registrieren, keine Kameraframes über die UI transportieren. Die Oberfläche und der Kontur-Painter sind Rust; Swift/Java enthalten ausschließlich die Plattformanbindung.

Ereignisse: Kamera-/Berechtigungsstatus, Vorschau-Transform, Frame-ID, Zeitstempel, erkannte Seitenpolygone, Split, Qualitätszustand. Alle Konturen in einem eindeutig definierten Koordinatensystem; Rotation, Spiegelung und aspect-fill berücksichtigen.

Freigabe: Kamera bereit, aktueller Frame, vollständige plausible Einzel-/Doppelseite, ausreichend stabile Geometrie, keine laufende Auslösung. Konkrete Schärfe-/Stabilitätsschwellen am Gerät messen. Kleine Analysedimension und höchstens eine laufende Inferenz; veraltete Frames verwerfen. Kein vollständiges Dewarping in der Live-Vorschau. Modell-/Kameraausfall zeigt eine eindeutige Ursache, nicht dauerhaft „suchen“.

Texte: „Seite ins Bild halten“, „Ganze Seite ins Bild nehmen“, „Ruhig halten“, „Seite erkannt“, „Doppelseite erkannt“. Freigabe belegt Erkennung, nicht garantiert erfolgreiche Entzerrung. Eine blockierte Aufnahme bekommt den konkreten Grund; kein stummes Grau. Bei Doppelseiten beide Konturen plus Mittelfalz und die Leserichtung zeigen. Fotoimport führt unabhängig davon in die reguläre Analyse; nicht jedes Bild ist ein Kamerafoto.

Implementiert: native AVFoundation-/CameraX-Anbindung mit tatsächlicher Seitenerkennung, Rust-Kontur-Painter, aktuellem Erkennungszustand und gesperrtem Auslöser ohne gültige Seite. Die Hardwarequalität und Schwellen sind noch auf beiden physischen Geräten zu prüfen. Die Kamera ist ein eigener Navigationseintrag, damit Androids interaktive Stack-Zurückgeste verfügbar bleibt; NSHs Dialog-Backend bietet sie in 0.4.0 nicht.

## Exportvertrag

| Einstellung | Umsetzung |
|---|---|
| PDF | Ein Dokument, eine PDF-Seite je ausgewählter Smude-Seite |
| JPEG | Eine nummerierte Datei je Seite; gemeinsam per nativer Mehrfachfreigabe |
| Graustufen | Tatsächlich ein Graukanal, nicht RGB mit drei gleichen Kanälen |
| Schwarz-Weiß (PDF) | Echte gepackte 1-Bit-Pixel, PDF `DeviceGray`, `BitsPerComponent=1`; Flate oder CCITT Group 4 verlustfrei |
| Klein | Lange Kante maximal 1.600 px; bewusst gewählte Verkleinerung |
| Standard | Lange Kante maximal 2.600 px; zunächst kein ungeprüfter Qualitätsstandard |
| Original | Volle Ergebnisauflösung; niemals hochskalieren |

Voreinstellung PDF + Graustufen + Original. Keine erfundenen dpi bei unbekannter Papiergröße. PDF-Seitenverhältnis aus der fertigen Seite erhalten; für einen Druckmaßstab eine explizite Papiergrößenoption statt stillschweigend A4 erzwingen. Exportgrößen unterhalb der Originalauflösung sind Obergrenzen, keine Pflicht zum Upscaling.

Für Graustufen-PDF Flate (verlustfrei) oder bewusst gewählte Graustufen-JPEG-Kompression; Presets anhand Noten bewerten. Kein verlustbehaftetes JBIG2 und keine Symbolersetzung. JPEG unterstützt hier Grau, nicht „echtes 1 Bit“. Erst skalieren, dann für die gewünschte Zielauflösung binarisieren. Ein verlustfreier PDF-Filter verhindert keine vorherigen Binarisierungsfehler.

Vor Freigabe echte Exportdatei fertigstellen, anschließend tatsächliche Größe anzeigen. Kleine Exportvorschau aus denselben kodierten Daten dekodieren. Bittiefe, Seitenzahl, Sortierung und Rotation automatisiert prüfen; Punktierungen, Vorzeichen, kleine Noten und dünne Bögen zusätzlich visuell gegen das Masterbild prüfen.

Systemfreigabe: iOS UIActivityViewController; Android ACTION_SEND/ACTION_SEND_MULTIPLE mit Content-URIs, passenden MIME-Typen und Leseberechtigung. WhatsApp nur anbieten, wenn das Betriebssystem die App als Ziel verfügbar macht. Teilen-Abbruch ist kein Fehler und keine Zustellbestätigung. Encoder und native Systemfreigabe sind angeschlossen.

## Geprüft / offen

Der aktuelle Implementierungs- und Prüfstand steht in README.md und VALIDATION.md. Entzerrungsjobs, Wiederherstellung, Import und echter Export sind angeschlossen. Große Projekte, Barrierefreiheit und vollständige reale Doppelseiten müssen auf Hardware weiter geprüft werden. Die bestehende produktive Mobile-App wird durch diesen Entwurf nicht ersetzt.
