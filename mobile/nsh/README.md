# Smude Mobile – Rust / Native Shell

Die mobile Oberfläche steht in `src/` und ist vollständig in Rust geschrieben. NSH 0.4.0 erzeugt native UIKit-/Android-Views. Swift und Java verbinden Kamera, Medienauswahl und Teilen mit dem Betriebssystem. Die App-ID `com.gridnotelabs.smude.design` bleibt vorerst getrennt von der bisherigen App.

## Funktionen

- Kamera direkt auf dem Home-Screen: dauerhafte, zunächst unzugeordnete Aufnahmeserien lassen sich später benennen, einer Sammlung hinzufügen, teilen oder löschen.
- Dauerhafte Projekte mit eigenen Originalen und Ergebnissen im App-Datenverzeichnis; atomare Metadaten mit Wiederherstellungskopie.
- Kompakte Projektliste mit Änderungsdatum oder Rasteransicht; die Ansicht bleibt gespeichert. Sortierung immer nach letzter Benutzerbearbeitung. Renderjobs ändern diesen Zeitpunkt nicht. Für alte Projekte ohne Bearbeitungszeit wird die letzte bekannte Erfassungs-/Erstellungszeit übernommen.
- Batch-Kamera mit sofort reservierten Seiten und unabhängiger JPEG-Speicherung. Abschließen verlässt die Kamera auch bei laufender Speicherung; Einzel- und Doppelseiten behalten Reihenfolge und IDs. Mehrfachimport aus Fotos und Dateien; HEIC wird nativ konvertiert, JPEG samt EXIF unverändert übernommen.
- Live-Seitenkonturen mit gelber transparenter Füllung; Einzel-/Doppelseitenerkennung. Kurze Erkennungslücken erhalten den stabilen Zustand, tatsächliche Bewegung und veraltete Frames sperren die Aufnahme.
- Importierte Notenausschnitte ohne vollständige Seitenränder werden bei ausreichenden Notenzeilen-/Taktbelegen automatisch verarbeitet. Sichtbarer schmaler Farbhintergrund wird konservativ entfernt; fehlende Blattteile werden nicht ergänzt.
- Gemeinsame Smude-Core-Pipeline mit Mesh. Drei unabhängige, begrenzte Worker übernehmen Import, Bildverarbeitung und Export; die Entzerrung verarbeitet eine Aufnahme gleichzeitig und nutzt intern CPU-Parallelisierung/GPU. Während die Kamera offen ist, starten keine neuen Entzerrungen. Unterbrochene Jobs werden beim nächsten App-Start wieder aufgenommen; der Betrieb im Hintergrund richtet sich nach den OS-Limits.
- NSH-Fortschrittsbalken für die laufende Seite, auch in der Seitenübersicht. Echte Stufenwechsel setzen die Grenzen; dazwischen schätzen gemessene Zeitanteile den Fortschritt. Erfolgreiche Durchläufe kalibrieren die Zeiten lokal, getrennt nach GCS/Mesh und Bildgröße. Doppelseiten und erneute Analysen sind berücksichtigt. Langsame Stufen warten vor ihrer Grenze; die Anzeige bleibt bis zur Ergebnisfreigabe unter 100 %.
- Kamera und Mehr-Menü oben in der Sammlungsansicht; Mehr enthält Fotos/Dateien importieren und Teilen für alle Seiten oder die aktuelle Auswahl. Langes Drücken wählt Seiten aus; Auswahlkreise, Haken und gelbe Rahmen liegen direkt auf den Bildern. Beim Ziehen rücken die übrigen Karten mit kurzen Animationen in die Vorschau-Reihenfolge; gespeichert wird beim Loslassen. Am oberen/unteren Rand scrollt die Übersicht automatisch, mit zunehmender Geschwindigkeit. Alternativ scrollt ein zweiter Finger die Übersicht, während der erste die Seite hält; die gehaltene Seite bleibt dabei unter dem Finger. Löschen erscheint nur im Auswahlmodus. Seiten lassen sich außerdem drehen und unmittelbar wiederherstellen. Eine fehlgeschlagene Neuaufnahme erhält die bisherige Seite. Doppelseiten bleiben zwei unabhängig verwaltbare Seiten derselben Aufnahme.
- Bereichsweise Schärfeprüfung an originalen Notenlinien; gelber Hinweis mit „Erneut aufnehmen“ oder „Behalten“. Die Kamera zeigt eine geglättete seitenbezogene Ausrichtungshilfe; sie sperrt die Aufnahme nicht und ist kein kalibrierter Neigungswinkel.
- Sammlungstitel im nativen Header, Scan-/Import-Icons und keine doppelte Seitenüberschrift. Smartphones bleiben im Hochformat, Tablets dürfen rotieren; die Kamerapose wird unabhängig davon erkannt.
- Viewer mit Durchblättern, Zoom, sichtbarem Original-/Bearbeitet-Umschalter und einem eindeutigen Drehen-Symbol. Eine fertige Seite drehen beeinflusst Anzeige und Export; ein noch nicht erkanntes Original drehen startet eine neue Analyse.
- PDF-Teilen für einzelne, ausgewählte oder alle fertigen Seiten; native Systemfreigabe. Farbige Cover bleiben RGB und Notenseiten Graustufen, auch gemeinsam in einem PDF. Drei vorbereitete Profile zeigen exakte Dateigrößen und verwenden JPEG-Kompression mit Qualität 75/88/95. Der gewählte PDF wird unverändert geteilt. Die verlustlosen und 1-Bit-Exportfunktionen bleiben intern für Tests verfügbar.
- Teilengröße Groß (volle Auflösung), Standard (maximal 2.600 px) oder Klein (maximal 1.600 px), ohne Hochskalierung. Originale bleiben unverändert. Die akzeptierte Drehung aus der Kamera-Erkennung wird mit der Aufnahme gespeichert und vor der Analyse angewendet.
- Projekte per langem Drücken auswählen und über das Löschsymbol oben rechts entfernen. Löschen lässt sich über einen schwebenden Toast rückgängig machen; die Anzeigedauer berücksichtigt die Bedienungshilfen. Toast und Aktionsleisten berücksichtigen die Safe Areas.
- Native Navigation: iOS-Randgeste, Android Predictive Back; Projektanlage öffnet die Tastatur und lässt sich per Fertig-Taste abschließen.

## Sprache und Seitenaktionen

Die Oberfläche verwendet Englisch als Rückfall und wählt Englisch, Deutsch oder Portugiesisch anhand der OS-Sprachreihenfolge beziehungsweise der OS-Einstellung für diese App. `src/i18n.rs` enthält den typisierten Rust-Katalog; native Kamera-/Service-Texte und Berechtigungsressourcen liegen unter `native/ios/*.lproj` und Android `res/values*`. Datumsangaben kommen von den nativen regionalen Formatierern. Kein eigener Sprachschalter ist erforderlich. Die portugiesische Kopie verwendet brasilianische Begriffe; regionale Datenformate folgen weiterhin dem Betriebssystem.

„Sammlung“ ersetzt „Projekt“ in der Oberfläche. Bestehende Dateipfade und serialisierte `Project`-Felder bleiben kompatibel. Seiten- und Sammlungsraster zeigen auf Smartphones drei Spalten und auf Tablets mehr. Im Viewer liegen Drehen, Neuaufnahme und Mehr kompakt über dem Dokument; Teilen bleibt im nativen Header. Seltene Aktionen öffnen einen nativen Auswahldialog. Ergebnisse älterer Pipeline-Versionen werden nach dem Update mit erhaltenem Original und alter Vorschau neu berechnet. Frühere GCS-Seiten wechseln dabei einmalig zu Mesh.

## Bauen und prüfen

[NSH 0.4.0 installieren](https://nsh.dev/docs/installation/), dann in diesem Verzeichnis:

```sh
cargo test
cargo check --all-targets
native/ios/package.sh              # Simulator
native/ios/package.sh ios          # Gerätepaket; anschließend signieren
android/build.sh                   # Android mit CameraX/native Services
```

Der Android-Host ist erforderlich: NSH 0.4.0 bindet eigene Java-Komponenten nicht über seinen Standard-Packager ein. Details in [android/README.md](android/README.md). macOS eignet sich zur UI-Entwicklung; die nativen mobilen Import-/Share-Services sind dort nicht registriert.

`vendor/nsh-platform-ios` behebt einen Paketierungsfehler des veröffentlichten 0.4.0-Crates: dessen gemeinsame `speech.rs` fehlt. Quelle, Änderungen und Lizenz stehen unter `vendor/`. Die dortige Android-ScrollBox-Anpassung erhält die gehaltenen Touch-Pointer beim programmgesteuerten Scrollen; Gradle ersetzt nur die entsprechende Klasse im generierten Runtime-Jar.

## Tests und Grenzen

[VALIDATION.md](VALIDATION.md) trennt Rust-, Simulator- und echte Gerätetests. Die positive automatische Doppelseiten-Testvorlage ist ein bestehendes Buch-Mockup; sie belegt die technische Verbindung, keine Erkennungsquote auf beliebigen realen Büchern. Die Kamera kann vollständige Doppelseiten als Einzelblatt klassifizieren; ein positiver Test auf einem realen Buchfoto bleibt erforderlich.

Native iOS-Tests mit XcodeGen:

```sh
xcodegen generate --spec native/ios/test-project.yml --project .nsh
# App im Simulator installieren und das Testfoto mit simctl addmedia importieren.
xcodebuild test -project .nsh/SmudeNativeTests.xcodeproj -scheme SmudeNativeTests \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro Max' \
  -only-testing:SmudeNativeTests/WorkflowTests
```

`NavigationTests` verwendet die explizit gesetzte Vier-Seiten-Testbibliothek; die normale App startet ohne Beispielprojekte. Export-/Realbildtests mit externen Dateien oder Poppler sind als `ignored` markiert und dokumentieren ihre Voraussetzungen direkt im Test.


A4-Rahmung: Die gemeinsame Core-Pipeline vereinheitlicht DIN-nahe Graustufenseiten nach der Entzerrung mit 10 mm Zielrand (bis auf Pixelrundung), ohne die Notenpixel erneut zu skalieren. Für A4 nötiger zusätzlicher Weißraum bleibt auf der anderen Achse erhalten. Die Rastermaße werden auf einzelne Pixel statt auf ein 70×99-Pixel-Raster gerundet. Der PDF-Export übernimmt deklarierte A4-Seiten als echte 210 × 297 mm (beim Drehen entsprechend quer), unabhängig von der Exportauflösung. Andere Formate und farbige Cover behalten ihre Seitengestaltung. Bestehende Ergebnisse werden über Rendercache page-v8 unter Beibehaltung ihrer Vorschau neu berechnet.

Die native Umsortierprüfung erhält eine eigene 30-Seiten-Testsammlung ausschließlich im Simulator:

```sh
python3 native/ios/seed-reordering-fixture.py SIMULATOR_UDID
xcodebuild test -project .nsh/SmudeNativeTests.xcodeproj -scheme SmudeNativeTests \
  -destination 'platform=iOS Simulator,id=SIMULATOR_UDID' \
  -only-testing:SmudeNativeTests/Reordering26Tests
```

`PageReorderingInstrumentedTest` prüft auf einem Android-Emulator tatsächliche Zwei-Finger-Eingaben, festen Haltepunkt, drei Rasterspalten und die gespeicherte Ablageposition. Er sichert die Emulator-Bibliothek vorübergehend und stellt sie danach wieder her.

Build 27: Umbenennen ist im Sammlungsmenü sichtbar; Freigabe und Löschaktionen nennen den Umfang. Leere Ansichten bieten direkte Aktionen, unzugeordnete Scans folgen der Listen-/Rasteransicht. Der Vergleich trennt Original, aktuelles und bisheriges Ergebnis, ohne fremde Beispielbilder einzublenden. Kamera-Aufnahmen erhalten einen kurzen dunklen Shutter-Flash. [Umsetzung und Prüfung](../../outputs/research/mobile-ux27/IMPLEMENTATION.md).
