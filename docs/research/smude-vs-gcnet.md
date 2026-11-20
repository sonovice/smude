# Bestehende Smude-Pipeline gegen GC-Net auf den App-Testfotos

10. November 2026. Vergleich der zusätzlichen Tonwert-/Beleuchtungskorrektur, nicht der Geometrie.

## Entscheidung

**Die bestehende Smude-Korrektur behalten. GC-Net 256 ist auf diesen echten App-Testfotos kein sinnvoller Ersatz.** Die vorherige Empfehlung aus dem Magazin-/synthetischen Pilotversuch war für Smudes reale Eingaben zu optimistisch und wird hiermit eingeschränkt.

Die aktuelle Korrektur erzeugt deutlich kräftigere Notation. GC-Net allein lässt Linien und Überschriften blass und erzeugt in der RGB-Ausgabe lokale Farbstiche. Besonders klar scheitert der Ansatz am blauen Cover aus der früheren App-Testserie: Der obere blaue Bereich wird weitgehend weiß, Teile des weißen Titels verlieren dadurch ihren Kontrast. Die vorhandene Farberkennung lässt dieses Cover dagegen in RGB.

Der geringe Laufzeitvorteil rechtfertigt diese Qualitätsprobleme nicht: Auf derselben 28,44-MP-Notenseite und demselben iPhone 16 Pro benötigt die bestehende Tonwertstufe rund **15,76 ms**, die GC-256-Pipeline **11,97 ms**. Beide liegen deutlich unter dem Ziel von 200 ms.

## Welche echten Bilder?

Alle **zehn Fotos aus der Sammlung „Test“** der installierten NSH-Smude-App. Zuordnung anhand des zuletzt lokal gesicherten Bibliotheksstands `outputs/research/rims32/library-after-update33.json`; Originale aus `outputs/research/test-rims30/device/sources/`, jeweils 6048×8064 Pixel nach EXIF-Ausrichtung. Zusätzlich das farbige Cover „Erste Konzertstücke großer Meister“ aus der früheren Phone-Batch12-Serie, mit bereits vorhandener RGB-Entzerrung.

Ein frischer Bibliotheksabruf vom Telefon schlug fehl; es wird deshalb ausdrücklich die vorhandene lokale Kopie der echten App-Testserie verwendet. Es wurden keine Sammlungen geändert, keine Bilder in Smude ersetzt und Smude nicht neu installiert.

Für die zehn Fotos wurde einmal eine gemeinsame Geometrie mit dem vorhandenen CLI rekonstruiert und die unveränderte RGB-/Graustufen-Ausgabe gespeichert. A4-Rahmung und Randbereinigung wurden bei dieser gemeinsamen Vorbereitung deaktiviert. Danach bekommen **alle photometrischen Varianten dieselben Pixel, denselben Ausschnitt und dieselbe Auflösung**. Die Ausgaben sind daher ein isolierter Vergleich der Tonwertstufe und keine Behauptung, die zuletzt in der App sichtbare Rahmung exakt nachzubilden.

Die aktuelle `src/rendering/output/paper.rs` wurde unverändert in den Versuchsbereich kopiert und in einem kleinen Rust-Programm ausgeführt. SHA-256: `0bea8223fe2cb25f4806fb61ee8fc1b4031b3118aa07c26b4201755292cc1be4`. **Auf allen zehn Seiten stimmt die so erzeugte aktuelle `enhance_print`-Ausgabe pixelgenau mit der Tonwertausgabe des gemeinsamen CLI-Replays überein.** Das verhindert eine unbemerkte alte oder nachgebaute Vergleichsbaseline.

## Varianten und Beobachtungen

1. **Aktuelles Smude:** identische Integer-Luma aus RGB, `normalize_paper`, danach die in `enhance_print` enthaltene lokale Weiß-/Schwarzpunktkorrektur. Bei Notation Graustufen; bei erkannten Farbseiten ohne Notensysteme behält die App RGB.
2. **GC-Net 256:** RGB-Beleuchtungsfeld bei 256×256 schätzen und bilinear auf das Original anwenden. RGB-Ergebnis und daraus gebildete identische Integer-Luma separat gespeichert. Keine neue Geometrie, kein Nachzeichnen durch DR-Net.
3. **Nur klassische Papierkorrektur:** `normalize_paper` ohne zusätzlichen Druckkontrast. Dieser Kontrollversuch trennt die Beleuchtungsnormalisierung vom gewünschten kräftigen Druckbild.
4. **Gezielte Zusatzversuche auf Seiten 1, 3 und 6:** GC mit 512/1024 Pixeln, außerdem GC gefolgt von der vollständigen bisherigen Smude-Korrektur. Letzteres enthält bewusst auch eine zweite klassische Papiernormalisierung und ist kein kostenloser Austausch nur des Schätzers.

Die kräftigere Schwärzung von Smude ist teilweise eine beabsichtigte Wirkung seiner zusätzlichen Kontrastkorrektur. Deshalb wird aus einem blasseren GC-Ergebnis **keine pauschale Quote gelöschter Noten** abgeleitet. Die Nahansicht zeigt viele Strukturen noch vorhanden, aber erheblich kontrastärmer.

GC 256 plus die bisherige Korrektur stellt einen Großteil des kräftigen Druckbildes wieder her; damit bleibt jedoch die bisherige Arbeit bestehen und ein Modell kommt hinzu. Ein klarer Nutzen wurde auf diesen Fällen nicht sichtbar. Bei GC 1024 bleiben auf Seite 3 auch nach dieser Nachkorrektur einzelne Notenbereiche auffällig aufgehellt. Eine höhere Analyseauflösung löst die Aufgabe somit nicht automatisch.

Der Coverfehler ist davon unabhängig: Hier soll die farbige Fläche erhalten bleiben. Sie als aufzuhellendes Papier zu behandeln ist sachlich falsch. Auch für Farbdokumente ist die getestete GC-only-Konfiguration deshalb kein allgemeiner Ersatz.

Es gibt für diese realen Fotos keine digitale Ground Truth und keine manuelle Vollannotation aller Zeichen. Weißanteile und Dunkelanteile in `summary.json` beschreiben nur Tonwerte, keine objektive Qualitätsrangliste. Keine OCR-/OMR-Genauigkeit gemessen.

## Direkter iPhone-Laufzeitvergleich

Eigenständige Benchmark-App `com.notavium.smude.photometryab`; vorhandene Smude-App bleibt unverändert. Derselbe entzerrte Ausschnitt von **Testseite 6, 4561×6236 Pixel = 28,44 MP**, iPhone 16 Pro, iOS 26.6.2, Release-Build, thermischer Zustand nominal.

| Pfad | Warmer Median | Warme Einzelwerte | Erster Durchlauf |
| --- | ---: | ---: | ---: |
| Unverändertes Smude `enhance_print` | **15,76 ms** | 15,39–19,55 ms | 28,47 ms |
| GC-Net 256 + Metal | **11,97 ms** | 9,85–16,44 ms | 1730,63 ms |
| GC-Net 512 + Metal | **43,36 ms** | 40,34–50,09 ms | 42,28 ms |

Jeweils neun Durchläufe, erster separat, Median der acht übrigen. Unterschied zwischen Smude und GC 256 rund **3,79 ms** in dieser kurzen Messreihe; überlappende Streuung und unterschiedliche Ein-/Ausgabeformate verbieten eine allgemeine Geschwindigkeitsgarantie.

**Smude-Scope:** bereits dekodierte Graupixel → originale Rust-Funktionen → Graupixel; einschließlich Eingabekopie für ImageBuffer und Ausgabekopie. Graubild liegt in der bestehenden Pipeline ohnehin vor.

**GC-Scope:** bereits dekodiertes RGBA → Metal-Verkleinerung/BGR-Packing → Core ML FP16 (`all`) → bilineare Feldanwendung in Metal → RGBA. Puffer wiederverwendet, Feldkopie und Synchronisation enthalten. Die Ausgabe ist noch RGB, nicht Smudes graues Exportbild.

Beide Messungen schließen Geometrie, Randbereinigung, A4-Rahmung, Dateidekodierung/-export und UI aus. Ein Produktionsport könnte zusätzlich RGB-Ausgabe/-Transfer im Renderer benötigen; diese Integrationskosten wurden nicht gemessen. Keine Android-Messung, kein thermischer Dauertest. Der hohe erste GC-256-Durchlauf zeigt zusätzliche Startkosten; das Ziel gilt ausdrücklich für warm geladene Modelle.

Die tatsächliche graue Smude-Ausgabe wurde vom iPhone zurückkopiert und mit dem Desktop-Vergleichsbild derselben Seite geprüft: **100 % der Pixel identisch**, maximaler Fehler 0. Damit verwendet der Gerätebenchmark dieselbe Tonwertfunktion und dieselben Bilddaten wie der sichtbare Vergleich.

## Artefakte

- [Vergleichsworkspace](../../outputs/research/color-restoration-20261110/smude-comparison/)
- [Zuordnung der Originale und gemeinsamen Entzerrungen](../../outputs/research/color-restoration-20261110/smude-comparison/manifest.json)
- [Ausgaben und Tonwertbeschreibungen](../../outputs/research/color-restoration-20261110/smude-comparison/results/summary.json)
- [Echte iPhone-Rohmessungen](../../outputs/research/color-restoration-20261110/smude-comparison/results/iphone-report.json)
- [Gleicher Detailausschnitt aller Zusatzvarianten](../../outputs/research/color-restoration-20261110/smude-comparison/results/ablations-detail.png)
- [Coververgleich](../../outputs/research/color-restoration-20261110/smude-comparison/results/cover-contact.png)

Quellstände, Gewichte und Exportherkunft entsprechen dem [ersten Modellversuch](document-color-restoration-experiment.md). Modellqualität bleibt insbesondere durch den dort dokumentierten Drittspiegel der GC-Gewichte begrenzt. Es wurde kein Produktionscode verändert und nichts committed.
