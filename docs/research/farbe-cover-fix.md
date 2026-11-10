# Sammlung „Farbe“: Verarbeitung und GC-Net

Stand: 10.11.2026. Build 36 auf dem gekoppelten iPhone „SiWa“ installiert und das Ergebnis aus der App zurückgelesen.

## Fehler und Korrektur

Die Aufnahme des blauen Covers „Erste Konzertstücke großer Meister“ scheiterte mit `no page produced a successful result`. Der Detailbericht zeigt einen unterbestimmten Perspektivfit. Der Musikdetektor fand im Schriftzug vermeintliche fünfzeilige Systeme, auf dem iPhone vier und im Desktop-Replay zwei, jeweils ohne Taktstriche. Das unabhängig erkannte Seitengitter war gültig.

Die gemeinsame Pipeline erhält nun eine begrenzte Rückfallbehandlung: Nur bei vorhandenem gelerntem Seitengitter, deutlicher Farbe, erkannten Systemkandidaten ohne Taktstriche und genau diesem Perspektivfehler wird der Versuch ohne Musikmerkmale wiederholt. Die gültige Seitenentzerrung bleibt aktiv. Die üblichen Notenseiten behalten ihren bisherigen Ablauf; andere Fehler werden weiter gemeldet. Das ist eine Absicherung gegen diesen Fehlertyp, keine allgemeine Reparatur des Musikdetektors.

Verworfene Merkmale und Profil werden als `rejected-music-observations.json` und `rejected-music-profile.json` gesichert. Der Seitenbericht und das öffentliche Verarbeitungsergebnis enthalten eine Warnung zur ausgelassenen Musikkorrektur. Dadurch entscheidet die vorhandene Farbauswahl wieder anhand des Covers und erhält RGB.

Implementierung: [pages.rs](../../src/application/pages.rs). Regressionstest: [application.rs](../../tests/application.rs), `color_cover_with_false_staffs_keeps_its_page_geometry_and_color`, mit dem Original-JPEG und derselben EXIF-Normalisierung wie in der mobilen Ablage.

## Prüfung

- Der Regressionstest reproduzierte vor der Änderung denselben Abbruch. Danach bestanden alle neun Application-Tests.
- `cargo clippy --all-targets -- -D warnings` bestanden.
- Das vollständige Original wurde erneut verarbeitet; anschließend wurde Build 36 auf SiWa installiert.
- XCTest scheiterte bereits beim Initialisieren der UI-Automation mit `Timed out while enabling automation mode`; es wurde kein bestandener UI-Test behauptet.
- Bei geschlossener App wurde eine frische Bibliothekskopie gesichert und ausschließlich die betroffene fehlgeschlagene Seite mit den drei Änderungen von `Page::invalidate` erneut eingereiht. Nach dem App-Start meldete sie **Ready**, keinen Fehler und **4921 × 6534 Pixel**. Das zurückgelesene PNG ist RGB und wurde visuell geprüft. Die anderen Sammlungen sind im Vergleich der Manifeste unverändert.
- Der Gerätebericht misst 1,509 s für die Core-Verarbeitung einschließlich Erkennung, Geometrie, Rendern und Speichern, nach der vorgeschalteten mobilen EXIF-Normalisierung. Dies ist **kein** Tonwert-Benchmark und nicht mit dem 200-ms-Ziel nur für die zusätzliche Farbaufbereitung gleichzusetzen.

## GC-Net auf genau dieser Aufnahme

GC-Net erhielt die erfolgreich entzerrten Originalfarbpixel aus dem Replay. Getestet wurden 256, 512 und 1024 Pixel Modelleingabe; das Korrekturfeld wurde auf die volle Ausgabegröße angewandt. Alle drei Varianten bleichen bedruckte blaue Flächen massiv aus. Bei 256 und 512 wird der obere Bereich nahezu weiß, der weiße Titel verliert weitgehend seinen Kontrast. 1024 zeigt zusätzliche fleckige Aufhellungen. Größere Modelleingabe behebt diesen Fall also nicht.

Smude erhält hier die fotografierten Farben; es rekonstruiert noch keine hypothetischen Originaldruckfarben. GC-Net ist für dieses Cover in der getesteten Form ungeeignet. Die Läufe auf dem Mac dienen der Bildprüfung und sind keine neuen mobilen Laufzeitmessungen.

Artefakte: [interaktiver Vergleich](../../outputs/research/farbe-20261110/vergleich.html), [Kontaktbogen](../../outputs/research/farbe-20261110/comparison.jpg), [Geräteausgabe](../../outputs/research/farbe-20261110/iphone-smude.png), [Verifikation](../../outputs/research/farbe-20261110/verification.json), [Gerätebericht](../../outputs/research/farbe-20261110/phone-fixed-report.json). Der vorangehende Laufzeitvergleich steht in [Smude vs. GC-Net](smude-vs-gcnet.md).
