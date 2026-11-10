# Adobe-Verfahren im direkten Vergleich

10.11.2026. Ausführbarer, isolierter Testaufbau; keine Änderung am produktiven Smude-Filter und kein neues Smude-Deployment in diesem Experiment.

## Was tatsächlich läuft

1. **Originale native Adobe-Pipeline:** Eine separate Android-Test-App lädt die unveränderten DEX-Klassen und ARM64-Bibliotheken aus der bereitgestellten Adobe-Scan-APK 21.03.20. Sie ruft `CameraCleanAndroidShim.CropAndClean` mit normalisierten Vollbild-Ecken, ausgeschalteten Beta-Funktionen und den nativen Stufen 2 (AutoColor) bzw. 4 (in Adobes App als GrayScale bezeichnet) auf. Ausgeführt in einem eigenen Android-35-ARM64-Emulator auf dem M3 Max.
2. **Portabler Schatten-Prototyp:** Das extrahierte TFLite-Modell wird mit TensorFlow Lite auf RGB/255 ausgeführt. Der Python-Prototyp rekonstruiert Differenzmaske, 3×3-Mittelwertfilter, Mischung auf maximal 1024×1024, Hochskalierung, kanalweise Lookup-Korrektur und Weißsetzung ab RGB ≥250. Er enthält **nicht** die anschließende komplette native Farb-/Tonwertverarbeitung. OpenCV-Resampling ist zudem kein Nachweis bitgleicher LibYUV-Rundung.

Die separaten Verfahren verhindern, dass ein unvollständiger Nachbau als das vollständige Adobe-Ergebnis ausgegeben wird. Die native Ausführung belegt die konkret gewählten Bibliothekspfade, nicht das Verhalten der heutigen Adobe-App oder sämtlicher Modi des damaligen UI.

## Eingaben und Vergleich

- Das neue blaue Cover aus „Farbe“.
- Die echten Notenseiten 01, 03 und 06 aus „Test“.
- Zusätzlich die farbige Beispielseite `for_appearance.png` aus dem lokalen DocRes-Repository. Sie stammt nicht aus der Nutzersammlung; links ist dort ausdrücklich das unveränderte Foto dargestellt.

Für die vier Nutzeraufnahmen erhält Adobe dieselben bereits entzerrten RGB-Pixel wie die bisherigen GC-Net-Läufe. Smude und GC-Net werden aus den vorhandenen Ergebnissen auf dieser gemeinsamen Geometrie verglichen. Die Adobe-Bibliothek reduziert große Bilder intern auf rund **9 MP**: das Cover auf 2600×3448, die Notenseiten auf 2560×3512 bzw. 2560×3504. Smude und GC-Net liegen in voller ursprünglicher Ausgabeauflösung vor. Die HTML-Vorschau skaliert sie auf gleiche Anzeigegrößen; das ist kein gleichauflösender Benchmark kleinster Details.

## Beobachtungen

- **Blaues Cover:** AutoColor erhält den weißen Titel wesentlich besser als GC-Net und kräftigt das dunkle Blau. Die obere blaue Fläche bekommt jedoch starke fleckige Aufhellungen. Auch der reine Schatten-Nachbau zeigt diese Flecken. Damit ist das Verfahren kein geeigneter pauschaler Filter für vollflächig bedruckte Cover.
- **Notenseiten:** Adobe produziert weißen Hintergrund und klaren Text-/Notenkontrast. GC-Net allein bleibt deutlich blasser. Smude bleibt auf den geprüften Ausschnitten kräftig und bewahrt seine volle Auflösung; aus dieser Sichtprüfung lässt sich keine objektive Detailerkennungsquote ableiten.
- **Farbe auf weißem Papier:** AutoColor liefert den sichtbar größten Gewinn gegenüber bloßem Erhalten der Fotofarben: weißer Hintergrund, kräftigere Illustration und dunklerer Text. Der reine Schatten-Nachbau ist zurückhaltender. Ohne digitales Original bedeutet kräftiger nicht nachgewiesen farbtreuer.

Für Smude ist daher eine optionale Farbaufbereitung für passende Papierseiten interessant, mit konservativer Behandlung flächig farbiger Cover. Das Experiment rechtfertigt noch keinen generellen Austausch der vorhandenen Tonwertkorrektur.

## Validierung und Grenzen

Die fünf primären AutoColor- und fünf GrayScale-Ausgaben sind erfolgreich zurückgelesen, nicht leer und vollständig deckend. Cover und Notenausschnitte sowie die farbige Referenz wurden visuell geprüft. Bei dem Cover liefern `kCropAndCleanModeDefault` und `Override` für die getesteten Stufen pixelidentische Ergebnisse. Der als GrayScale bezeichnete native Modus enthält weiterhin kleine RGB-Kanalunterschiede; diese wurden nicht heimlich entfernt.

Ein zusätzlicher direkter Aufruf der reinen Crop-Stufe 0 lieferte in diesem Harness trotz Erfolgscode einen leeren Ausgabepuffer. Diese Ergebnisse sind aus dem Vergleich ausgeschlossen. Deshalb wird auch kein bitgleicher Adobe-Originalmodus als Geometriekontrolle behauptet; der Vergleich verwendet die unabhängig vorhandenen Eingabepixel. Die funktionierenden Aufrufe von Stufe 2 und 4 sind direkt anhand ihrer gespeicherten Bildausgaben prüfbar.

Die portable Schatten-Inferenz dauerte in einzelnen Mac-Läufen ungefähr 2–3 ms, die unoptimierte Python-Nachverarbeitung der großen Seiten etwa 0,7–1,0 s. Die gesamte originale JNI-Verarbeitung einschließlich ihrer internen Verkleinerung lag in den einzelnen Emulator-Aufrufen bei etwa 235–446 ms für AutoColor der Nutzeraufnahmen und 181 ms für das kleinere Referenzbild. Das sind **keine** belastbaren iPhone-/Android-Gerätemessungen, keine systematischen Warm-Mediane und kein Nachweis des ≤200-ms-Ziels.

## Artefakte und Wiederholung

- [Interaktiver Vergleich](../../outputs/research/adobe-comparison-20261110/vergleich.html): Bildwahl, Details, ursprüngliche Eingabe und drei Adobe-Varianten.
- [Portabler Prototyp](../../outputs/research/adobe-comparison-20261110/adobe_shadow.py), [Abhängigkeiten](../../outputs/research/adobe-comparison-20261110/requirements.txt).
- [Android-Harness](../../outputs/research/adobe-comparison-20261110/android/src/com/smude/adobecomparison/MainActivity.java), [Build-Skript](../../outputs/research/adobe-comparison-20261110/android/build.py), [Upload und Start](../../outputs/research/adobe-comparison-20261110/android/run.py), [Ergebnisse abrufen](../../outputs/research/adobe-comparison-20261110/android/pull.py).
- [Test-APK](../../outputs/research/adobe-comparison-20261110/android/comparison.apk), ausschließlich im isolierten lokalen Versuch verwendet.
- [Native Messdaten](../../outputs/research/adobe-comparison-20261110/native/report.json), [Referenz-Messdaten](../../outputs/research/adobe-comparison-20261110/native/ref-report.json), [Ausgabeverifikation](../../outputs/research/adobe-comparison-20261110/native-verification.json), [Dateien, Maße und Hashes](../../outputs/research/adobe-comparison-20261110/comparison-manifest.json).

`adobe_shadow.py` enthält eine wiederverwendbare `AdobeShadow.process(rgb)`-Schnittstelle. Der Android-Testaufbau bleibt separat unter `outputs/research/adobe-comparison-20261110`; die Bibliothek ist kein Bestandteil des Smude-Produktbuilds. Für die statische Herleitung siehe [APK-Analyse](adobe-scan-apk.md).
