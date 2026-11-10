# Adobe Scan 21.03.20: lokale Bildaufbereitung

Nachtrag: Die native AutoColor-/GrayScale-Verarbeitung und ein separater Schatten-Nachbau wurden inzwischen auf den Smude-Testbildern ausgeführt. Ergebnisse und Grenzen stehen im [direkten Vergleich](adobe-direct-comparison.md). Der folgende Text dokumentiert den vorherigen Stand der statischen Untersuchung.

Stand: 10.11.2026. Statische Untersuchung ausschließlich der vom Nutzer angegebenen APK; keine Aussage über heutige Adobe-Scan-Versionen. Es wurde keine Adobe-Pipeline ausgeführt und keine Adobe-Laufzeit oder Farbqualität gemessen.

## Ergebnis

Diese Version kombiniert ein kleines gelerntes Schattenmodell mit klassischer nativer Nachverarbeitung. Die entscheidende Architektur ist: **kleines RGB-Schattenbild schätzen, daraus ein Korrekturfeld aufbereiten, das hochaufgelöste Original pixelweise korrigieren und die Ausgabe tonwertmäßig nachbearbeiten**. Der Netzoutput wird nicht einfach als fertige, hochskalierte Dokumentseite verwendet.

Das passt grundsätzlich zum mobilen Laufzeitziel und ähnelt der Aufteilung des getesteten GC-Net-Ansatzes. Es belegt jedoch weder gleiche Gewichte oder Trainingsziele noch gleiche Farbtreue. Insbesondere ist nicht geprüft, ob Adobe das blaue Smude-Cover besser behandelt.

## Belegte Aufrufkette

Die Java-/Kotlin-Schicht ruft `Page.cleanImageWithMagicClean` → `MagicCleanEdgeDetection.cropAndCleanCustom` → `CameraCleanAndroidShim.CropAndClean` → JNI `cropAndClean` auf. Das Shim lädt `libMagicClean.so`.

Die UI-Werte werden in native Reinigungsstufen übersetzt:

| UI-/Analytics-Modus | UI-Wert | Native Stufe |
|---|---:|---:|
| Original | 0 | 0 |
| AutoColor | 1 | 2 |
| GrayScale | 2 | 4 |
| WhiteBoard | 3 | 6 |

Quellen: [MagicCleanEdgeDetection.java](../../outputs/research/adobe-scan-20261110/jadx/sources/com/adobe/dcmscan/algorithms/MagicCleanEdgeDetection.java), [CameraCleanAndroidShim.java](../../outputs/research/adobe-scan-20261110/jadx/sources/com/adobe/magic_clean/CameraCleanAndroidShim.java), [DCMScanAnalytics.java](../../outputs/research/adobe-scan-20261110/jadx/sources/com/adobe/dcmscan/analytics/DCMScanAnalytics.java).

Die native Bibliothek ist weitgehend ohne Funktionsnamen, aber JNI-Registrierung, Relokationen und RTTI erlauben die Zuordnung. Ghidra lud sie mit Basis `0x100000`; die folgenden Adressen sind Ghidra-Adressen, keine Dateioffsets. Der Pfad verläuft über `0x10e630` (JNI), `0x12de7c`, `0x12e030`, `0x12e50c` und `0x130604`. Der letzte Controller initialisiert das Schattenobjekt über `0x143bd8` und ruft dessen virtuelle Verarbeitungsfunktion auf. Je nach Modus und interner Klassifikation gibt es Abzweigungen; das Netz läuft nicht nachweislich bei jeder Auswahl.

Quelle: [native-decompiled.c](../../outputs/research/adobe-scan-20261110/native-decompiled.c). Decompiler-Ausgabe ist rekonstruierter Pseudocode, kein Originalquelltext. Einzelne Typen und Funktionssignaturen sind erkennbar unvollständig.

## Schattenmodell und tatsächliche Nachverarbeitung

Das Modell `asr_v001.bin` ist direkt in `.rodata` eingebettet. Die native Ladefunktion schreibt **934.204 Bytes** ab Dateioffset `0x8e51b` in die Modellablage. Der FlatBuffer ist lesbar; SHA-256: `5c2f20ed83d094fee8c6974386704da6a845d05c2c9e44c7cc4726e00b620c71`.

- Eingang `ExpandDims`: Float32 `[1,256,256,3]`. Die native Vorbereitung in `0x144364` teilt die RGB-Bytes durch 255.
- Ausgang `output_shadow_image`: UInt8 `[1,256,256,3]`.
- 9 `CONV_2D`, 8 `TRANSPOSE_CONV`, 8 Konkatenationen sowie Normalisierungs-/Elementoperationen. Die Tensorbezeichnungen nennen Encoder und Decoder. Das ist ein Encoder-/Decoder-Netz mit Verbindungen zwischen Stufen; eine bestimmte Publikation oder Trainingsmethode ist damit nicht identifiziert.
- 856.604 Bytes konstante Buffer. Das ist kein exakter Parameterzähler und keine Laufzeitmessung.

Quellen: [geparste Modellstruktur](../../outputs/research/adobe-scan-20261110/models.json), [reproduzierbares Inspektionsskript](../../outputs/research/adobe-scan-20261110/scripts/inspect_models.py), native Funktionen `0x142618`, `0x144364`.

Die Postprocessing-Funktion **`0x144908`** ist besonders aufschlussreich:

1. Sie vergleicht verkleinerten Eingang und vorhergesagtes Schattenbild kanalweise und erzeugt daraus eine Maske. Die Byte-Differenzen werden auf Werte 0 bis 2 geprüft; das ist kein symmetrischer Absolutdifferenztest.
2. Eine **3×3-Mittelwertglättung** des Schattenbilds wird mit der ungeglätteten Version maskiert gemischt. Die Schleifen von `0x1159d4` belegen den Mittelwertfilter.
3. Wenn beide Originaldimensionen größer als 1023 sind, werden Zwischenfelder zunächst auf **1024×1024** bearbeitet und danach auf die Originaldimensionen gebracht.
4. Eine **256×256-Lookup-Tabelle** kombiniert jeden Originalkanalwert `I` mit dem aufbereiteten Feldwert `B`. Aus dem Pseudocode, ohne Byte-Rundung und Sonderfall des Nulldivisors:

   `d = max(B − I, 0) / 255`

   `q = d² / 0.3`, falls `d ≤ 0.3`, sonst `q = d`

   `O = clamp(255 × I / (I + 255 × q), 0, 255)`

   Die Konstanten `0.3` und Exponent `2` sind im Konstruktor und in den regulären Reinigungsstufen sichtbar. Die Tabelle wird kanalweise auf die hochaufgelösten Originalpixel angewandt. Diese Beschreibung ist aus dem Code hergeleitet; die Benennung `B` als Beleuchtungs-/Hintergrundfeld ist eine semantische Interpretation des Modelloutputs und seiner Verwendung.
5. Sind anschließend **alle drei RGB-Kanäle mindestens 250**, setzt die Funktion sie ausdrücklich auf **255**. Hier ist das gewünschte `#FFFFFF` also als konkrete Ausgaberegel nachweisbar.

Quelle: [Schatten-Postprocessing](../../outputs/research/adobe-scan-20261110/00144908.c). Der Grenzwert 250 wurde aus der referenzierten Konstanten im ELF gelesen, nicht aus einem bloßen String geraten.

Zusätzlich enthalten die nativen Pfade Histogrammverarbeitung und LUT-basierte Tonwertkorrekturen. `0x114c8c` zählt ein 256-Bin-Histogramm und bestimmt Grenzen; `0x11485c` erzeugt vier stufenabhängige 256-Einträge-Tabellen. In der Bibliothek stehen auch `CALABBitmap` und Halide-Generatoren für Blur, Dilate, Histogram und StretchLChannelHistogram. Diese Namen belegen verfügbare Bausteine, aber allein nicht deren vollständige Reihenfolge in jedem Farbmodus.

## Andere Modelle nicht mit Farbrestaurierung verwechseln

Neben dem Schattenmodell wurden drei weitere eingebettete FlatBuffer geparst:

| Modellzweck laut Metadaten/RTTI | Eingabe | Ausgabe |
|---|---|---|
| Dokumentklassifikation | 384×384 RGB | 5 Wahrscheinlichkeiten |
| Dokumenterkennung | 160×160 RGB | 2 Wahrscheinlichkeiten |
| Seitengrenzen | 256×256 RGB | drei Karten mit 1, 1 und 3 Kanälen |

Die APK enthält `libtensorflowlite.so` und `libtensorflowlite_gpu_gl.so`. Das belegt die gebündelten Laufzeiten, jedoch keinen garantierten GPU-Pfad auf jedem Gerät. Quellen: [Inventar](../../outputs/research/adobe-scan-20261110/inventory.txt), [Modellgraphen](../../outputs/research/adobe-scan-20261110/models.json).

## Bedeutung für Smude

Für eine eigene mobile Lösung sind vor allem das kleine Netz, die Korrektur des Originals über ein aufbereitetes Feld, explizite Weißregeln und getrennte Farb-/Graustufenmodi interessant. Die problematische Unterscheidung zwischen bedruckter Fläche und Papier bleibt trotzdem zentral: Die Smude-Versuche zeigen gerade, dass ein schnelles Feldmodell ein blaues Cover zerstören kann.

Es wurde kein Adobe-Code und kein Adobe-Modell in Smude integriert. Für einen fairen Qualitätsvergleich müsste die vollständige Adobe-Verarbeitung einschließlich ihrer Verzweigungen auf identischen Eingaben ausgeführt werden. Modelltraining, digitale Farbgrundwahrheit, heutige Produktimplementierung und tatsächliche Android-Laufzeit bleiben offen. JADX meldete 31 Dekompilierungsfehler im Gesamtpaket; die oben zitierten Wrapper, Moduszuordnungen und nativen Funktionen waren lesbar.
