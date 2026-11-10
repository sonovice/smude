# Dokumentfarben: Modellversuche und mobile Messungen

**Nachtrag:** Der anschließende [Vergleich mit den echten Smude-App-Testfotos](smude-vs-gcnet.md) korrigiert die untenstehende vorläufige Empfehlung: Die vorhandene Smude-Tonwertkorrektur ist bereits sehr schnell und liefert auf diesen Notenseiten das passendere Ergebnis. GC-Net allein schwächt den Druckkontrast und zerstört im getesteten farbigen Cover große Farbflächen. Kein allgemeiner Ersatz für die bestehende Pipeline.

**10. November 2026. Ziel: ≤ 200 ms zusätzliche Verarbeitung pro Seite bei warmem Modell, auf iPhone und Android.**

## Ergebnis

**Für einen mobilen Prototyp ist die erste Stufe von GCDRNet, GC-Net allein, der derzeit beste Ausgangspunkt dieser Versuche.** Mit einem 256×256-Analysebild und Übertragung des Beleuchtungsfelds auf das Original benötigt der vollständige Photometrieschritt auf einem echten **iPhone 16 Pro rund 11 ms für 12 Megapixel**. Das ist eine abgeleitete Variante, nicht das vollständige GCDRNet. Sie entfernt harte Schatten nicht restlos und garantiert keine originalgetreuen Druckfarben.

GCDRNet vollständig und DocRes wurden mit trainierten Gewichten ausgeführt. **DescanDiffusion konnte nicht mit trainierten Gewichten ausprobiert werden:** Der frühere Download ist nicht verfügbar. Dessen Rechenarchitektur wurde separat untersucht; sie liefert keine Aussage zur restaurierten Bildqualität. **Android ist vorbereitet, aber nicht auf einem realen Gerät gemessen.** Ein plattformübergreifendes Erreichen des 200-ms-Ziels ist damit noch nicht belegt.

Der isolierte [Workspace](../../outputs/research/color-restoration-20261110/README.md) enthält Quellstände, Gewichte, Skripte, Ergebnisse, Core-ML-/ONNX-Exporte und eine eigenständige iPhone-Benchmark-App. Keine Änderung am Smude-Produktionscode, kein Commit.

## Echte iPhone-Messungen

Gerät: iPhone 16 Pro, iOS 26.6.2. Core ML FP16 mit FP32-Ein-/Ausgabe, Release-Build. Alle protokollierten Messungen dieser Serie haben thermischen Zustand 0, also nominal. Kurze Einzelgeräte-Versuche, keine thermische Dauerlast- oder Energieuntersuchung.

### Vollständige zusätzliche Photometriestufe

| Variante | Ausgabebild | Zeit warm | Erster Durchlauf bei geladenem Modell |
| --- | ---: | ---: | ---: |
| GC-Net 256 + Metal | 2352×3136, 7,38 MP | **8,65 ms** | 19,94 ms |
| GC-Net 512 + Metal | 2352×3136, 7,38 MP | 36,88 ms | 33,23 ms |
| GC-Net 256 + Metal | 3000×4000, 12 MP | **11,00 ms** | 22,96 ms |
| GC-Net 512 + Metal | 3000×4000, 12 MP | 41,76 ms | 38,08 ms |

Gemessen: bereits dekodierte RGBA-Seite → Metal-Verkleinerung und BGR-Float-Packing → Core ML (`all`) → Übertragung und bilineare Anwendung des RGB-Beleuchtungsfelds in Metal → fertige RGBA-Seite. Enthalten sind Synchronisation, Feldkopie/-allokation und Verarbeitung aller Ausgabepixel. Eingang-/Ausgangspuffer werden wiederverwendet. **Nicht enthalten:** Dateidekodierung, PNG-/PDF-Export, UI, Modellladen/Erstkompilierung. Dies entspricht einer zusätzlichen Stufe innerhalb einer bereits laufenden Bildpipeline.

Je Fall neun Durchläufe, erster separat, acht warme Werte; der Gerätecode berichtet den oberen Median. 7,38 MP sind die echte Originalgröße. Für den 12-MP-Durchsatzversuch wurde dasselbe Foto hochskaliert; das ist kein zusätzlicher unabhängiger Qualitätsfall.

[Rohmessungen](../../outputs/research/color-restoration-20261110/results/iphone-pipeline-report.json) · [Tatsächliches iPhone-Ausgabebild](../../outputs/research/color-restoration-20261110/results/iphone-gc256-native.png)

### Modell allein, ohne Bildvorbereitung oder Rückübertragung

| Modell / Analysegröße | Core ML `all` | CPU + GPU | Nur CPU |
| --- | ---: | ---: | ---: |
| GC-Net 256×256 | 4,22 ms | 5,47 ms | 2,86 ms |
| GC-Net 512×512 | 26,10 ms | 5,40 ms | 9,22 ms |
| GCDRNet 256×256 | 12,41 ms | 4,88 ms | 7,54 ms |
| GCDRNet 512×512 | 98,99 ms | 15,32 ms | 27,12 ms |
| DocRes 256×256 | **308,41 ms** | **259,93 ms** | **515,70 ms** |

Acht Vorhersagen je Konfiguration, erster separat, Median der übrigen sieben. Eingangstensor vorbereitet. `all` ist eine erlaubte Geräteauswahl, **kein Nachweis tatsächlicher Neural-Engine-Platzierung**. Die GPU-Variante ist hier teilweise erheblich schneller; automatische Auswahl darf nicht ungeprüft als schnellste Konfiguration gelten.

**DocRes verfehlt 200 ms in der getesteten 256er-Konfiguration bereits vor Prompt-Erzeugung und Ausgabeanwendung.** Kleinere Modelle, andere Präzision oder andere Architekturen wurden damit nicht ausgeschlossen. Das Original verwendet bei großen Seiten 1600×1600, die hier geprüfte 256er-Variante ist bereits eine starke Reduktion.

**GCDRNet ist als kleines Netz schnell, aber die Auflösung ist wesentlich:** 15 ms für 512×512 bedeuten keine 15 ms für eine vollständige 12-MP-Seite. Die getestete Übertragung seiner kleinen Pixelvorhersage als Verstärkungsfeld erzeugte sichtbare Säume. Eine 1024er-Core-ML-Version wurde zusätzlich exportiert/gebaut; ihr Geräteversuch scheiterte nach der ersten abgeschlossenen Messreihe an der verlorenen iPhone-Verbindung. Keine hochgerechnete Laufzeit als Messwert behandeln.

[Alle 15 Modell-/Backend-Messungen inklusive Ladezeiten](../../outputs/research/color-restoration-20261110/results/iphone-model-report.json)

### Präzision und Startkosten

Die trainierten ONNX-Exporte stimmen auf dem Mac unter ONNX Runtime CPU FP32 mit PyTorch FP32 überein: maximale absolute Abweichung GC-Net `1,2e-6`, GCDRNet `3,7e-6`, DocRes `2,2e-6` bei der geprüften 256er-Eingabe. ONNX-Prüfung erfolgreich.

FP16 Core ML ist nicht bitgleich: Beim vollständigen GCDRNet 256 erreicht der maximale einzelne Kanalfehler unter `all` 0,0536, bei mittlerem Fehler 0,00102. CPU+GPU reduziert diesen Maximalfehler auf 0,0172. Solche lokalen Fehler gehören vor einem Produktport geprüft, insbesondere an dünnen Zeichen.

Für die **komplette GC-256-Pipeline auf der nativen Magazinseite** wurde das zurückkopierte iPhone-Bild separat gegen einen FP32-Hostlauf verglichen: mittlerer absoluter Kanalfehler **0,352/255**, 99. Perzentil **3/255**, Maximum **5/255**. Orientierung und BGR/RGB-Anordnung sind geprüft. Keine nichtendlichen Ausgaben in den Modellvergleichen. Das belegt diesen konkreten Portierungstest, nicht sämtliche Dokumente.

Modellladen/Erstkompilierung kann deutlich länger dauern: beispielsweise DocRes `all` beim ersten Versuch etwa 24 s Laden, anschließend etwa 526 ms erste Vorhersage. GC/GCDR weisen ebenfalls Erstnutzungskosten auf. Das Nutzerziel „warm geladen“ wird ausdrücklich verwendet; daraus folgt keine Aussage über Kaltstart oder Downloaddauer.

[Pipeline-Parität](../../outputs/research/color-restoration-20261110/results/iphone-pipeline-parity.json)

## Bildqualität

Eingaben: zwei öffentliche Autorenbeispiele (Magazinseite mit Grafik/Foto und harter Schattenkante, Lehrbuchseite mit blauen Elementen), eine bereits entzerrte Smude-Notenseite und ein synthetischer Farbstress-Test. Die zwei Autorenbeispiele wurden ohne zusätzliche Geometriekorrektur verarbeitet; sie sind nicht als neue perfekte Entzerrungsreferenzen deklariert. Gemeinsamer Vergleich bei 1024 Pixeln längster Kante, zusätzlich beide Autorenbeispiele in nativer Auflösung. GCDRNet nutzt dabei volle gepaddete Auflösung, DocRes den originalen 1600er-Verhältnisfeldpfad.

Die PyTorch-Bildversuche verwenden FP32/MPS; das ist eine dokumentierte Abweichung vom FP16-DocRes-CLI. Die Modelle selbst und ihre Gewichte bleiben unverändert. Alle realen Bilder haben **keine bekannte ursprüngliche Digitalvorlage**; ihre Farbtreue lässt sich daher nicht objektiv aus dem schöneren Eindruck ableiten.

Beobachtungen:

- **GCDRNet vollständig:** Auf der nativen Magazinseite überzeugende Aufhellung und kräftige Farben. Im kontrollierten Stress-Test verschwinden jedoch große Teile sehr heller Flächen. Ein kleines vollständiges GCDRNet-Ergebnis als Beleuchtungsfeld zu verwenden verursacht zusätzlich auffällige Kontursäume und Textartefakte; dieser Schnellpfad wird verworfen.
- **DocRes appearance:** Gute Schattenbereinigung auf dem Magazinbeispiel; auch der verkleinerte Analysepfad wirkt dort brauchbar. Ebenfalls Verlust heller Flächen im Stress-Test. Laufzeit der getesteten mobilen Variante überschreitet das Ziel.
- **GC-Net allein:** Die 256er-Beleuchtungsfeldvariante erhält in diesen Tests helle Inhalte besser und passt gut zur schnellen Anwendung auf Originalpixel. An der harten Schattenkante bleibt ein Streifen sichtbar, ebenso teilweise Durchscheinen. Kein universeller Qualitätsgewinner.
- **Klassische lokale Papierweißdivision:** Günstiger und im einfachen synthetischen Beleuchtungsfall farbtreuer; auf dem Magazinbeispiel ebenfalls sichtbare Restschatten. Die synthetische Konstruktion entspricht ihrer Modellannahme und begünstigt sie bewusst.

### Kontrollierter Farbstress-Test

Bekannte digitale Seite mit acht kräftigen/pastelligen Farbflächen, Grautönen und feinen Zeichen. Synthetische räumliche Abdunklung, Farbstich und kleiner additiver Offset; keine realistische vollständige Druck-/Kamerasimulation. **Ein Diagnosefall, keine allgemeine Qualitätsrangliste.**

| Verfahren | Mittleres ΔE00 in acht festen Farbflächen ↓ | Helle Druckpixel fälschlich nahezu weiß ↓ |
| --- | ---: | ---: |
| Degradierte Eingabe | 18,07 | 0 % |
| Klassische Baseline | **2,28** | **0 %** |
| GC-Net 256, Feld auf Original | **4,50** | **1,21 %** |
| GCDRNet vollständig, Referenzauflösung | 9,48 | 48,59 % |
| DocRes, Referenzauflösung | 9,69 | 31,20 % |
| DocRes 256, Verhältnisfeld | 8,46 | 24,20 % |

„Helle Druckpixel“: alle Referenzkanäle ≥180, mindestens ein Kanal <245; „nahezu weiß“: alle Ausgabekanäle ≥250. Enthält helle Flächen und antialiaste Zeichenränder; es ist kein OCR-Verlustprozentsatz. Dunkle Referenzpixel wurden in diesem einfachen Test bei keinem Kandidaten zu hellen Pixeln nach dem verwendeten groben Schwellenmaß; das ersetzt keine Zeichen-/Notenerkennung.

Die Ergebnisse erklären, weshalb Papierweiß allein kein ausreichendes Optimierungsziel ist. Ein Modell kann hervorragend weißes Papier liefern und dabei beabsichtigte Pastellflächen entfernen.

## DescanDiffusion: Grenze des tatsächlichen Versuchs

Der Originalcode ist vorhanden. Der alte Gewichtslink im README ist **auskommentiert** und liefert HTTP 404; im [Autoren-Issue #2](https://github.com/jhcha08/Descanning/issues/2) werden vortrainierte Gewichte angekündigt. Über geprüfte öffentliche Modellverzeichnisse fand sich kein passender Checkpoint. Die frühere Forschungsnotiz wurde entsprechend korrigiert.

Die unveränderte UNet-Architektur hat **68.206.275 Parameter**, zusätzlich käme der ResNet-34-Farbencoder. Mit zufällig initialisierten Gewichten, drei warmen Wiederholungen auf Apple M3 Max/MPS, dauert ein Denoising-Schritt bei 256×256 im Median **272,6 ms**, bei 512×512 **652,5 ms**. Der veröffentlichte Sampling-Einstieg verwendet zehn Schritte. Dies spricht als Architekturindikator gegen den unveränderten Ansatz für ≤200 ms; **keine iPhone-/Android-Messung und kein Test der restaurierten Bildqualität**. Weder zehnfache Hochrechnung noch Desktopzahlen werden als gemessene mobile Gesamtlaufzeit ausgegeben.

[Architektur-Rohmessungen](../../outputs/research/color-restoration-20261110/results/latency-descan_arch-mps.json)

## Android und Reproduzierbarkeit

Kein echtes Android-Gerät angeschlossen; ein Emulator war offline. Deshalb keine behauptete Android-Latenz.

Vorbereitet und numerisch geprüft: ONNX FP32, feste 256er-Größe, Opset 17: GC-Net **6,08 MB**, GCDRNet **22,22 MB**, DocRes **61,40 MB**. Operatorlisten und Paritätswerte stehen bei den [Android-Artefakten](../../outputs/research/color-restoration-20261110/android-models/). Diese Exporte belegen weder NNAPI-Unterstützung noch tatsächliche NPU-/GPU-Platzierung. Auf einem physischen Android-Gerät müssen Backend, Datentransfers und kompletter Photometrieschritt geprüft werden.

GC-Net/Core-ML-FP16 benötigt etwa **2,9 MiB** kompilierten Modellinhalt, vollständiges GCDRNet etwa **11 MiB**, DocRes etwa **30 MiB**. Die deutlich größeren heruntergeladenen Trainingscheckpoints enthalten auch Optimizerzustand; ihre Downloadgröße ist nicht die notwendige App-Modellgröße.

GCDRNet-Gewichte stammen vom [öffentlichen Drittspiegel FahNos/GCDRnet](https://huggingface.co/FahNos/GCDRnet), weil der Autoren-OneDrive-Link 403/401 liefert. Striktes Laden in die Originalarchitektur und SHA-256 sind geprüft, Bytegleichheit mit dem unerreichbaren Autorenarchiv ist **nicht** bestätigt. DocRes stammt aus dem [vom Autorenrepository verlinkten Demo-Space](https://huggingface.co/spaces/qubvel-hf/documents-restoration/tree/main/checkpoints). Checkpoints werden mit `weights_only=True` gelesen. Alle genauen Quellstände, Prüfsummen und Wiederholungsschritte stehen im Workspace.

## Empfehlung

**Für den nächsten mobilen Prototyp GC-Net 256 als Beleuchtungsschätzer verwenden und die Originalpixel mit dem geschätzten RGB-Feld korrigieren.** Der iPhone-Versuch belegt dafür deutlichen Spielraum unter 200 ms. Vor Produktintegration: echte Android-Messung, mehr farbige/helle Dokumente und harte Schatten, Vergleich gegen eine native klassische Baseline sowie separate Prüfung kleinster Notenzeichen. Ein zusätzlicher globaler Sättigungsboost ist durch diese Versuche nicht gerechtfertigt.

DocRes eignet sich hier eher als Qualitätsvergleich für weitere Entwicklung. Für DescanDiffusion bleibt der trainierte Vergleich bis zu verfügbaren Gewichten offen. Die derzeitigen Befunde rechtfertigen keine Behauptung, beliebige ursprüngliche PDF-Farben rekonstruieren zu können.
