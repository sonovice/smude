# Photometrische Restaurierung geometrisch entzerrter Dokumentfotos

Stand und Rechercheabruf: **10. November 2026**. Gegenstand: bereits korrekt entzerrte RGB-Dokumentfotos; Papier soll weiß, Druck soll klar und farbig erscheinen. Geometriekorrektur, OCR und Binarisierung sind keine Hauptziele. **Befund** bezeichnet Aussagen aus den verlinkten Primärquellen oder dem gelesenen Originalcode. **Einschätzung / Vorschlag** bezeichnet eigene Ableitungen, keine gemessene Modellleistung. Es wurden keine Modelle ausgeführt und keine Laufzeiten gemessen.

## Entscheidung für Smude

**Empfehlung:** Zunächst **GCDRNet**, **DocRes ausschließlich mit `appearance`** und **FSENet mit SD7K-Gewichten** auf denselben entzerrten RGB-Bildern gegen eine konservative klassische Hintergrunddivision vergleichen. GCDRNet und DocRes passen unmittelbar zum gewünschten Erscheinungsbild; FSENet liefert den wichtigen Kontrollversuch, was allein Schattenentfernung erreicht. Diese Priorisierung bewertet Aufgabenpassung und Reproduzierbarkeit, nicht einen behaupteten universellen Qualitätsrang.

Die wichtigsten Befunde:

- **Weißes Papier ist erreichbar, exakte digitale Originalfarbe im Allgemeinen nicht eindeutig rekonstruierbar.** Weißnormierung, Beleuchtungsbereinigung und Rückgängigmachen der Druck-/Kamerakette sind unterschiedliche Probleme. Ein sauberer Weißpunkt kalibriert noch keine bunten Druckfarben. [Grundlage zur Mehrdeutigkeit von Color Constancy](https://dash.harvard.edu/entities/publication/73120378-ecfe-6bd4-e053-0100007fdf3b), [kameraspezifische Farbkorrektur](https://research-portal.uea.ac.uk/en/publications/colour-correction-using-root-polynomial-regression/).
- **Die Zielbilder entscheiden, was ein Modell tatsächlich lernt.** RealDAE enthält manuell verbesserte Fotos; SD7K schattenfreie Fotos desselben Drucks. Beide sind etwas anderes als eine digitale Druckvorlage. [RealDAE der Autoren](https://github.com/ZZZHANG-jx/GCDRNet), [SD7K, Abschnitt 3](https://arxiv.org/html/2308.14221v3).
- **Descanning ist die am nächsten liegende Forschung zur eigentlichen Originalrekonstruktion.** DescanDiffusion verwendet echte Scan-/Digitaloriginal-Paare. Die Übertragbarkeit auf Handyfotos bleibt eine zusätzliche, hier nicht belegte Hypothese. [AAAI-2024-Arbeit](https://arxiv.org/html/2402.05350v1).
- **Neuere Arbeiten sind nicht automatisch praktisch überlegen.** Für GL-PGENet fehlen veröffentlichte Gewichte; das geprüfte MMDIR-Repository enthält nur eine README. Bei ImpResDescan ist eine ICLR-2026-Annahme anhand der zugänglichen Primärquelle nicht bestätigt. Details unten.
- **Keiner der geprüften Nachweise garantiert gleichzeitig den Erhalt beliebig heller Tinte, fluoreszierender Textmarker, eingebetteter Fotos und kleinster Noten-/Textzeichen.** Das ist eine Grenze der gefundenen Evidenz, keine Behauptung, dass alle Modelle daran scheitern.

## 1. Was genau soll wiederhergestellt werden?

Die Literatur verwendet „illumination correction“ und „appearance enhancement“ teilweise überlappend. Für die Produktentscheidung ist folgende operative Trennung sinnvoll; insbesondere DocRes unterscheidet Schattenentfernung ausdrücklich vom umfassenderen Appearance-Ziel. [DocRes, Abschnitte 2–3](https://arxiv.org/html/2405.04408v1).

| Aufgabe | Ziel | Was dadurch noch nicht bestimmt ist |
| --- | --- | --- |
| Beleuchtungsnormalisierung | Räumlich veränderliche Helligkeit bzw. Beleuchtungsfarbe ausgleichen | Originale Drucksättigung, Druckprofil, Kamerafarbmatrix |
| Schattenentfernung | Aufnahme ohne den abschattenden Gegenstand approximieren; harte und weiche Schatten entfernen | Ob das schattenfreie Papier weiß und die Aufnahme farbmetrisch korrekt ist |
| Weißabgleich / Color Constancy | Farbstich der Beleuchtung kompensieren, Erscheinung unter einer Referenzbeleuchtung approximieren | Nichtlineare Druck-/Kameraverarbeitung und verlorene Farbinformation |
| Reflectance-/Albedo-Recovery | Beleuchtung und Oberflächeneigenschaften trennen; materialbezogene Reflexion schätzen | RGB-Albedo ist weder spektrale Reflexion noch die ursprüngliche RGB-/CMYK-Datei |
| Document Appearance Enhancement | Ein sauberes, scan- oder PDF-ähnliches Bild entsprechend Trainingsziel erzeugen | Ob es dem unbekannten historischen Original entspricht oder nur plausibel aussieht |
| Descanning / digitale Originalrekonstruktion | Druck-/Scanartefakte und Farbverschiebungen in Richtung bekannter digitaler Vorlagen rückgängig machen | Eindeutige Inversion beliebiger unbekannter Druck- und Kameraketten |

**Wichtige Zielentscheidung:** „Unbedrucktes weißes Papier nach `#FFFFFF`“ ist eine sinnvolle Darstellungsnorm. „Alle hellen Bereiche nach `#FFFFFF`“ zerstört unter Umständen Pastellflächen, Wasserzeichen, gelbe Schrift, Fotohighlights und dünne graue Linien. Auch cremefarbenes oder farbiges Papier muss nicht zwingend weiß werden. Für Smude empfiehlt sich eine explizite Annahme „weißes Papier“, gekoppelt an eine konservative Papiermaske.

### Weißnormalisierung ist kein Saturationsboost

Im vereinfachten linearen Modell `I_c(x) ≈ R_c(x) · L_c(x)` kann man die beobachtete Farbe durch ein geschätztes Beleuchtungsfeld teilen. Eine gemeinsame positive Verstärkung aller RGB-Kanäle erhält, solange nicht geclippt wird, ihre Verhältnisse. Kanalabhängige Verstärkung korrigiert zusätzlich einen Weißpunkt. Das beschreibt Belichtungs-/Beleuchtungskorrektur, nicht die Wiederherstellung beliebiger Originalfarben. Eine physikalisch motivierte dokumentbezogene Zerlegung untersucht beispielsweise [DocIIW](https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0906.pdf).

Eine Erhöhung der HSV-Sättigung oder der Lab-Chroma macht Farben kräftiger, ohne zu wissen, ob die Vorlage kräftig war. Sie kann Hauttöne, Fotos und absichtlich blasse Diagrammfarben verfälschen. Ein Schwarz-/Weißpunkt-Stretch kann ebenfalls den Farbeindruck verändern. **Eigene Folgerung:** Beleuchtung zuerst schätzen; eine zusätzliche Farbstilkorrektur getrennt bewerten und benennen. GL-PGENet lernt beispielsweise Helligkeits-, Kontrast- und Sättigungsparameter aus Daten – dies ist eine geschätzte Enhancement-Transformation, kein Beweis für Originaltreue. [GL-PGENet, Abschnitt 3](https://arxiv.org/html/2505.22021v1).

## 2. Identifizierbarkeitsgrenzen

Eine realistischere Beobachtungskette lautet schematisch:

`Digitalvorlage → Farbmanagement/Druck/Raster/Papier → spektrale Reflexion unter Licht → Kameraantwort/ISP → JPEG bzw. HEIC`.

Die Kamerakanäle integrieren Beleuchtungsspektrum, Materialreflexion und spektrale Sensorempfindlichkeit. Ohne Referenz können mehrere Kombinationen zum gleichen RGB führen. Bereits die gemeinsame Unbekanntheit von Licht und Reflexion macht Color Constancy mehrdeutig. Hinzu kommt eine im Allgemeinen nichtlineare Kameraantwort; ein JPEG lässt sich deshalb nicht allein durch Division sicher in physikalische Reflexion umwandeln. [Chakrabarti, Hirakawa, Zickler, 2010](https://dash.harvard.edu/entities/publication/73120378-ecfe-6bd4-e053-0100007fdf3b), [Grossberg und Nayar, TPAMI 2004](https://www.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_PAMI04.pdf).

**Eigene Ableitung aus dieser Kette:**

- Ein neutrales Papierweiß liefert einen wertvollen Anker, aber nicht die vollständige Farbabbildung für Cyan, Magenta, Gelb, Hauttöne usw.
- Schon im Produktmodell lässt sich eine positive Skalierung zwischen Reflexion und Beleuchtung verschieben. Eine Weißannahme reduziert diese Freiheit, löst aber keine unbekannte Druckgamut-Kompression oder Kanalvermischung.
- Überbelichtete Kanäle, in Rauschen verlorene Schatten und beim Druck nicht unterscheidbare Vorlagenfarben enthalten keine eindeutig invertierbare Information. Ein Netz kann dann nur aus gelernten Beispielen schätzen.
- Fluoreszierende Textmarker sowie glänzende Drucke verletzen einfache rein diffuse Reflexionsannahmen. Ihre Erhaltung braucht eigene Testfälle; „colored shadow“ ist kein Nachweis für Marker-Farbtreue.
- Selbst eine perfekte Rekonstruktion des heutigen bedruckten Blatts wäre nicht zwangsläufig die Farbe der damaligen digitalen Vorlage.

Mit einer bekannten Farbtafel und kontrollierter Aufnahme lassen sich Kamerafarben auf einen Referenzfarbraum kalibrieren; Root-Polynomial Color Correction ist dafür eine konkrete, relativ einfache Methode. Eine solche Kamerakalibrierung rekonstruiert jedoch nicht automatisch die unbekannte Druckvorstufe. [Finlayson, Mackiewicz, Hurlbert, IEEE TIP, Mai 2015](https://research-portal.uea.ac.uk/en/publications/colour-correction-using-root-polynomial-regression/).

**Produktformulierung:** „Papierweiß und Druckfarben restaurieren“ ist als Schätzziel vertretbar. „Die originalen PDF-Farben wiederherstellen“ wäre ohne Referenz und passende Validierung zu stark.

## 3. Priorisierte Methoden-Shortlist

### 3.1 GCDRNet – erster Appearance-Kandidat

**Publikation:** Zhang et al., *Appearance Enhancement for Camera-Captured Document Images in the Wild*, IEEE Transactions on Artificial Intelligence **5(5), Mai 2024, S. 2319–2330**. Das Autorenrepository bezeichnet die Arbeit als TAI 2023; das Heftdatum 2024 ist in den vom Verlag hinterlegten DOI-Metadaten bestätigt. [DOI](https://doi.org/10.1109/TAI.2023.3321257), [Metadaten](https://api.crossref.org/works/10.1109/TAI.2023.3321257).

**Befund:** Der Originalcode schätzt zunächst ein RGB-Feld mit GC-Net, dividiert das Eingangsbild dadurch und führt anschließend Original und korrigiertes Bild dem DR-Net zu. Die zweite Stufe erzeugt wieder Bildpixel. Der veröffentlichte Inferenzpfad nutzt CUDA und gepaddete volle Auflösung; eine vorherige `512×512`-Skalierung wird im Code unmittelbar durch das Originalbild überschrieben. [Original-Inferenzcode](https://github.com/ZZZHANG-jx/GCDRNet/blob/415c97a9a64d0a796dbbad43485ce2ab102f2a7d/infer.py).

**Daten/Ziel:** RealDAE umfasst 600 reale Fotos mit manuell verbesserten, pixelweise ausgerichteten Zielen; der dokumentierte Benchmark-Split ist 450 Training / 150 Test. Die genaue zusätzliche Vortrainingsmischung des veröffentlichten GCDRNet-Checkpoints ließ sich aus dem zugänglichen Originalrepository nicht vollständig verifizieren; es veröffentlicht Inferenzcode, keinen vollständigen Trainingsworkflow. [Autorenrepository](https://github.com/ZZZHANG-jx/GCDRNet), [RealDAE-Split im DocRes-Paper](https://arxiv.org/html/2405.04408v1).

**Einschätzung:** Gute erste Probe für „wie frisch digitalisiert“, einschließlich gemischter Layouts. GC-Net allein ist als zusätzliche Ablation interessant: Es trennt den Gewinn durch Division von Veränderungen durch das Detailnetz. Weder eine harte Weißgarantie noch Schadensfreiheit für feinste Zeichen folgt aus der Architektur. Vollauflösende zweite Stufe kann bei großen Seiten Speicher und Zeit beanspruchen; konkrete mobile Werte müssen gemessen werden.

### 3.2 DocRes – starker, klar auswählbarer Appearance-Vergleich

**Publikation:** Zhang et al., **CVPR 2024**, arXiv-Erstfassung **7. Mai 2024**. Restormer-basierter Generalist mit bildabhängigen Task-Prompts. Der Appearance-Trainingszweig kombiniert synthetische Beleuchtungsdegradation aus Doc3DShade mit RealDAE; die Datenanleitung beschreibt saubere PDF-Raster als Syntheseziele. Der Deshadowing-Zweig verwendet RDD und FSDSRD. [Paper](https://arxiv.org/html/2405.04408v1), [originale Datenanleitung](https://github.com/ZZZHANG-jx/DocRes/blob/master/data/README.md).

**Wichtiger Implementierungsbefund:** `appearance()` behandelt Bilder ab einer längsten Seite von 1600 Pixeln anders: Das Netz erhält eine auf `1600×1600` skalierte Fassung. Anschließend wird aus Eingabe und Vorhersage ein RGB-Verhältnisfeld berechnet, hochskaliert und auf das hochaufgelöste Original angewendet. Kleinere Bilder erhalten direkt die Netzvorhersage. [Gelesener Originalcode](https://github.com/ZZZHANG-jx/DocRes/blob/d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560/inference.py).

**Einschätzung:** Für Smude besonders interessant, weil der große Bildpfad die vorhandenen Originaldetails als Träger nutzt. Das ist dennoch kein Beweis für deren Erhalt: fehlerhafte Gain-Felder, Quantisierung und Clipping können dünne/helle Zeichen beschädigen. Der Auflösungswechsel ist im Test gesondert zu kontrollieren. Ausschließlich `--task appearance` einsetzen; `deshadowing` separat als Ablation. `end2end` enthält erneutes Dewarping und temporäre JPEGs und passt nicht zu bereits fertiger Geometrie. [Task-Verkettung im selben Originalcode](https://github.com/ZZZHANG-jx/DocRes/blob/d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560/inference.py).

### 3.3 FSENet / DocShadow-SD7K – hochauflösender Schattenkontrollversuch

**Publikation:** Li, Chen, Pun, Cun, *High-Resolution Document Shadow Removal via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net*, **ICCV, Oktober 2023**; arXiv v3 vom **1. September 2023**. **FSENet ist das Netz, SD7K der Datensatz, DocShadow-SD7K der Projekt-/Repositoryname**; „DocShadow“ in Vergleichstabellen ist hier kein zusätzliches unabhängiges Verfahren. [Paper](https://arxiv.org/html/2308.14221v3), [Autorenprojekt](https://cxh-research.github.io/DocShadow-SD7K/).

**Daten/Ziel:** SD7K bietet über 7000 hochauflösende reale Paare. Eine fixierte Canon-Kamera fotografiert den Druck mit und ohne abschattenden Gegenstand bei festgehaltenen Einstellungen. Ziel ist also das **schattenfreie Kamerabild**, nicht die PDF. Das Modell zerlegt niedrige und hohe Frequenzen und restauriert auch Detailbänder. Die Autoren nennen Rechenaufwand und fehlende damalige Eignung für Echtzeit-/Mobilgeräte als Grenze. [Datenerhebung und Limitations](https://arxiv.org/html/2308.14221v3).

**Reproduktionshinweis:** Das Repository ist seit **30. März 2026 archiviert**. Sein README erklärt ausdrücklich, dass der veröffentlichte Stand gegenüber dem Paper GDFN verwendet. Daher „Release-FSENet/SD7K“ protokollieren und nicht ungeprüft mit Paper-Zahlen gleichsetzen. Die Release-Seite enthält separate SD7K-, Jung- und Kligler-Gewichte. [Repository](https://github.com/CXH-Research/DocShadow-SD7K), [Gewichte](https://github.com/CXH-Research/DocShadow-SD7K/releases/tag/Weights).

**Einschätzung:** Sinnvoll als dritter Kandidat, wenn harte Schatten relevant sind. Restfarbstich oder nichtweißes Papier können trotz erfolgreicher Schattenentfernung übrig bleiben, weil das Trainingsziel sie nicht grundsätzlich ausschließt.

### 3.4 DocIIW – physikalisch relevante historische Referenz

**Publikation:** Das et al., *Intrinsic Decomposition of Document Images In-the-Wild*, **BMVC 2020**. Trennt Weißabgleich und Shading-/Materialkomponenten mit physikalischen Randbedingungen. Doc3DShade kombiniert synthetische Dokumenttexturen mit realistischer Papierabschattung; das Repository nennt 90.000 Bilder, davon 80.000 Training und 10.000 Validierung, sowie WBNet-/SMTNet-Gewichte. [Paper](https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0906.pdf), [Originalrepository, MIT](https://github.com/cvlab-stonybrook/DocIIW).

**Einschätzung:** Besonders wichtig als begrifflich passender Albedo-/Weißabgleich-Ansatz und Datenquelle. Kein Beleg einer Inversion realer Druckfarben. Gegenüber den ersten drei Kandidaten älterer Workflow und zusätzliche Reproduktionsarbeit; als gezielte Ablation bei Mischlicht interessant. OCR-Gewinne in dieser Arbeit sind kein Maß für Farbtreue.

### 3.5 DocTr-IllTr – verifiziert, aber nachrangig

**Publikation:** Feng et al., *DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction*, **ACM MM 2021**. Relevant ist ausschließlich **IllTr**. Laut Autoren wurde es mit DocProj trainiert; das ist ein synthetischer Rendering-Kontext und keine reale Druck-zu-Digital-Kalibrierung. [DocTr](https://github.com/fh2019ustc/DocTr), [DocProj, SIGGRAPH Asia 2019](https://github.com/xiaoyu258/DocProj).

**Codebefund:** IllTr verarbeitet `128×128`-Patches mit 12,5 % Überlappung in verschachtelten Schleifen. Die Zusammensetzung überschreibt Überlappungsbereiche; außerdem setzt sie die erste Bildzeile weiß. Ein implementierter Kontrastfaktor 2 ist im aufgerufenen Pfad auskommentiert. [Originalcode](https://github.com/fh2019ustc/DocTr/blob/044648919b6b28f2cb1b288db6d8fd97749db537/inference_ill.py).

**Einschätzung:** Historisch nützlicher Vergleich, jedoch Risiken durch lokale Kontextbegrenzung, Patchübergänge und Detailrekonstruktion. Die aktuelle eigene Lizenz erlaubt nichtkommerzielle Nutzung mit Bedingungen, kommerzielle Verwendung nur nach schriftlicher Erlaubnis; deshalb kein unkritischer Produktstandard. [Lizenz](https://github.com/fh2019ustc/DocTr/blob/main/LICENSE.md).

### 3.6 BGShadowNet und BEDSR-Net – warum nicht zuerst?

**BGShadowNet**, Zhang et al., **CVPR, Juni 2023**, schätzt farbbewusste Hintergründe; das ist relevanter für farbige Dokumentbereiche als die Annahme eines einzigen Papierfarbwerts. Originalcode und auf RDD trainierte Gewichte sind verlinkt. Im geprüften Repository wurde keine Lizenzdatei gefunden. [Paper der Autoren](https://labzhengliu.github.io/papers/%28CVPR23%29Document%20Image%20Shadow%20Removal%20Guided%20by%20Color-Aware%20Background.pdf), [Repository](https://github.com/hyyh1314/BGShadowNet). **Einschätzung:** Reservekandidat bei dominierenden farbigen Hintergründen; Schattenziel und ungeklärte Lizenz reduzieren die unmittelbare Eignung gegenüber dem Dreiervergleich.

**BEDSR-Net**, Lin, Chen, Chuang, **CVPR, Juni 2020**, nutzt einen geschätzten globalen Hintergrundfarbwert und eine Attention-Map; Training auf synthetischen Schatten-/schattenfreien Paaren. [Originalpaper und Autoren](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_BEDSR-Net_A_Deep_Shadow_Removal_Network_From_a_Single_Document_CVPR_2020_paper.html). **Einschätzung:** Starke historische Aufgabenbaseline, aber kein Beleg digitaler Farbtreue. Ein verifiziertes offizielles Autorenrepository mit vollständigem Code-/Gewichtspaket wurde nicht gefunden; häufig verlinkte Implementierungen bezeichnen sich ausdrücklich als inoffiziell. Für den ersten reproduzierbaren Versuch daher FSENet vorziehen.

## 4. Was 2024–2026 zusätzlich bringt

### Echte digitale Referenzen: DescanDiffusion

**Cha et al., AAAI 2024; arXiv 8. Februar 2024.** Ein Farbencoder schätzt kanalweise Farbstatistiken; eine konditionierte Diffusion restauriert lokale Bildanteile. DESCAN-18K umfasst **18.360 reale Scan-/Original-Paare**, aus Zeitschriften und deren digitalen PDFs, als registrierte `1024×1024`-Ausschnitte. Der Split ist 17.640/360/360; Testscanner unterscheiden sich von Trainingsscannern. Die Scans werden laut Paper nach IT 8.7 kalibriert. Die Plus-Variante ergänzt synthetische Degradationen. [Paper, Dataset und Verfahren](https://arxiv.org/html/2402.05350v1).

Code ist im Originalrepository verfügbar; ein Update vom **13. April 2026** meldet die Datenfreigabe. **Korrektur nach dem praktischen Downloadversuch vom 10. November 2026:** Der früher erwähnte Modelllink steht in einem auskommentierten README-Abschnitt und liefert HTTP 404. Ein Autorenkommentar in Issue #2 kündigt die Gewichte erst an. Ein nutzbarer vortrainierter Checkpoint wurde nicht gefunden; die ursprüngliche Formulierung „vortrainiertes Modell verlinkt“ war als Verfügbarkeitsaussage zu stark. Eine klare allgemeine Code-/Gewichtslizenz wurde im geprüften Baum nicht gefunden. Die zugrunde liegenden Magazine werden mit **CC BY-NC-SA 3.0** angegeben; dies nicht mit einer pauschalen permissiven Modelllizenz verwechseln. [Repository](https://github.com/jhcha08/Descanning), [Gewichte-Issue](https://github.com/jhcha08/Descanning/issues/2), [Datensatzkarte](https://huggingface.co/datasets/ENCLab/DESCAN-18K).

**Einschätzung:** Beste konzeptionelle Referenz für „zurück zur digitalen Vorlage“, einschließlich Fotos und Druckraster. Kein erster Smude-Standard: Scan-/Telefon-Domänenwechsel, iterative Inferenz und mögliche Detailveränderungen sind gesondert zu prüfen. Nach dem Dreiervergleich als Forschungszweig testen, falls bekannte digitale Vorlagen zur Verfügung stehen.

### Neuere Appearance- und Schattenmethoden

| Arbeit / verifizierter Status | Belegter Ansatz und Referenzziel | Bewertung für diese Aufgabe |
| --- | --- | --- |
| **GL-PGENet**, Tang; arXiv **28.05.2025**, keine weitere Venue hier bestätigt | Globale Helligkeits-/Kontrast-/Sättigungsparameter plus lokale lineare Parameter und Glättungszweig; über 500.000 synthetische Vortrainingsbeispiele, anschließendes Fine-Tuning; Evaluation auf RealDAE und ausgerichtetem DocUNet. [Paper](https://arxiv.org/html/2505.22021v1) | Sehr relevante Architekturrichtung für hochauflösende Farbdokumente. Keine Garantie gegen Glättung. Originalrepo: Gewichte explizit TODO, `inference.py` beim Abruf leer, MIT-Badge ohne vorhandene LICENSE-Datei. Derzeit kein gleichwertig reproduzierbarer Hauptkandidat. [Repo](https://github.com/kukugpt/GL-PGENet) |
| **Uni-DocDiff**, Zhao et al., **ACM MM, 27.–31.10.2025**; arXiv **06.08.2025** | Diffusion mit lernbaren Task-Prompts und Frequenz-Priors. Beleuchtungszweig: synthetische saubere Dokumente mit DocShade-Beleuchtung und 450 RealDAE-Bilder; Ziele bleiben synthetisch bzw. manuell verbessert. [Paper mit ACM-DOI](https://arxiv.org/html/2508.04055v1) | Autoren berichten Fortschritte auf Appearance-Benchmarks. Kein unabhängig verifizierter Nachweis für Marker-/Originalfarbtreue. Öffentliches vollständiges Autorenpaket mit Gewichten und Softwarelizenz nicht gefunden; iterative Wiederherstellung erst nach Verfügbarkeitsprüfung benchmarken. |
| **DocShaDiffusion**, Liu et al.; arXiv **02.07.2025**, weitere Venue nicht bestätigt | Latente, maskengesteuerte Diffusion; explizit farbige Schatten. SDCSRD synthetisiert Schatten über schattenfreien Ausgangsbildern; 17.624 Beispiele. [Paper](https://arxiv.org/html/2507.01422v1) | Schließt eine relevante Lücke bei farbigem Licht/Schatten, aber kein Nachweis originaler Druckfarben. Verifiziertes öffentliches Code-/Gewichtspaket und Lizenz nicht gefunden. |
| **MMDIR**, Li et al., **CVPR, Juni 2026** | Multimodale Instruktionen für Mischdegradationen: Unschärfe, Schatten, Textwasserzeichen, Stempel. MixedDoc nutzt synthetische Kombinationen auf Dokumentseiten. [CVF-Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_MMDIR_Multimodal_Instruction-Driven_Framework_for_Mixed-Degradation_Document_Image_Restoration_CVPR_2026_paper.pdf) | Kein auf Farbmetrik ausgerichteter Nachfolger für diese Frage. Automatisches Entfernen von Stempeln/Markierungen wäre teils ein falsches Produktziel. Geprüfter Repo-Baum enthält **nur README**, Daten/Ergebnisse verlinkt, keine Modellimplementierung/Gewichte; dort **CC BY-NC-ND 4.0**. [Repo](https://github.com/xiaomore/MMDIR) |
| **ImpResDescan**, Einreichung zu **ICLR 2026** | Implizite nichtlineare Farbkorrektur plus lokale Diffusionsverfeinerung; DESCAN-18K und weitere Scan-Domänen. [Zugängliches Manuskript](https://openreview.net/pdf?id=ACkxqOJJyj) | Thematisch sehr passend zur Originalrekonstruktion. PDF trägt „Under review“; Forumsseite war zugangsgeschützt. Annahme/Publikationsstatus, Code, Gewichte und deren Lizenz **nicht bestätigt**. Keine Gleichsetzung mit einer angenommenen ICLR-Arbeit. |
| **CrossMamba**, Liu et al., *The Visual Computer*, veröffentlicht **27.07.2026** | Laut Verlagsabstract Laplace-Pyramide, globales Farbwörterbuch, lokale Farbkorrektur und Mamba-basierte Texturbehandlung. [Verlagsquelle](https://link.springer.com/article/10.1007/s00371-026-04658-w) | Existenz/Datum verifiziert; Volltext hinter Zugangsschranke. Trainingsmischung, Referenzfarbziel, Code/Gewichte/Lizenz und belastbare Laufzeitdetails nicht ausreichend geprüft. Deshalb keine Qualitätsrangfolge daraus ableiten. |

**Zusätzlicher 2026-Treffer:** *Plug and play document image shadow removal with conditional diffusion model* (**DocShaDif**, *Information Sciences* 741, **15. Juni 2026**, 123266) beschreibt Hintergrundfarbkonditionierung und gradientenregularisiertes Sampling. Nicht mit DocShaDiffusion verwechseln. Der zugängliche Verlagsabstract reicht zur Methodenidentifikation, nicht zur Reproduktions- oder Farbtreuebewertung; Code/Gewichte hier nicht verifiziert. [Original-Verlagsseite](https://www.sciencedirect.com/science/article/abs/pii/S0020025526001970).

**Wichtige ältere Gegenprobe:** *DocStormer: Revitalizing Multi-Degraded Colored Document Images to Pristine PDF*, arXiv **27. Oktober 2023**, klingt exakt nach dem Nutzerziel. Tatsächlich beruhen MD-CDE-Ziele auf automatisch und manuell verbesserten Fotos; synthetische Schatten, Falten und Durchscheinen werden ergänzt. PDFs dienen zusätzlich zur Anpassung der Ausgabeverteilung per GAN. Das ist keine durchgängig gepaarte Wiederherstellung der jeweiligen originalen PDF. PFILI überträgt niedrig aufgelöste Korrekturfaktoren auf das Original. Kein vollständiges öffentliches Autorenpaket mit Gewichten/Lizenz wurde gefunden. **Einschätzung:** Relevante Ideenquelle, aber weder sicherer Originalfarbbeweis noch sofortiger Vergleichskandidat. [Paper, insbesondere Abschnitte 3.4–4](https://arxiv.org/html/2310.17910v1).

## 5. Code, Gewichte und Einsatzimplikationen

„Verlinkte Gewichte“ bedeutet hier: Der Autorenverweis wurde gesehen; große Cloud-Downloads und Modellinitialisierung wurden **nicht** durchgeführt. Repositorylizenzen sind getrennt von unklaren Bedingungen externer Gewichts-/Datenarchive zu lesen.

| Kandidat | Code / Gewichte | Verifizierter Lizenzbefund | Praktische Folge |
| --- | --- | --- | --- |
| GCDRNet | Inferenzcode; GC-/DR-Checkpoints über Autoren-OneDrive verlinkt | [MIT, Copyright 2024 Jiaxin Zhang](https://github.com/ZZZHANG-jx/GCDRNet/blob/main/LICENSE.txt); externe Archive nicht vollständig auditiert | CUDA-PyTorch-Ausgangspunkt; Export und native Ausführung separat prüfen |
| DocRes | Training/Inferenz; `docres.pkl` verlinkt | [MIT](https://github.com/ZZZHANG-jx/DocRes/blob/master/LICENSE); zusätzliche Datenrechte separat | Photometrischen Task isolieren; aktuelle Inferenz nutzt FP16, daher CPU-/ONNX-Parität nicht voraussetzen |
| FSENet | Training/Inferenz; [SD7K/Jung/Kligler-Release](https://github.com/CXH-Research/DocShadow-SD7K/releases/tag/Weights) tatsächlich vorhanden | [MIT](https://github.com/CXH-Research/DocShadow-SD7K/blob/main/LICENSE) | Archivierter Upstream; Release-Modell und Paperarchitektur unterscheiden |
| DocIIW | Trainings-/Testcode und verlinkte WBNet-/SMTNet-Modelle | [MIT-Datei](https://github.com/cvlab-stonybrook/DocIIW/blob/master/LICENSE.txt) | Ältere Umgebung, unvollständige Trainingsanleitung; als Ablation einsetzen |
| DocTr IllTr | Separater Illumination-Code, Modelle verlinkt | [Eigene nichtkommerzielle Lizenz](https://github.com/fh2019ustc/DocTr/blob/main/LICENSE.md) | Patchpfad und Nutzungsbedingungen sind zusätzliche Hürden |
| BGShadowNet | Training/Test, RDD-Modell verlinkt | Keine Lizenzdatei im untersuchten Originalbaum gefunden | Nicht automatisch als permissiv verfügbar behandeln |
| DescanDiffusion | Code verfügbar; alter auskommentierter Gewichtslink liefert 404, kein nutzbarer Checkpoint gefunden | Keine klare allgemeine Software-/Gewichtslizenz gefunden; Magazinbasis nichtkommerziell lizenziert | Vortrainierter Qualitätsversuch derzeit blockiert; Architekturprofiling allein bewertet keine Restaurierung |

**Laufzeitbewertung ohne erfundene Benchmarks:** Die im Paper oder in Portierungen gemessenen GPU-Zeiten sind keine Messungen auf Smudes iOS-/Android-/Desktop-Zielgeräten. Vollauflösende Merkmalsnetze brauchen Speicher proportional zu ihren Zwischenrepräsentationen; Patchverfahren erhöhen die Zahl der Aufrufe und können Übergänge erzeugen. Niedrig aufgelöste Feldschätzung mit Anwendung am Original ist für Smude besonders plausibel. Diffusionsverfahren benötigen wiederholte Netzaufrufe, deren Zahl und Sampler Teil jeder Laufzeitangabe sein müssen. Ein ONNX-Export allein beweist weder Operatorabdeckung noch NPU-/GPU-Ausführung.

## 6. Farbige Inhalte und feinste Details: Was ist wirklich belegt?

Die folgende Tabelle ist eine **Risikobewertung und Testplanung**, keine gemessene Fehlerliste der Kandidaten. Architektur oder farbige Beispielbilder reichen nicht als Nachweis über alle Inhaltsklassen.

| Inhalt | Typische Mehrdeutigkeit / Gefahr | Erforderliche Gegenprobe |
| --- | --- | --- |
| Große farbige Flächen | Können als Beleuchtungsfeld oder farbiges Papier interpretiert werden; lokale Weißschätzung findet im Inneren kein Papier | Gleicher Farbton als kleine Schrift, große Fläche und Balkendiagramm; innerhalb/außerhalb Schatten vergleichen |
| Diagramme | Farbtonverschiebungen können Klassen verwechseln; dünne Pastellachsen verschwinden | Getrennte Farbpatch- und Linienmessung, inklusive Legenden |
| Textmarker | Absichtlich helle/fluoreszierende Farbe kann als Farbstich, Schatten oder störende Markierung behandelt werden | Reale Marker über schwarzem Text unter mehreren Lichtarten; Farbe und darunterliegende Zeichen getrennt bewerten |
| Eingebettete Fotos | Papierannahmen und aggressive Kontrastkurven verfälschen Hauttöne, Schatten und Highlights | Fotomasken; Neutralität, Tonwertverlauf und Farbfehler innerhalb der Fotos messen |
| Helle Druckfarben | Überlappen mit Papierweiß, Rauschen und Papierstruktur | Pastell- und Graurampen, gelbe Kleinschrift, schwache Gitter; nicht nur kräftige Farben prüfen |
| Kleinster Text / Notation | Neuzeichnen, Glättung, Clipping, Halos oder Kachelgrenzen verändern Zeichentopologie | Punkte, Serifen, Vorzeichen, Hilfslinien, Notenhälse und offene Punzen bei nativer Auflösung vergleichen |
| Glanz / überbelichtete Bereiche | Im Eingang fehlen gegebenenfalls tatsächlich Informationen | Clippingmaske aus Eingang erfassen; verlorene Stellen nicht als verlässlich rekonstruiert bewerten |

**Befundgrenze:** FSENet behandelt hochauflösende Text-/Figurdetails explizit, DocIIW materialbezogene Beleuchtung, und Descanning enthält Fotos und Text. Daraus folgt keine Garantie für die oben genannten Spezialfälle. [FSENet](https://arxiv.org/html/2308.14221v3), [DocIIW](https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0906.pdf), [DESCAN-Datensatzkarte](https://huggingface.co/datasets/ENCLab/DESCAN-18K).

## 7. Ein einfacher klassischer Vergleich

**Vorschlag: maskierte, robuste RGB-Hintergrunddivision mit separater Weißdarstellung.** Das Grundmotiv ist etabliert: Bako et al., *Removing Shadows from Images of Documents*, **ACCV 2016**, schätzen lokale Text-/Hintergrundfarben, bilden daraus relativ zu einer globalen Referenz eine Schattenkarte und korrigieren per Gain. Autorenpaper und Quellcode sind auf der Projektseite verlinkt. Eine Lizenzprüfung des alten Codearchivs wurde hier nicht durchgeführt. [Autorenprojekt](https://civc.ucsb.edu/graphics/Papers/ACCV2016_DocShadow/), [Originalpaper](https://web.ece.ucsb.edu/~psen/Papers/ACCV16_RemovingShadows.pdf).

Der vorgeschlagene Smude-Vergleich ist eine einfache eigene Baseline, keine behauptete exakte Bako-Reimplementierung:

1. Eingabe in einen definierten Farbraum überführen; möglichst lineares RGB verwenden. Bei fertig verarbeitetem JPEG/HEIC ist inverse sRGB-Transferfunktion nur eine Näherung an die physikalische Kameraantwort. Zusätzlich eine dokumentierte sRGB-Variante als Ablation zulassen.
2. Auf einem verkleinerten Analysebild Papierkandidaten ermitteln: lokal helle, strukturarme Bereiche, Farbkonsistenz zum Papieranker; Text, Fotos, zusammenhängende Farbflächen und unsichere Stellen ausschließen. Helligkeit allein genügt nicht.
3. Aus robusten hellen Quantilen ein langsam veränderliches positives RGB-Feld `B_c(x)` schätzen. Fehlende Bereiche durch glatte, robust regularisierte Interpolation überbrücken. Morphologisches Closing ist eine einfache Startvariante, versagt aber unter großen bedruckten Flächen und sehr breiten Schatten ohne lokale Papierbeobachtung.
4. Am Originalraster anwenden: `J_c(x) = I_c(x) / max(B_c(x), ε)`. Weißbezug auf 1 normieren; extreme Verstärkung begrenzen und geringe Schätzsicherheit dokumentieren. Nicht das Ergebnisbild weichzeichnen.
5. Erst danach eine gesonderte Darstellungsstufe: **nur sehr sichere Papierpixel** exakt weiß setzen. Unsichere helle Inhalte bewahren. Ein globaler Sättigungsboost gehört nicht in diese Baseline.

**Erwartung, keine Messung:** Sehr günstig und nachvollziehbar bei weichem Lichtverlauf und ausreichend sichtbarem Papier; weniger robust bei scharfen Schattenkanten, Mischlicht, vollflächiger Grafik und Glanz. Der große Vorteil als Kontrollversuch ist, dass die Methode keine neuen Zeichenformen erzeugt. Trotzdem können ihre Farbtransformation und ihr Clipping Inhalte abschwächen oder löschen.

## 8. Evaluationsprotokoll, das die eigentliche Frage beantwortet

### 8.1 Drei Referenzarten auseinanderhalten

**Eigener Vorschlag:** Ein kleiner Pilot mit etwa 30 unterschiedlichen Seiten, mehreren Kameras und Lichtbedingungen reicht zur Fehlersuche; für eine belastbare Auswahl anschließend mehr Inhalte und geräte-/dokumentgetrennte Tests einplanen. Keine Fotos derselben Seite zwischen Tuning und Test verteilen.

- **Digitaltreue-Set:** Eigene PDF-/SVG-Vorlagen mit bekannten Farben, Bildern und feinen Zeichen drucken und fotografieren. Dieselben Drucke auch kontrolliert erfassen. Digitale Rasterreferenz und kontrollierte Druckreferenz getrennt behalten. So wird sichtbar, welcher Fehler vom Druck und welcher von der Aufnahme stammt. Reale Marker separat annotieren, da sie in der digitalen Vorlage fehlen.
- **Aufnahmetreue-Set:** Reale Seiten jeweils unter kontrolliertem diffusem Licht sowie mit weichem/hartem Schatten, warmem/kühlem Licht und Mischlicht aufnehmen. Kameraposition und manuelle Einstellungen fixieren. Das schattenfreie Foto ist hier die Referenz für Beleuchtungsbereinigung, nicht für Original-PDF-Farbe.
- **Praxis-Set ohne Ground Truth:** Smude-Notenseiten, farbige Lehrbücher, Diagramme, Marker, Fotos und problematische helle Töne. Blindes A/B-Urteil und Fehlertaxonomie; keine objektive „Originalfarbtreue“ aus referenzfreien Bildern behaupten.

RealDAE und SD7K als externe Tests ergänzen, deren eigene Ziele aber nicht vermischen. DESCAN-18K prüft zusätzlich die Scan-/Digitaldomäne. Scanreferenzen benötigen selbst einen definierten Farbprozess; ein beliebiger Scanner ist keine absolute Farbreferenz.

### 8.2 Vergleichsbedingungen fixieren

Alle Kandidaten bekommen exakt dasselbe bereits entzerrte Bild, dieselbe Auflösung und dieselben gültigen Seitenpixel. Keine erneute Geometrieoptimierung pro Methode. Falls Referenzen noch registriert werden müssen, Registrierung unabhängig von der Kandidatenausgabe bestimmen und anschließend einfrieren; sonst kann die Evaluation Schäden wegregistrieren.

Vergleichen: unveränderte Eingabe, klassische Baseline, GCDRNet, DocRes-Appearance, FSENet-SD7K. Separat und klar beschriftet: GC-Net ohne DR-Net, DocRes-Deshadowing sowie optional Schattenentfernung plus identische Weißdarstellung. So lässt sich eine fehlende Weißnormierung von einer schlechten Schattenentfernung unterscheiden. Für DocRes zusätzlich beide Seiten der 1600-Pixel-Verzweigung testen.

Keine individuellen nachträglichen Histogramm-, Weißabgleich- oder Sättigungsanpassungen anhand der Testreferenz. Export verlustfrei mit festem Farbprofil; Originalfarben nicht durch implizite BGR/RGB-Verwechslung oder doppeltes Gamma verändern.

### 8.3 Metriken und ihre Aussagekraft

| Metrik | Aussage | Grenze / vorgeschlagene Verwendung |
| --- | --- | --- |
| PSNR / RGB-RMSE | Pixelabweichung zur jeweiligen Referenz | Nur so sinnvoll wie das Referenzziel; weiße Flächen können das Gesamtergebnis dominieren. Separat für Papier, Druck, Fotos und Schattenregionen ausweisen |
| SSIM / MS-SSIM | Strukturelle Bildähnlichkeit | Kein kalibriertes Farbmaß; Graustufenvarianten verlieren Farbinformation. Fenster, Skalierung und Kanäle festschreiben |
| CIEDE2000 `ΔE00` | Wahrnehmungsbezogene Farbdifferenz nach definierter Farbtransformation | Gemeinsam festgelegter Weißpunkt/ICC-Prozess erforderlich. Median, hohe Quantile und einzelne kritische Farben statt nur Gesamtmittel |
| LPIPS / DISTS | Wahrnehmungsbezogene Ähnlichkeit von Bildmerkmalen | Ergänzung für Fotos/Artefakte, keine Garantie identischer Ziffern, Noten oder Druckfarben |
| OCR-CER bzw. OMR-Auswertung | Lesbarkeit/semantische Folgeeffekte | Sekundäres Schutzmaß; kann eine Farbverfälschung oder verlorene Marker vollständig übersehen |
| Papierweiß / Clipping | Weißabweichung, Restfarbstich, räumliche Schwankung, Anteil exakt weißer sicherer Papierpixel | Gemeinsam mit fälschlich weißgesetzter Druckfläche melden; allein leicht durch Auslöschen optimierbar |
| Detailmaße | Linienkontrast, Strichbreite, Randposition, Unterbrechungen, verlorene/zusätzliche Komponenten | Auf festen, manuell kontrollierten ROIs und bei nativer Auflösung; nicht auf verkleinerten Übersichtsseiten |

SSIM wurde als strukturelles Qualitätsmaß entwickelt; die Autorenimplementierung dokumentiert sowohl Skalenwahl als auch Graustufenverwendung in bestimmten Bewertungen. Für `ΔE00` liefern Sharma, Wu und Dalal Testdaten gegen Implementierungsfehler. [SSIM-Primärquelle](https://ece.uwaterloo.ca/~z70wang/research/ssim/), [CIEDE2000-Autorenmaterial](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/). Die Grenzen und ROI-Aufteilung in der Tabelle sind eigene Empfehlungen.

**Entscheidungsregel:** Zuerst schädliche Veränderungen an Zeichen und hellen Farben ausschließen, dann Farbfehler, Restschatten und Papierweiß beurteilen, zuletzt Laufzeit/Speicher abwägen. Keine einzelne gewichtete Kennzahl sollte eine gelöschte Note durch schönere große Weißflächen ausgleichen. Zusätzlich „saubere Seite hinein → nahezu identische Seite heraus“ und erneute Anwendung auf die eigene Ausgabe testen: starke Drift zeigt eine instabile Enhancement-Vorgabe.

### 8.4 Laufzeit und Reproduzierbarkeit

Auf tatsächlichen Zielgeräten messen: Laden/Initialisierung separat, danach kalte und warme Durchläufe; Modellzeit und vollständige Pipeline inklusive Vorverarbeitung, Transfers, Nachverarbeitung und Export. Native Bildgrößen, Spitzen-RAM/VRAM, Threads, FP32/FP16, Provider und tatsächlich ausgeführte Operatoren festhalten. Bei Diffusion zusätzlich Sampler, Schrittzahl und Seeds. Bei exportierten Modellen numerische und visuelle Parität zum Original prüfen. Hierzu werden ausdrücklich keine fremden GPU-Messwerte auf Smude übertragen.

## 9. Einordnung in das bestehende Repository

Gelesen wurden die übergeordneten Vorgaben aus `/Users/simon/AGENTS.md`, die Projekt-[README](../../README.md) und die vorhandene themenspezifische Recherchekonvention in [mobile/nsh/RESEARCH.md](../../mobile/nsh/RESEARCH.md). Eine allgemeine Forschungsablage war nicht vorhanden; deshalb der angeforderte Pfad `docs/research/document-color-restoration.md`.

Laut README verarbeitet Smude Geometrie nativ in Rust, rekonstruiert aus dem Original und hat bereits getrennte Papiernormalisierung und Druckkontrastkorrektur. Farbige Seiten ohne Notensysteme behalten derzeit RGB; Notation erhält überwiegend Grauausgabe. Die README nennt ausdrücklich mögliches Clipping sehr heller Inhalte bei der Druckkontraststufe. Das ist gelesener Projektkontext, keine zusätzliche Codeprüfung.

**Eigene Integrationsfolgerung:** Eine spätere Farbrestaurierung als gesonderte Photometriestufe auf dem unbeschnittenen Tonwertumfang der fertigen RGB-Rekonstruktion testen. Für den Modellvergleich müssen Farbseiten und farbige Notenseiten denselben RGB-Zugang erhalten. Geometrie, Papiernormierung und ästhetische Farbanhebung getrennt profilieren und ablieren. Eine aus niedrig aufgelöster Analyse gewonnene glatte Transformation am hochaufgelösten Original passt konzeptionell zum vorhandenen Ansatz. Erst nach dem beschriebenen Vergleich über ONNX/native Integration entscheiden.

## 10. Grenzen und geprüfte Quellstände

Die Recherche nutzte Websuche, Paper/Preprints, Autorenprojektseiten und Originalrepositories. Sammlungen und Suchtreffer dienten zum Auffinden; ihre Ranglisten wurden nicht als unabhängige Evidenz übernommen. Es handelt sich um eine gezielte Recherche bis zum genannten Stichtag, nicht um eine garantierte vollständige systematische Literaturübersicht.

Einige CVF-/IEEE-/Elsevier-Direktabrufe lieferten Zugriffssperren; soweit möglich wurden Autorenfassungen auf arXiv oder Autorenservern gelesen. Für GCDRNet konnten Verfahren/Inferenz, Datensatzbeschreibung, Lizenz und bibliografische Daten geprüft werden, nicht sämtliche Trainingsdetails des Volltexts. Bei CrossMamba und DocShaDif reicht der Zugang nur für begrenzte Aussagen. Suchindex-Angaben wie „published months ago“ wurden nicht als Veröffentlichungsdatum verwendet. Ein Treffer zu DADoc nennt Dezember 2026; ohne verifiziertes Online-Erstdatum vor dem Stichtag wurde er nicht zur aktuellen Empfehlung herangezogen. [Verlagstreffer](https://www.sciencedirect.com/science/article/pii/S0952197626028514).

Für die Codeprüfung wurden folgende GitHub-HEADs am Recherchestichtag abgefragt; die Empfehlungen beziehen sich auf diese überprüften Stände, nicht auf spätere Änderungen:

| Repository | Commit |
| --- | --- |
| GCDRNet | [`415c97a9a64d0a796dbbad43485ce2ab102f2a7d`](https://github.com/ZZZHANG-jx/GCDRNet/tree/415c97a9a64d0a796dbbad43485ce2ab102f2a7d) |
| DocRes | [`d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560`](https://github.com/ZZZHANG-jx/DocRes/tree/d3e3a18c7c7ad10e4615ca2eb6a10f83128d8560) |
| DocShadow-SD7K | [`b9395ed333c916051d7ae58be212ac62119b61f5`](https://github.com/CXH-Research/DocShadow-SD7K/tree/b9395ed333c916051d7ae58be212ac62119b61f5) |
| DocTr | [`044648919b6b28f2cb1b288db6d8fd97749db537`](https://github.com/fh2019ustc/DocTr/tree/044648919b6b28f2cb1b288db6d8fd97749db537) |
| GL-PGENet | [`42b34533160e24068ef5b030e88cf93f5a8ea49c`](https://github.com/kukugpt/GL-PGENet/tree/42b34533160e24068ef5b030e88cf93f5a8ea49c) |
| MMDIR | [`b1066ed4184b4c669b7105bd3620c42311e10602`](https://github.com/xiaomore/MMDIR/tree/b1066ed4184b4c669b7105bd3620c42311e10602) |
| Descanning | [`56cdaa97ad22a2664d65f343a8f99855dc9d70a1`](https://github.com/jhcha08/Descanning/tree/56cdaa97ad22a2664d65f343a8f99855dc9d70a1) |

Arbeitsgrenzen: keine weiteren Agenten, keine Codeänderungen, keine Commits, keine Modellbenchmarks. `rtk` war weder im Shell-Pfad noch an den geprüften üblichen Installationsorten vorhanden; nach fehlgeschlagenen Aufrufen wurden direkte Lese-/Recherchekommandos verwendet. Bestehende Änderungen im Working Tree wurden nicht bearbeitet.
