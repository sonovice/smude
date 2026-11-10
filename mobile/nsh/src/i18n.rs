//! Application copy. Message identities and on-disk model identities stay separate.
use nsh_core::prelude::*;
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    En,
    De,
    Pt,
}
impl Language {
    pub fn from_tag(tag: &str) -> Self {
        match tag
            .split(['-', '_'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "de" => Self::De,
            "pt" => Self::Pt,
            _ => Self::En,
        }
    }
    pub fn negotiate(tags: &[&str]) -> Self {
        tags.iter()
            .find_map(|tag| {
                let tag = tag.split(['-', '_']).next()?.to_ascii_lowercase();
                match tag.as_str() {
                    "en" => Some(Self::En),
                    "de" => Some(Self::De),
                    "pt" => Some(Self::Pt),
                    _ => None,
                }
            })
            .unwrap_or_default()
    }
    pub fn text(self, key: Key) -> &'static str {
        let row = key.row();
        match self {
            Self::En => row.0,
            Self::De => row.1,
            Self::Pt => row.2,
        }
    }
    pub fn pages(self, n: usize) -> String {
        let word = match (self, n) {
            (Self::En, 1) => "page",
            (Self::En, _) => "pages",
            (Self::De, 1) => "Seite",
            (Self::De, _) => "Seiten",
            (Self::Pt, 1) => "página",
            (Self::Pt, _) => "páginas",
        };
        format!("{n} {word}")
    }
    pub fn page_of(self, n: usize, total: usize) -> String {
        match self {
            Self::En => format!("Page {n} of {total}"),
            Self::De => format!("Seite {n} von {total}"),
            Self::Pt => format!("Página {n} de {total}"),
        }
    }
}
#[derive(Clone, Copy)]
pub struct Localization {
    pub preferred: Signal<Option<Language>>,
    pub dates: Signal<BTreeMap<String, String>>,
}
impl Localization {
    pub fn new() -> Self {
        let value = Self {
            preferred: signal(None),
            dates: signal(BTreeMap::new()),
        };
        provide_context(value);
        value
    }
    pub fn language(self) -> Language {
        self.preferred
            .get()
            .unwrap_or_else(|| Language::from_tag(locale().as_str()))
    }
}
pub fn language() -> Language {
    use_context::<Localization>()
        .map(|l| l.language())
        .unwrap_or_else(|| Language::from_tag(locale().as_str()))
}
pub fn t(key: Key) -> &'static str {
    language().text(key)
}
#[derive(Clone, Copy, Debug)]
pub enum Key {
    Scans,
    UnfiledScans,
    CreateNewCollection,
    NoCollectionsFound,
    ClearSearch,
    PreviousResult,
    ImageNotAvailable,
    PreparingPdf,
    CouldNotPreparePdf,
    OpenSettings,
    NoPagesToShare,
    AddToCollection,
    DeleteScans,
    NewCollection,
    Collections,
    CollectionName,
    SearchCollections,
    NoCollectionsYet,
    DeleteCollection,
    CollectionDeleted,
    CollectionsDeleted,
    Rename,
    Save,
    Pages,
    Page,
    Share,
    Done,
    Cancel,
    Delete,
    NoPagesYet,
    TakeAPhotoOrImportImages,
    AddPages,
    ImportPhotos,
    Files,
    ImportingImages,
    Result,
    Original,
    CouldNotProcessThisPageTryAgainOrRetakeThePhoto,
    WaitingToProcess,
    TryAgain,
    Retake,
    Keep,
    CorrectionMethod,
    DeletePage,
    PageDeleted,
    PagesDeleted,
    Rotate,
    StartingCamera,
    ReplacePage,
    ScanPages,
    PlaceThePageInTheFrame,
    CaptureOnlyThePageYouWantToReplace,
    WellAligned,
    KeepTheCameraParallelToThePage,
    Light,
    Photos,
    Capture,
    SharePages,
    OnePdfFile,
    OneJpegPerPage,
    FileName,
    Format,
    Color,
    Automatic,
    Grayscale,
    BlackAndWhite,
    Size,
    OriginalsArePreserved,
    PreviewExport,
    PleaseWait,
    Sharing,
    Back,
    Undo,
    PageProcessing,
    Preparing,
    PageDetection,
    DetectingPage,
    AnalyzingPage,
    CorrectingGeometry,
    RenderingImage,
    EnhancingImage,
    GridView,
    ListView,
    DateUnavailable,
    ChoosingImages,
    WaitUntilAllSelectedPagesAreReady,
    PreparingPreview,
    PreparingExport,
    TheReplacementCouldNotBeMatchedYourPreviousPageIsPreserved,
    ThisPageWasNotFullyDetected,
    CouldNotSaveTheResultCheckAvailableStorageAndTryAgain,
    More,
    MoreActions,
    EditPage,
    PreviousPage,
    NextPage,
    RotateClockwise,
    CouldNotStartTheCamera,
    CameraPaused,
    HoldTheCameraSteady,
    AllowCameraAccessInSettings,
    NoCameraAvailable,
    PlaceTheNextPageInTheFrame,
    HoldThePageSteadyInTheFrame,
    IncludeTheEntirePageInTheFrame,
    PageDetectionUnavailable,
    CouldNotSaveThePhoto,
    CloseTheCurrentDialogFirst,
    CollectionStorageIsNotReadyYet,
    CannotOpenTheDialogRightNow,
    TheExportIsMissingOrNotReadyYet,
    ThisExportDoesNotBelongToThisCollection,
    CouldNotDecodeTheImage,
    TheImageIsUnreadableOrExceeds64MegapixelsOr12000PixelsPerEdge,
    CouldNotLoadTheFullResolutionImage,
    CouldNotLoadTheSelectedPhoto,
    CouldNotCreateTheImportedFile,
    CouldNotSaveTheImportedFile,
    AnImagePickerIsAlreadyOpen,
    CouldNotOpenTheImagePicker,
    CouldNotCreateTheImportFolder,
    ImportInterrupted,
    CouldNotReadTheImage,
    CouldNotImportTheImage,
    CouldNotFinishSavingTheImage,
    CouldNotSaveTheImage,
    CouldNotOpenFilesForSharing,
    NotEnoughMemoryForThisImage,
    CouldNotOpenCollectionStorage,
    InvalidFilePath,
    Standard,
    Small,
    Sharpness,
    TheTopOfThePageLooksBlurryRetakeThePhoto,
    TheBottomOfThePageLooksBlurryRetakeThePhoto,
    PartsOfThePageLookBlurryRetakeThePhoto,
}
impl Key {
    fn row(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Scans => ("Scans", "Scans", "Digitalizações"),
            Self::UnfiledScans => (
                "Unfiled scans",
                "Unzugeordnete Scans",
                "Digitalizações sem coleção",
            ),
            Self::CreateNewCollection => (
                "＋ Create new collection…",
                "＋ Neue Sammlung erstellen …",
                "＋ Criar nova coleção…",
            ),
            Self::NoCollectionsFound => (
                "No collections found",
                "Keine Sammlungen gefunden",
                "Nenhuma coleção encontrada",
            ),
            Self::ClearSearch => ("Clear search", "Suche löschen", "Limpar busca"),
            Self::PreviousResult => (
                "Previous result",
                "Bisheriges Ergebnis",
                "Resultado anterior",
            ),
            Self::ImageNotAvailable => (
                "No image available yet",
                "Noch kein Bild verfügbar",
                "Nenhuma imagem disponível ainda",
            ),
            Self::CouldNotPreparePdf => (
                "Could not prepare the PDF. Try again.",
                "PDF konnte nicht erstellt werden. Bitte erneut versuchen.",
                "Não foi possível preparar o PDF. Tente novamente.",
            ),
            Self::PreparingPdf => (
                "Preparing PDF…",
                "PDF wird vorbereitet …",
                "Preparando PDF…",
            ),
            Self::OpenSettings => ("Open Settings", "Einstellungen öffnen", "Abrir ajustes"),
            Self::NoPagesToShare => (
                "No pages to share",
                "Keine Seiten zum Teilen",
                "Nenhuma página para compartilhar",
            ),
            Self::AddToCollection => (
                "Add to collection",
                "Zu Sammlung hinzufügen",
                "Adicionar à coleção",
            ),
            Self::DeleteScans => ("Delete scans", "Scans löschen", "Excluir digitalizações"),
            Self::NewCollection => ("New collection", "Neue Sammlung", "Nova coleção"),
            Self::Collections => ("Collections", "Sammlungen", "Coleções"),
            Self::CollectionName => ("Collection name", "Sammlungsname", "Nome da coleção"),
            Self::SearchCollections => {
                ("Search collections", "Sammlungen suchen", "Buscar coleções")
            }
            Self::NoCollectionsYet => (
                "No collections yet",
                "Noch keine Sammlungen",
                "Nenhuma coleção ainda",
            ),
            Self::DeleteCollection => ("Delete collection", "Sammlung löschen", "Excluir coleção"),
            Self::CollectionDeleted => (
                "Collection deleted",
                "Sammlung gelöscht",
                "Coleção excluída",
            ),
            Self::CollectionsDeleted => (
                "Collections deleted",
                "Sammlungen gelöscht",
                "Coleções excluídas",
            ),
            Self::Rename => ("Rename", "Umbenennen", "Renomear"),
            Self::Save => ("Save", "Speichern", "Salvar"),
            Self::Pages => ("Pages", "Seiten", "Páginas"),
            Self::Page => ("Page", "Seite", "Página"),
            Self::Share => ("Share", "Teilen", "Compartilhar"),
            Self::Done => ("Done", "Fertig", "Concluir"),
            Self::Cancel => ("Cancel", "Abbrechen", "Cancelar"),
            Self::Delete => ("Delete", "Löschen", "Excluir"),
            Self::NoPagesYet => ("No pages yet", "Noch keine Seiten", "Nenhuma página ainda"),
            Self::TakeAPhotoOrImportImages => (
                "Take a photo or import images.",
                "Fotografiere eine Seite oder füge Bilder hinzu.",
                "Fotografe uma página ou importe imagens.",
            ),
            Self::AddPages => ("Add pages", "Seiten hinzufügen", "Adicionar páginas"),
            Self::ImportPhotos => ("Import photos", "Fotos importieren", "Importar fotos"),
            Self::Files => ("Files", "Dateien", "Arquivos"),
            Self::ImportingImages => (
                "Importing images…",
                "Bilder werden importiert …",
                "Importando imagens…",
            ),
            Self::Result => ("Processed", "Bearbeitet", "Processada"),
            Self::Original => ("Original", "Original", "Original"),
            Self::CouldNotProcessThisPageTryAgainOrRetakeThePhoto => (
                "Could not process this page. Try again or retake the photo.",
                "Verarbeitung fehlgeschlagen",
                "Não foi possível processar esta página. Tente novamente ou tire outra foto.",
            ),
            Self::WaitingToProcess => (
                "Waiting to process…",
                "Wartet auf Verarbeitung …",
                "Aguardando processamento…",
            ),
            Self::TryAgain => ("Try again", "Erneut versuchen", "Tentar novamente"),
            Self::Retake => ("Retake", "Erneut aufnehmen", "Fotografar novamente"),
            Self::Keep => ("Keep", "Behalten", "Manter"),
            Self::CorrectionMethod => ("Correction method", "Entzerrung", "Método de correção"),
            Self::DeletePage => ("Delete page", "Seite löschen", "Excluir página"),
            Self::PageDeleted => ("Page deleted", "Seite gelöscht", "Página excluída"),
            Self::PagesDeleted => ("Pages deleted", "Seiten gelöscht", "Páginas excluídas"),
            Self::Rotate => ("Rotate", "Drehen", "Girar"),
            Self::StartingCamera => ("Starting camera…", "Kamera starten …", "Iniciando câmera…"),
            Self::ReplacePage => ("Replace page", "Seite ersetzen", "Substituir página"),
            Self::ScanPages => ("Scan pages", "Seiten aufnehmen", "Digitalizar páginas"),
            Self::PlaceThePageInTheFrame => (
                "Place the page in the frame",
                "Seite ins Bild halten",
                "Posicione a página no enquadramento",
            ),
            Self::CaptureOnlyThePageYouWantToReplace => (
                "Capture only the page you want to replace.",
                "Bitte nur die zu ersetzende Seite aufnehmen.",
                "Fotografe apenas a página que deseja substituir.",
            ),
            Self::WellAligned => ("Well aligned", "Gut ausgerichtet", "Bem alinhado"),
            Self::KeepTheCameraParallelToThePage => (
                "Keep the camera parallel to the page",
                "Frontal ausrichten",
                "Mantenha a câmera paralela à página",
            ),
            Self::Light => ("Light", "Licht", "Luz"),
            Self::Photos => ("Photos", "Fotos", "Fotos"),
            Self::Capture => ("Capture", "Aufnehmen", "Fotografar"),
            Self::SharePages => ("Share", "Teilen", "Compartilhar"),
            Self::OnePdfFile => ("one PDF file", "eine PDF-Datei", "um arquivo PDF"),
            Self::OneJpegPerPage => (
                "one JPEG per page",
                "ein JPEG je Seite",
                "um JPEG por página",
            ),
            Self::FileName => ("File name", "Dateiname", "Nome do arquivo"),
            Self::Format => ("Format", "Format", "Formato"),
            Self::Color => ("Color", "Farbe", "Cor"),
            Self::Automatic => ("Automatic", "Automatisch", "Automático"),
            Self::Grayscale => ("Grayscale", "Graustufen", "Escala de cinza"),
            Self::BlackAndWhite => ("Black and white", "Schwarz-Weiß", "Preto e branco"),
            Self::Size => ("Size", "Größe", "Tamanho"),
            Self::OriginalsArePreserved => (
                "Originals are preserved.",
                "Originale bleiben erhalten.",
                "Os originais são preservados.",
            ),
            Self::PreviewExport => (
                "Preview export",
                "Exportvorschau ansehen",
                "Pré-visualizar exportação",
            ),
            Self::PleaseWait => ("Please wait…", "Bitte warten …", "Aguarde…"),
            Self::Sharing => ("Sharing…", "Teilen …", "Compartilhando…"),
            Self::Back => ("Back", "Zurück", "Voltar"),
            Self::Undo => ("Undo", "Rückgängig", "Desfazer"),
            Self::PageProcessing => (
                "Page processing",
                "Seitenverarbeitung",
                "Processamento da página",
            ),
            Self::Preparing => ("Preparing", "Vorbereiten", "Preparando"),
            Self::PageDetection => ("Page detection", "Seitenerkennung", "Detecção de página"),
            Self::DetectingPage => ("Detecting page", "Seite erkennen", "Detectando página"),
            Self::AnalyzingPage => ("Analyzing page", "Seite analysieren", "Analisando página"),
            Self::CorrectingGeometry => {
                ("Correcting geometry", "Entzerren", "Corrigindo geometria")
            }
            Self::RenderingImage => ("Rendering image", "Bild erstellen", "Renderizando imagem"),
            Self::EnhancingImage => ("Enhancing image", "Bild aufbereiten", "Melhorando imagem"),
            Self::GridView => ("Grid view", "Rasteransicht", "Visualização em grade"),
            Self::ListView => ("List view", "Listenansicht", "Visualização em lista"),
            Self::DateUnavailable => ("Date unavailable", "Datum unbekannt", "Data indisponível"),
            Self::ChoosingImages => (
                "Choosing images…",
                "Bilder auswählen …",
                "Selecionando imagens…",
            ),
            Self::WaitUntilAllSelectedPagesAreReady => (
                "Wait until all selected pages are ready.",
                "Bitte warten, bis alle ausgewählten Seiten fertig sind.",
                "Aguarde até que todas as páginas selecionadas estejam prontas.",
            ),
            Self::PreparingPreview => (
                "Preparing preview…",
                "Vorschau wird erstellt …",
                "Preparando pré-visualização…",
            ),
            Self::PreparingExport => (
                "Preparing export…",
                "Export wird erstellt …",
                "Preparando exportação…",
            ),
            Self::TheReplacementCouldNotBeMatchedYourPreviousPageIsPreserved => (
                "The replacement could not be matched. Your previous page is preserved.",
                "Neue Aufnahme konnte nicht zugeordnet werden. Die bisherige Seite bleibt erhalten.",
                "Não foi possível identificar a substituição. A página anterior foi preservada.",
            ),
            Self::ThisPageWasNotFullyDetected => (
                "This page was not fully detected.",
                "Diese Seite wurde nicht vollständig erkannt.",
                "Esta página não foi detectada por completo.",
            ),
            Self::CouldNotSaveTheResultCheckAvailableStorageAndTryAgain => (
                "Could not save the result. Check available storage and try again.",
                "Ergebnis konnte nicht gespeichert werden. Bitte Speicherplatz prüfen und erneut versuchen.",
                "Não foi possível salvar o resultado. Verifique o espaço disponível e tente novamente.",
            ),
            Self::More => ("More", "Mehr", "Mais"),
            Self::MoreActions => ("More actions", "Weitere Aktionen", "Mais ações"),
            Self::EditPage => ("Edit page", "Seite bearbeiten", "Editar página"),
            Self::PreviousPage => ("Previous page", "Vorherige Seite", "Página anterior"),
            Self::NextPage => ("Next page", "Nächste Seite", "Próxima página"),
            Self::RotateClockwise => (
                "Rotate clockwise",
                "Drehen im Uhrzeigersinn",
                "Girar no sentido horário",
            ),
            Self::CouldNotStartTheCamera => (
                "Could not start the camera",
                "Kamera konnte nicht gestartet werden",
                "Não foi possível iniciar a câmera",
            ),
            Self::CameraPaused => ("Camera paused", "Kamera pausiert", "Câmera pausada"),
            Self::HoldTheCameraSteady => (
                "Hold the camera steady",
                "Kamera ruhig halten",
                "Mantenha a câmera estável",
            ),
            Self::AllowCameraAccessInSettings => (
                "Allow camera access in Settings",
                "Kamerazugriff in den Einstellungen erlauben",
                "Permita o acesso à câmera nos Ajustes",
            ),
            Self::NoCameraAvailable => (
                "No camera available",
                "Keine Kamera verfügbar",
                "Nenhuma câmera disponível",
            ),
            Self::PlaceTheNextPageInTheFrame => (
                "Place the next page in the frame",
                "Nächste Seite ins Bild halten",
                "Posicione a próxima página no enquadramento",
            ),
            Self::HoldThePageSteadyInTheFrame => (
                "Hold the page steady in the frame",
                "Seite ruhig ins Bild halten",
                "Mantenha a página estável no enquadramento",
            ),
            Self::IncludeTheEntirePageInTheFrame => (
                "Include the entire page in the frame",
                "Ganze Seite ins Bild nehmen",
                "Enquadre a página inteira",
            ),
            Self::PageDetectionUnavailable => (
                "Page detection unavailable",
                "Seitenerkennung nicht verfügbar",
                "Detecção de página indisponível",
            ),
            Self::CouldNotSaveThePhoto => (
                "Could not save the photo",
                "Aufnahme konnte nicht gespeichert werden",
                "Não foi possível salvar a foto",
            ),
            Self::CloseTheCurrentDialogFirst => (
                "Close the current dialog first.",
                "Bitte den geöffneten Dialog zuerst schließen.",
                "Feche primeiro a janela atual.",
            ),
            Self::CollectionStorageIsNotReadyYet => (
                "Collection storage is not ready yet.",
                "Der Sammlungsspeicher ist noch nicht bereit.",
                "O armazenamento das coleções ainda não está pronto.",
            ),
            Self::CannotOpenTheDialogRightNow => (
                "Cannot open the dialog right now.",
                "Der Dialog kann gerade nicht geöffnet werden.",
                "Não é possível abrir a janela agora.",
            ),
            Self::TheExportIsMissingOrNotReadyYet => (
                "The export is missing or not ready yet.",
                "Die Exportdatei fehlt oder ist noch nicht fertig.",
                "A exportação está ausente ou ainda não está pronta.",
            ),
            Self::ThisExportDoesNotBelongToThisCollection => (
                "This export does not belong to this collection.",
                "Die Exportdatei gehört nicht zu dieser Sammlung.",
                "Esta exportação não pertence a esta coleção.",
            ),
            Self::CouldNotDecodeTheImage => (
                "Could not decode the image.",
                "Bild konnte nicht dekodiert werden.",
                "Não foi possível decodificar a imagem.",
            ),
            Self::TheImageIsUnreadableOrExceeds64MegapixelsOr12000PixelsPerEdge => (
                "The image is unreadable or exceeds 64 megapixels or 12,000 pixels per edge.",
                "Bild nicht lesbar oder größer als 64 Megapixel / 12.000 Pixel Kantenlänge.",
                "A imagem não pode ser lida ou excede 64 megapixels ou 12.000 pixels por lado.",
            ),
            Self::CouldNotLoadTheFullResolutionImage => (
                "Could not load the full-resolution image.",
                "Das Bild konnte nicht in voller Auflösung geladen werden.",
                "Não foi possível carregar a imagem em resolução máxima.",
            ),
            Self::CouldNotLoadTheSelectedPhoto => (
                "Could not load the selected photo.",
                "Das ausgewählte Foto konnte nicht geladen werden.",
                "Não foi possível carregar a foto selecionada.",
            ),
            Self::CouldNotCreateTheImportedFile => (
                "Could not create the imported file.",
                "Importdatei konnte nicht angelegt werden.",
                "Não foi possível criar o arquivo importado.",
            ),
            Self::CouldNotSaveTheImportedFile => (
                "Could not save the imported file.",
                "Importdatei konnte nicht gespeichert werden.",
                "Não foi possível salvar o arquivo importado.",
            ),
            Self::AnImagePickerIsAlreadyOpen => (
                "An image picker is already open",
                "Eine Bildauswahl ist bereits geöffnet",
                "Uma seleção de imagens já está aberta",
            ),
            Self::CouldNotOpenTheImagePicker => (
                "Could not open the image picker",
                "Bildauswahl konnte nicht geöffnet werden",
                "Não foi possível abrir a seleção de imagens",
            ),
            Self::CouldNotCreateTheImportFolder => (
                "Could not create the import folder",
                "Importverzeichnis konnte nicht angelegt werden",
                "Não foi possível criar a pasta de importação",
            ),
            Self::ImportInterrupted => (
                "Import interrupted",
                "Import unterbrochen",
                "Importação interrompida",
            ),
            Self::CouldNotReadTheImage => (
                "Could not read the image",
                "Bild konnte nicht gelesen werden",
                "Não foi possível ler a imagem",
            ),
            Self::CouldNotImportTheImage => (
                "Could not import the image",
                "Bild konnte nicht übernommen werden",
                "Não foi possível importar a imagem",
            ),
            Self::CouldNotFinishSavingTheImage => (
                "Could not finish saving the image",
                "Bild konnte nicht abgeschlossen werden",
                "Não foi possível concluir o salvamento da imagem",
            ),
            Self::CouldNotSaveTheImage => (
                "Could not save the image",
                "Bild konnte nicht gespeichert werden",
                "Não foi possível salvar a imagem",
            ),
            Self::CouldNotOpenFilesForSharing => (
                "Could not open files for sharing",
                "Dateien konnten nicht zum Teilen geöffnet werden",
                "Não foi possível abrir os arquivos para compartilhar",
            ),
            Self::NotEnoughMemoryForThisImage => (
                "Not enough memory for this image.",
                "Für dieses Bild ist zu wenig Arbeitsspeicher verfügbar.",
                "Memória insuficiente para esta imagem.",
            ),
            Self::CouldNotOpenCollectionStorage => (
                "Could not open collection storage",
                "Sammlungsspeicher konnte nicht geöffnet werden",
                "Não foi possível abrir o armazenamento das coleções",
            ),
            Self::InvalidFilePath => (
                "Invalid file path",
                "Ungültiger Dateipfad",
                "Caminho de arquivo inválido",
            ),
            Self::Standard => ("Standard", "Standard", "Padrão"),
            Self::Small => ("Small", "Klein", "Pequeno"),
            Self::Sharpness => ("Sharpness", "Schärfe", "Nitidez"),
            Self::TheTopOfThePageLooksBlurryRetakeThePhoto => (
                "The top of the page looks blurry. Retake the photo?",
                "Oben unscharf. Foto erneut aufnehmen?",
                "A parte superior da página parece desfocada. Fotografar novamente?",
            ),
            Self::TheBottomOfThePageLooksBlurryRetakeThePhoto => (
                "The bottom of the page looks blurry. Retake the photo?",
                "Unten unscharf. Foto erneut aufnehmen?",
                "A parte inferior da página parece desfocada. Fotografar novamente?",
            ),
            Self::PartsOfThePageLookBlurryRetakeThePhoto => (
                "Parts of the page look blurry. Retake the photo?",
                "Seite unscharf. Foto erneut aufnehmen?",
                "Partes da página parecem desfocadas. Fotografar novamente?",
            ),
        }
    }
}

impl Default for Localization {
    fn default() -> Self {
        Self::new()
    }
}
pub fn page_title(n: usize) -> String {
    format!("{} {n}", t(Key::Page))
}
pub fn page_count(n: usize) -> String {
    language().pages(n)
}
pub fn on_device(n: usize) -> String {
    let count = page_count(n);
    match language() {
        Language::En => format!("{count} · on this device"),
        Language::De => format!("{count} · auf diesem Gerät"),
        Language::Pt => format!("{count} · neste dispositivo"),
    }
}
pub fn selection(pages: &str) -> String {
    match language() {
        Language::En => format!("Selected: {pages}"),
        Language::De => format!("Auswahl: {pages}"),
        Language::Pt => format!("Seleção: {pages}"),
    }
}
pub fn size_label(size: crate::model::ExportSize) -> &'static str {
    use crate::model::ExportSize;
    t(match size {
        ExportSize::Small => Key::Small,
        ExportSize::Standard => Key::Standard,
        ExportSize::Original => Key::Original,
    })
}
pub fn size_detail(size: crate::model::ExportSize) -> String {
    use crate::model::ExportSize;
    match (language(), size) {
        (Language::En, ExportSize::Original) => "Full result resolution".into(),
        (Language::De, ExportSize::Original) => "Volle Ergebnisauflösung".into(),
        (Language::Pt, ExportSize::Original) => "Resolução máxima do resultado".into(),
        (lang, size) => {
            let n = match (lang, size == ExportSize::Small) {
                (Language::En, true) => "1,600",
                (Language::En, false) => "2,600",
                (_, true) => "1.600",
                (_, false) => "2.600",
            };
            match lang {
                Language::En => format!("Longest edge: up to {n} px"),
                Language::De => format!("Lange Kante: maximal {n} px"),
                Language::Pt => format!("Lado maior: até {n} px"),
            }
        }
    }
}
pub fn quality(q: &smude_geometry::detection::quality::CaptureQuality) -> Option<String> {
    let weak: Vec<_> = q
        .bands
        .iter()
        .filter(|b| b.blurred)
        .map(|b| b.band)
        .collect();
    if weak.is_empty() {
        return None;
    }
    let text = match (language(), weak.as_slice()) {
        (Language::En, [0]) => "The top may be blurry",
        (Language::En, [1]) => "The middle may be blurry",
        (Language::En, [2]) => "The bottom may be blurry",
        (Language::En, _) => "Parts of the page may be blurry",
        (Language::De, [0]) => "Oben möglicherweise unscharf",
        (Language::De, [1]) => "Mitte möglicherweise unscharf",
        (Language::De, [2]) => "Unten möglicherweise unscharf",
        (Language::De, _) => "Teile der Seite möglicherweise unscharf",
        (Language::Pt, [0]) => "A parte superior pode estar desfocada",
        (Language::Pt, [1]) => "A parte central pode estar desfocada",
        (Language::Pt, [2]) => "A parte inferior pode estar desfocada",
        (Language::Pt, _) => "Partes da página podem estar desfocadas",
    };
    Some(text.into())
}

pub fn camera_message(code: &str) -> String {
    let row = match code {
        "page_in_frame" => (
            "Place the page in the frame",
            "Seite ins Bild halten",
            "Posicione a página no enquadramento",
        ),
        "include_spread" => (
            "Spread detected — include both complete pages",
            "Doppelseite erkannt – beide Seiten vollständig ins Bild nehmen",
            "Página dupla detectada — enquadre as duas páginas inteiras",
        ),
        "include_page" => (
            "Include the entire page in the frame",
            "Ganze Seite ins Bild nehmen",
            "Enquadre a página inteira",
        ),
        "spread_detected" => (
            "Spread detected",
            "Doppelseite erkannt",
            "Página dupla detectada",
        ),
        "page_detected" => ("Page detected", "Seite erkannt", "Página detectada"),
        _ => return code.into(),
    };
    match language() {
        Language::En => row.0,
        Language::De => row.1,
        Language::Pt => row.2,
    }
    .into()
}
#[derive(Clone, Copy)]
pub enum Failure {
    Save,
    Preference,
    Import,
    Export,
    Retake,
}
pub fn failure(kind: Failure, diagnostic: &str) -> String {
    let row = match kind {
        Failure::Save => (
            "Could not save",
            "Speichern fehlgeschlagen",
            "Não foi possível salvar",
        ),
        Failure::Preference => (
            "Could not save this preference",
            "Ansicht konnte nicht gespeichert werden",
            "Não foi possível salvar esta preferência",
        ),
        Failure::Import => (
            "Could not import the image",
            "Import fehlgeschlagen",
            "Não foi possível importar a imagem",
        ),
        Failure::Export => (
            "Could not prepare the export",
            "Export fehlgeschlagen",
            "Não foi possível preparar a exportação",
        ),
        Failure::Retake => (
            "Could not process the replacement. Your previous page is preserved.",
            "Neue Aufnahme konnte nicht verarbeitet werden. Die bisherige Seite bleibt erhalten.",
            "Não foi possível processar a substituição. A página anterior foi preservada.",
        ),
    };
    eprintln!("smude mobile: {diagnostic}");
    match language() {
        Language::En => row.0,
        Language::De => row.1,
        Language::Pt => row.2,
    }
    .into()
}

/// Context belongs in action labels, not in extra rows above selected cards.
pub fn share_pages(n: usize) -> String {
    let count = page_count(n);
    match language() {
        Language::En => format!("Share {count}"),
        Language::De => format!("{count} teilen"),
        Language::Pt => format!("Compartilhar {count}"),
    }
}
pub fn delete_pages(n: usize) -> String {
    let count = page_count(n);
    match language() {
        Language::En => format!("Delete {count}"),
        Language::De => format!("{count} löschen"),
        Language::Pt => format!("Excluir {count}"),
    }
}
pub fn delete_collections(n: usize) -> String {
    match (language(), n) {
        (Language::En, 1) => "Delete 1 collection".into(),
        (Language::En, _) => format!("Delete {n} collections"),
        (Language::De, 1) => "1 Sammlung löschen".into(),
        (Language::De, _) => format!("{n} Sammlungen löschen"),
        (Language::Pt, 1) => "Excluir 1 coleção".into(),
        (Language::Pt, _) => format!("Excluir {n} coleções"),
    }
}
pub fn review_pages(n: usize) -> String {
    let count = page_count(n);
    match language() {
        Language::En => format!("Review {count}"),
        Language::De => format!("{count} ansehen"),
        Language::Pt => format!("Ver {count}"),
    }
}
pub fn replace_page(n: usize) -> String {
    match language() {
        Language::En => format!("Replace page {n}"),
        Language::De => format!("Seite {n} ersetzen"),
        Language::Pt => format!("Substituir página {n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_language_preferences_preserve_english_and_match_both_portuguese_regions() {
        assert_eq!(Language::negotiate(&["fr", "de-DE"]), Language::De);
        assert_eq!(Language::negotiate(&["en-GB", "de-DE"]), Language::En);
        assert_eq!(Language::negotiate(&["pt-BR", "en"]), Language::Pt);
        assert_eq!(Language::negotiate(&["pt-PT"]), Language::Pt);
        assert_eq!(Language::negotiate(&["fr", "ja"]), Language::En);
        assert_eq!(Language::En.pages(0), "0 pages");
        assert_eq!(Language::Pt.pages(1), "1 página");
        assert_eq!(Language::Pt.pages(2), "2 páginas");
        assert_eq!(Language::De.text(Key::NewCollection), "Neue Sammlung");
    }
}
