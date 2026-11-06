//! CLI orchestration; geometry lives in the library.
mod args;
mod pages;
mod process;
mod profile;

use std::{fs, time::Instant};

use clap::Parser;
use smude_geometry::Observations;

use args::Args;
use pages::process_pages;
use process::process;
use profile::{StepProfile, finish_profile};

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.binary_page_margin.is_finite() || !(0. ..=0.02).contains(&args.binary_page_margin) {
        return Err("binary page margin must be a finite fraction in 0..0.02".into());
    }
    if !args.crop_padding.is_finite() || !(0. ..=0.1).contains(&args.crop_padding) {
        return Err("crop padding must be a finite fraction in 0..0.1".into());
    }
    if args
        .page_aspect
        .is_some_and(|r| !r.is_finite() || !(0.1..=10.).contains(&r))
    {
        return Err("page aspect must be finite in 0.1..10".into());
    }
    if args.uses_page_mode() {
        if args.same_size.is_some() || args.probe_points.is_some() {
            return Err("--same-size and --probe-points require --no-pages or --geometry".into());
        }
        return process_pages(&args);
    }
    fs::create_dir_all(&args.out)?;
    let mut profile = StepProfile::start();
    let result = (|| {
        let mut source = None;
        let detection_start = Instant::now();
        let data: Observations = if let Some(path) = &args.geometry {
            profile.failure_stage = Some("observations_load");
            let start = Instant::now();
            let loaded = fs::read(path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                .and_then(|b| Ok(serde_json::from_slice(&b)?));
            profile.observations_load_seconds = start.elapsed().as_secs_f64();
            loaded?
        } else {
            profile.failure_stage = Some("decode");
            let start = Instant::now();
            let decoded = image::open(&args.image).map(|image| image.to_rgb8());
            profile.decode_seconds = start.elapsed().as_secs_f64();
            source = Some(decoded?);
            profile.failure_stage = Some("native_detection");
            let start = Instant::now();
            let detected = smude_geometry::detect(
                source.as_ref().unwrap(),
                args.analysis_width_for(source.as_ref().unwrap()),
            );
            profile.native_detection_seconds = start.elapsed().as_secs_f64();
            detected?
        };
        let detection_seconds = detection_start.elapsed().as_secs_f64();
        process(
            &args,
            &args.out,
            &data,
            source,
            detection_seconds,
            None,
            &mut profile,
        )
    })();
    finish_profile(&args.out, &mut profile, &result)?;
    result.map(|_| ())
}
