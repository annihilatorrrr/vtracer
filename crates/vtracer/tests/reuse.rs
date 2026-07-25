//! Two-phase pipeline: cache the expensive segmentation, re-run the cheap
//! downstream stages with different parameters (the interactive tuning loop).

use vtracer::{ColorImage, Config, FitMode};

/// A few colored blocks — several clusters, a few holes.
fn blocks() -> ColorImage {
    let (w, h) = (48usize, 48usize);
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let c = match (x / 16, y / 16) {
                (0, _) => (220u8, 40, 40),
                (1, 0) => (40, 200, 60),
                (1, _) => (50, 60, 220),
                _ => (230, 210, 40),
            };
            pixels.extend_from_slice(&[c.0, c.1, c.2, 255]);
        }
    }
    ColorImage {
        pixels,
        width: w,
        height: h,
    }
}

fn cfg(mode: FitMode) -> Config {
    Config {
        mode,
        ..Config::default()
    }
}

/// `finish(segment(img))` equals the one-shot `run(img)`.
#[test]
fn two_phase_matches_one_shot() {
    let img = blocks();
    let pipeline = cfg(FitMode::Spline).build().unwrap();

    let one_shot = pipeline.run(&img).unwrap();
    let seg = pipeline.segment(&img).unwrap();
    let two_phase = pipeline.finish(&seg).unwrap();

    assert_eq!(
        pipeline.writer.write(&one_shot),
        pipeline.writer.write(&two_phase),
        "splitting segment/finish must not change the output"
    );
}

/// A cached segmentation stays pristine — `finish` can be called repeatedly and
/// deterministically (color fitting mutates only an internal clone).
#[test]
fn cached_segmentation_is_reusable() {
    let img = blocks();
    let pipeline = cfg(FitMode::Polygon).build().unwrap();

    let seg = pipeline.segment(&img).unwrap();
    let first = pipeline.writer.write(&pipeline.finish(&seg).unwrap());
    let second = pipeline.writer.write(&pipeline.finish(&seg).unwrap());

    assert_eq!(first, second, "reusing a cached segmentation must be stable");
}

/// The tuning workflow: segment once, then feed that segmentation to pipelines
/// with different curve-fitting parameters. Same regions, different geometry —
/// and no re-segmentation.
#[test]
fn tune_curve_fitting_on_cached_segmentation() {
    let img = blocks();

    // Same clustering parameters (defaults), different fit modes → the
    // segmentation from one is valid input to the other's `finish`.
    let pixel = cfg(FitMode::Pixel).build().unwrap();
    let spline = cfg(FitMode::Spline).build().unwrap();

    let seg = pixel.segment(&img).unwrap();

    let doc_pixel = pixel.finish(&seg).unwrap();
    let doc_spline = spline.finish(&seg).unwrap();

    // Same partition → same number of shapes.
    assert_eq!(doc_pixel.shapes.len(), doc_spline.shapes.len());
    assert!(!doc_pixel.shapes.is_empty());

    // But the fitted geometry differs (straight edges vs cubic curves).
    assert_ne!(
        pixel.writer.write(&doc_pixel),
        spline.writer.write(&doc_spline),
        "pixel and spline fitting should produce different paths"
    );
}

/// `filter_speckle` is now a finish-phase area filter, so tuning it reuses the
/// cached segmentation — no re-clustering. A higher threshold drops the tiny
/// dots; the low threshold keeps them.
#[test]
fn tune_speckle_on_cached_segmentation() {
    // White background, a 10x10 block, and four 2x2 dots (4px each).
    let (w, h) = (32usize, 32usize);
    let dots = [(4usize, 4usize), (4, 26), (26, 4), (26, 26)];
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let c = if (12..22).contains(&x) && (12..22).contains(&y) {
                (220u8, 40, 40) // center block
            } else if dots.iter().any(|&(dx, dy)| x >= dx && x < dx + 2 && y >= dy && y < dy + 2) {
                (10, 10, 10) // dots
            } else {
                (245, 245, 245) // background
            };
            pixels.extend_from_slice(&[c.0, c.1, c.2, 255]);
        }
    }
    let img = ColorImage {
        pixels,
        width: w,
        height: h,
    };

    // filter_speckle no longer affects the frontend, so these share a
    // segmentation: cluster once, filter differently in finish. Disable the
    // thin filter so speckle area is the only thing varying.
    let keep = Config {
        filter_speckle: 1, // area 1 → keep the 4px dots
        filter_thin: false,
        ..Config::default()
    }
    .build()
    .unwrap();
    let drop = Config {
        filter_speckle: 3, // area 9 → drop the 4px dots
        filter_thin: false,
        ..Config::default()
    }
    .build()
    .unwrap();

    let seg = keep.segment(&img).unwrap(); // the expensive step, done once
    let n_keep = keep.finish(&seg).unwrap().shapes.len();
    let n_drop = drop.finish(&seg).unwrap().shapes.len(); // same cached seg

    assert!(
        n_keep > n_drop,
        "raising filter_speckle should drop the tiny dots without re-clustering: \
         keep={n_keep}, drop={n_drop}"
    );
}

/// The thin-strand filter is also a finish-phase toggle on a cached
/// segmentation: a 2px-wide strand survives with `filter_thin = false` and is
/// dropped with `filter_thin = true`, while the compact block always survives.
#[test]
fn tune_thin_filter_on_cached_segmentation() {
    let (w, h) = (40usize, 40usize);
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let c = if (4..24).contains(&x) && (4..24).contains(&y) {
                (40u8, 120, 220) // 20x20 compact block (not thin)
            } else if (30..32).contains(&x) && (8..28).contains(&y) {
                (220, 40, 40) // 2x20 thread-like strand
            } else {
                (245, 245, 245) // background
            };
            pixels.extend_from_slice(&[c.0, c.1, c.2, 255]);
        }
    }
    let img = ColorImage {
        pixels,
        width: w,
        height: h,
    };

    // Small speckle area so the 40px strand isn't removed by the area filter;
    // only filter_thin varies between the two.
    let base = Config {
        filter_speckle: 1,
        ..Config::default()
    };
    let keep_thin = Config {
        filter_thin: false,
        ..base.clone()
    }
    .build()
    .unwrap();
    let drop_thin = Config {
        filter_thin: true,
        ..base
    }
    .build()
    .unwrap();

    let seg = keep_thin.segment(&img).unwrap(); // cluster once
    let n_keep = keep_thin.finish(&seg).unwrap().shapes.len();
    let n_drop = drop_thin.finish(&seg).unwrap().shapes.len(); // same cached seg

    assert!(
        n_keep > n_drop,
        "filter_thin should drop the thread-like strand: keep={n_keep}, drop={n_drop}"
    );
}
