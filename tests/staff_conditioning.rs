use smude_geometry::{Barline, Observations, Staff, condition_staff_observations};

fn packet(noise: f64, bump: bool) -> (Observations, Observations) {
    let x: Vec<_> = (0..129).map(|i| 40. + 880. * i as f64 / 128.).collect();
    let y = std::array::from_fn(|k| {
        x.iter()
            .map(|x| {
                let center = 240.
                    + 0.000008 * (x - 480.).powi(2)
                    + 0.4 * (x / 180.).sin()
                    + if bump {
                        2. * (-((x - 480.) / 30.).powi(2)).exp()
                    } else {
                        0.
                    };
                center + (k as f64 - 2.) * (10. + 0.0002 * x)
            })
            .collect()
    });
    let truth = Observations {
        width: 1000,
        height: 700,
        staves: vec![Staff { x, y }],
        barlines: vec![Barline {
            staff: 0,
            aligned_staffs: vec![],
            x: vec![500., 500.],
            y: vec![210., 270.],
        }],
    };
    let mut observed = truth.clone();
    let mut state = 20261106_u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    for y in &mut observed.staves[0].y {
        for v in y {
            let gaussian =
                (-2. * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
            *v += noise * gaussian;
        }
    }
    (observed, truth)
}
fn mse(a: &Observations, b: &Observations) -> f64 {
    a.staves[0]
        .y
        .iter()
        .flatten()
        .zip(b.staves[0].y.iter().flatten())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / (a.staves[0].x.len() * 5) as f64
}

#[test]
fn coherent_bends_and_spacing_are_exact_identity() {
    let (data, _) = packet(0., false);
    let (result, report) = condition_staff_observations(&data);
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::to_value(&data).unwrap()
    );
    assert_eq!(report.changed_samples, 0);
}

#[test]
fn independent_line_noise_is_reduced_without_changing_domains_or_bars() {
    let (data, truth) = packet(0.12, false);
    let (result, report) = condition_staff_observations(&data);
    assert!(mse(&result, &truth) < mse(&data, &truth) * 0.5);
    let staff = &report.staves[0];
    assert!((0.06..0.22).contains(&staff.estimated_noise_pixels));
    assert!(staff.max_deviation_pixels <= 3. * staff.estimated_noise_pixels + 1e-12);
    assert_eq!(data.staves[0].x, result.staves[0].x);
    assert_eq!(
        serde_json::to_value(&data.barlines).unwrap(),
        serde_json::to_value(&result.barlines).unwrap()
    );
    assert_eq!((data.width, data.height), (result.width, result.height));
}

#[test]
fn a_real_localized_bend_is_preserved_rather_than_becoming_noise() {
    let (data, _) = packet(0., true);
    let (result, report) = condition_staff_observations(&data);
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::to_value(&data).unwrap()
    );
    assert_eq!(report.changed_samples, 0);
    let (noisy, truth) = packet(0.12, true);
    let (result, _) = condition_staff_observations(&noisy);
    let mid = 64;
    let actual = result.staves[0].y.iter().map(|y| y[mid]).sum::<f64>() / 5.;
    let expected = truth.staves[0].y.iter().map(|y| y[mid]).sum::<f64>() / 5.;
    assert!((actual - expected).abs() < 0.25);
    assert!(mse(&result, &truth) < mse(&noisy, &truth) * 0.6);
}

#[test]
fn persistent_line_specific_offsets_are_not_forced_to_equal_spacing() {
    let (mut data, _) = packet(0., false);
    for (j, y) in data.staves[0].y[0].iter_mut().enumerate() {
        *y += 0.3 + 0.2 * (j as f64 / 40.).sin();
    }
    let (result, report) = condition_staff_observations(&data);
    assert_eq!(report.changed_samples, 0);
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::to_value(&data).unwrap()
    );
}
