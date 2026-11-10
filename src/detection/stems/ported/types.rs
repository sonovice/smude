//! Percentages of a whole.

const PERCENT_DENOMINATOR: i32 = 100;

/// A percentage from 0 to 100 inclusive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Percent(u8);

impl Percent {
    /// 0 %.
    pub const ZERO: Percent = Percent(0);
    /// 100 %.
    pub const FULL: Percent = Percent(PERCENT_DENOMINATOR as u8);

    /// `value` percent, `None` above 100.
    pub const fn new(value: u8) -> Option<Self> {
        if value as i32 <= PERCENT_DENOMINATOR {
            Some(Percent(value))
        } else {
            None
        }
    }

    /// `value` percent, for constants: evaluating it with a value above 100 fails (at compile
    /// time in a `const` item).
    pub const fn constant(value: u8) -> Self {
        match Percent::new(value) {
            Some(p) => p,
            None => panic!("a percentage is at most 100"),
        }
    }

    /// `part` as a percentage of `whole`, saturating at 100 and 0 unless both are positive.
    ///
    /// The quotient is formed first and then scaled, in `f64`, and truncated, so a fraction
    /// just below a whole percent stays below it (29 of 100 gives 28 %). Thresholds tuned on
    /// this arithmetic depend on that.
    pub fn of_fraction(part: i32, whole: i32) -> Self {
        if part < 1 || whole < 1 {
            return Percent::ZERO;
        }
        if part > whole {
            return Percent::FULL;
        }
        let scaled = part as f64 / whole as f64 * PERCENT_DENOMINATOR as f64;
        Percent(scaled as u8)
    }

    /// The percentage as a number from 0 to 100.
    pub const fn get(self) -> u8 {
        self.0
    }

    /// This percentage of `whole`, truncated towards zero (`whole * p / 100`).
    pub const fn of(self, whole: i32) -> i32 {
        whole * self.0 as i32 / PERCENT_DENOMINATOR
    }

    /// `part` as a percentage of `whole`, truncated; `None` if `whole` is zero or `part` is
    /// outside `0..=whole`.
    pub fn ratio(part: u64, whole: u64) -> Option<Self> {
        if whole == 0 || part > whole {
            return None;
        }
        let value = part * PERCENT_DENOMINATOR as u64 / whole;
        Percent::new(u8::try_from(value).ok()?)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Px(pub i32);
impl Px {
    pub const fn get(self) -> i32 {
        self.0
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SpacePercent(pub u16);
#[derive(Clone, Copy, Debug)]
pub struct StaffSpace(pub Px);
impl StaffSpace {
    pub const fn px(self) -> Px {
        self.0
    }
    pub const fn scale(self, percent: SpacePercent) -> Px {
        Px(self.0.0 * percent.0 as i32 / 100)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct StemLine {
    pub x: i32,
    pub left: i32,
    pub right: i32,
    pub y: i32,
    pub len: i32,
}
impl StemLine {
    pub fn end(&self) -> i32 {
        self.y + self.len
    }
}
