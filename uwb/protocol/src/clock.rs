//! Bounded affine clock correlation for mission and radio time domains.

pub const CLOCK_SAMPLE_CAPACITY: usize = 32;
pub const MAPPING_MAX_ERROR_US: u32 = 750;
pub const INITIAL_CLOCK_DRIFT_PPM: u32 = 50;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CorrelationSample {
    pub source: u64,
    pub target: u64,
    pub error_us: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClockSampleError {
    NonMonotonic,
    ErrorBound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffineClockModel {
    source_anchor: u64,
    target_anchor: u64,
    target_ticks_per_source_q32: u64,
    error_at_anchor_us: u32,
    drift_ppm: u32,
    generation: u32,
}

impl AffineClockModel {
    pub const fn from_sample(
        sample: CorrelationSample,
        nominal_target_ticks_per_source_q32: u64,
        drift_ppm: u32,
    ) -> Self {
        Self {
            source_anchor: sample.source,
            target_anchor: sample.target,
            target_ticks_per_source_q32: nominal_target_ticks_per_source_q32,
            error_at_anchor_us: sample.error_us,
            drift_ppm,
            generation: sample.generation,
        }
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub const fn anchor_source(self) -> u64 {
        self.source_anchor
    }

    pub fn target_at(self, source: u64) -> Option<u64> {
        if source >= self.source_anchor {
            let delta = (u128::from(source - self.source_anchor)
                * u128::from(self.target_ticks_per_source_q32))
                >> 32;
            self.target_anchor.checked_add(delta.try_into().ok()?)
        } else {
            let delta = (u128::from(self.source_anchor - source)
                * u128::from(self.target_ticks_per_source_q32))
                >> 32;
            self.target_anchor.checked_sub(delta.try_into().ok()?)
        }
    }

    pub fn source_at(self, target: u64) -> Option<u64> {
        let rate = u128::from(self.target_ticks_per_source_q32);
        if target >= self.target_anchor {
            let delta = ((u128::from(target - self.target_anchor) << 32) + rate / 2) / rate;
            self.source_anchor.checked_add(delta.try_into().ok()?)
        } else {
            let delta = ((u128::from(self.target_anchor - target) << 32) + rate / 2) / rate;
            self.source_anchor.checked_sub(delta.try_into().ok()?)
        }
    }

    pub fn error_at(self, source: u64) -> u32 {
        let age = source.abs_diff(self.source_anchor);
        let growth = u128::from(age) * u128::from(self.drift_ppm) / 1_000_000;
        self.error_at_anchor_us
            .saturating_add(growth.min(u128::from(u32::MAX)) as u32)
    }

    pub fn maps_at_source(self, source_time: u64) -> bool {
        self.error_at(source_time) <= MAPPING_MAX_ERROR_US
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClockCorrelator<const N: usize> {
    samples: [Option<CorrelationSample>; N],
    next: usize,
    count: usize,
    generation: Option<u32>,
    nominal_target_ticks_per_source_q32: u64,
    drift_ppm: u32,
}

impl<const N: usize> ClockCorrelator<N> {
    pub const fn new(nominal_target_ticks_per_source_q32: u64) -> Self {
        Self {
            samples: [None; N],
            next: 0,
            count: 0,
            generation: None,
            nominal_target_ticks_per_source_q32,
            drift_ppm: INITIAL_CLOCK_DRIFT_PPM,
        }
    }

    pub fn clear(&mut self) {
        self.samples = [None; N];
        self.next = 0;
        self.count = 0;
        self.generation = None;
    }

    pub fn push(&mut self, sample: CorrelationSample) -> Result<(), ClockSampleError> {
        if sample.error_us > MAPPING_MAX_ERROR_US {
            return Err(ClockSampleError::ErrorBound);
        }
        if self
            .generation
            .is_some_and(|generation| generation != sample.generation)
        {
            self.clear();
        }
        if self
            .samples
            .into_iter()
            .flatten()
            .any(|current| sample.source <= current.source || sample.target <= current.target)
        {
            return Err(ClockSampleError::NonMonotonic);
        }
        self.generation = Some(sample.generation);
        self.samples[self.next] = Some(sample);
        self.next = (self.next + 1) % N;
        self.count = (self.count + 1).min(N);
        Ok(())
    }

    pub fn model(self) -> Option<AffineClockModel> {
        let newest_source = self
            .samples
            .into_iter()
            .flatten()
            .map(|sample| sample.source)
            .max()?;
        let mut best = None;
        let mut best_error = u128::MAX;
        for sample in self.samples.into_iter().flatten() {
            let age = newest_source - sample.source;
            let error = u128::from(sample.error_us)
                + u128::from(age) * u128::from(self.drift_ppm) / 1_000_000;
            if error < best_error
                || (error == best_error
                    && best.is_none_or(|current: CorrelationSample| sample.source > current.source))
            {
                best = Some(sample);
                best_error = error;
            }
        }
        Some(AffineClockModel::from_sample(
            best?,
            self.nominal_target_ticks_per_source_q32,
            self.drift_ppm,
        ))
    }
}

pub const fn q32_rate(integer_ticks_per_source: u64) -> u64 {
    integer_ticks_per_source << 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_maps_both_directions_and_grows_its_error() {
        let sample = CorrelationSample {
            source: 2_000_000,
            target: 11_000_000,
            error_us: 50,
            generation: 7,
        };
        let model = AffineClockModel::from_sample(sample, q32_rate(1), 50);
        assert_eq!(model.target_at(2_100_000), Some(11_100_000));
        assert_eq!(model.source_at(11_100_000), Some(2_100_000));
        assert_eq!(model.error_at(2_000_000), 50);
        assert_eq!(model.error_at(12_000_000), 550);
        assert!(!model.maps_at_source(17_000_000));
    }

    #[test]
    fn correlator_rejects_bad_samples_and_resets_on_generation_change() {
        let mut correlator = ClockCorrelator::<4>::new(q32_rate(1));
        assert_eq!(
            correlator.push(CorrelationSample {
                error_us: MAPPING_MAX_ERROR_US + 1,
                ..CorrelationSample::default()
            }),
            Err(ClockSampleError::ErrorBound),
        );
        correlator
            .push(CorrelationSample {
                source: 1,
                target: 2,
                error_us: 1,
                generation: 1,
            })
            .unwrap();
        correlator
            .push(CorrelationSample {
                source: 1_000_001,
                target: 1_000_002,
                error_us: 1,
                generation: 2,
            })
            .unwrap();
        assert_eq!(correlator.model().unwrap().generation(), 2);
    }

    #[test]
    fn correlator_rejects_non_monotonic_samples() {
        let mut correlator = ClockCorrelator::<4>::new(q32_rate(1));
        let first = CorrelationSample {
            source: 1,
            target: 10,
            error_us: 1,
            generation: 1,
        };
        correlator.push(first).unwrap();
        assert_eq!(
            correlator.push(CorrelationSample {
                source: 2,
                target: 9,
                ..first
            }),
            Err(ClockSampleError::NonMonotonic),
        );
    }
}
