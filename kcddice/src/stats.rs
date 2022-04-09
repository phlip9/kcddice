use claim::{debug_assert_ge, debug_assert_le};
use ndarray::{ArrayView1, Zip};
use statrs::distribution::{ChiSquared, ContinuousCDF};

const EPS: f64 = 1e-10;

/// Given `X_1` and `X_2` independent random variables defined by CDFs `cdf1` and
/// `cdf2` respectively, returns the `Pr[X_1 <= X_2]`. Both "CDF" arrays must be
/// the same length and have the same support at each index.
pub(crate) fn p_rv1_le_rv2(cdf1: ArrayView1<f64>, cdf2: ArrayView1<f64>) -> f64 {
    assert_eq!(cdf1.len(), cdf2.len());

    // p = ∑_{x_i} Pr[X_1 = x_i] * Pr[X_2 >= x_i]
    let p = 0.0;

    // c1_i1 = cdf1[i - 1] = Pr[X_1 <= prev(x_i)]
    let c1_i1 = 0.0;

    // c2_i1 = cdf2[i - 1] = Pr[X_2 <= prev(x_i)]
    let c2_i1 = 0.0;

    let (p, _c1_i1, _c2_i2) = cdf1.into_iter().zip(cdf2.into_iter()).fold(
        (p, c1_i1, c2_i1),
        |(p, c1_i1, c2_i1), (&c1_i, &c2_i)| {
            // c1_i = cdf1[i] = Pr[X_1 <= x_i]
            // c2_i = cdf2[i] = Pr[X_2 <= x_i]

            // p_1 = Pr[X_1 = x_i]
            //     = Pr[X_1 <= x_i] - Pr[X_1 <= prev(x_i)]
            //     = cdf1[i] - cdf1[i - 1]
            let p_1 = c1_i - c1_i1;

            // p_2 = Pr[X_2 >= x_i]
            //     = 1 - Pr[X_2 < x_i]
            //     = 1 - Pr[X_2 <= prev(x_i)]
            //     = 1 - cdf2[i - 1]
            let p_2 = 1.0 - c2_i1;

            (p + (p_1 * p_2), c1_i, c2_i)
        },
    );

    p
}

/// Return true iff `supp(p) ⊆ supp(q)` for dense PMFs `p` and `q`.
pub(crate) fn is_pmf_subset(p: ArrayView1<f64>, q: ArrayView1<f64>) -> bool {
    Zip::from(p).and(q).all(|&p_i, &q_i| {
        // A = (q_i == 0.0)
        // B = (p_i == 0.0)
        // (A ==> B) <==> (¬A ∨ B)
        (q_i > 0.0) || (p_i <= 0.0)
    })
}

/// Compute the [KL-divergence](https://www.wikiwand.com/en/Kullback%E2%80%93Leibler_divergence).
/// between dense PMFs `p` and `q`.
///
/// `D_{KL}(p || q) = \sum_i p_i * \ln(p_i / q_i)`
///
/// Note: p's support must be a strict subset of q's support!
///       this is also referred to as the "absolute continuity" assumption,
///       where `q_i = 0` implies `p_i = 0`.
pub(crate) fn kl_divergence(p: ArrayView1<f64>, q: ArrayView1<f64>) -> f64 {
    // caller should check this before
    debug_assert!(is_pmf_subset(p, q));

    Zip::from(p)
        .and(q)
        .fold(0.0, |sum, &p_i, &q_i| sum + kl_div_term(p_i, q_i))
}

#[inline]
pub(crate) fn kl_div_term(p_i: f64, q_i: f64) -> f64 {
    if q_i > EPS {
        if p_i > EPS {
            p_i * (p_i / q_i).ln()
        } else {
            0.0
        }
    } else {
        if p_i > EPS {
            debug_assert!(false);
            f64::INFINITY
        } else {
            0.0
        }
    }
}

/// The G-test statistic.
///
/// * Used for comparing observed multinomial distribution with expected
///   hypothesis multinomial distribution.
/// * Asmyptotically approximates chi^2-test statistic.
///
/// `n`: the number of samples
/// `p`: the expected PMF
/// `p_hat`: the observed PMF
///
/// G-test: https://www.wikiwand.com/en/G-test
pub(crate) fn g_test(n: usize, p: ArrayView1<f64>, p_hat: ArrayView1<f64>) -> f64 {
    (n as f64) * (2.0 * kl_divergence(p_hat, p))
}

// pub(crate) fn chisq_test(counts_exp: ArrayView1<f64>, counts_obs: ArrayView1<f64>) -> f64 {
//     Zip::from(counts_exp)
//         .and(counts_obs)
//         .fold(0.0, |sum, &e_i, &o_i| sum + chisq_test_term(e_i, o_i))
// }
//
// #[inline]
// pub(crate) fn chisq_test_term(e_i: f64, o_i: f64) -> f64 {
//     if e_i > EPS {
//         let diff = o_i - e_i;
//         ((diff * diff) / e_i)
//     } else {
//         0.0
//     }
// }

/// The CDF of the Chi^2-distribution, where `dof` is the
/// "degrees-of-freedom" parameter and `x ∈ R`.
pub(crate) fn chisq_cdf(dof: f64, x: f64) -> f64 {
    ChiSquared::new(dof).unwrap().cdf(x)
}

/// A goodness-of-fit test between a hypothesized multinomial distribution, `p`,
/// and an experimentally observed distribution, `p_hat`, both represented as
/// dense PMFs. `n` is the number of samples taken to construct `p_hat`.
///
/// Returns a p-value, `Pr[G(x) >= p-value | H_0: x ~ p]`.
pub(crate) fn multinomial_test(n: usize, p: ArrayView1<f64>, p_hat: ArrayView1<f64>) -> f64 {
    // want to compute the DOF (nnz of p)
    let nnz = p.fold(0.0, |nnz, &x| nnz + if x > 0.0 { 1.0 } else { 0.0 });
    let dof = nnz - 1.0;

    debug_assert_le!(nnz, p.dim() as f64);
    debug_assert_ge!(dof, 1.0);

    // impossible to draw p_hat from p
    if !is_pmf_subset(p_hat, p) {
        return 0.0;
    }

    let g = g_test(n, p, p_hat);
    let pvalue = 1.0 - chisq_cdf(dof, g);

    println!(
        "multinomial_test: n: {n}, |p|: {}, dof: {dof}, g: {g}, p-value: {pvalue}",
        p.dim()
    );

    pvalue
}

// // using chi^2 test
// pub(crate) fn multinomial_test2(
//     n: usize,
//     counts_exp: ArrayView1<f64>,
//     counts_obs: ArrayView1<f64>,
// ) -> f64 {
//     // want to compute the DOF (nnz of p)
//     let nnz = counts_exp.fold(0.0, |nnz, &x| nnz + if x > EPS { 1.0 } else { 0.0 });
//     let dof = nnz - 1.0;
//
//     debug_assert_le!(nnz, counts_exp.dim() as f64);
//     debug_assert_ge!(dof, 1.0);
//
//     // impossible to draw
//     if !is_pmf_subset(counts_obs, counts_exp) {
//         return 0.0;
//     }
//
//     let chisq = chisq_test(counts_exp, counts_obs);
//     let pvalue = 1.0 - chisq_cdf(dof, chisq);
//
//     println!(
//         "multinomial_test: n: {n}, |counts_exp|: {}, dof: {dof}, chi^2: {chisq}, p-value: {pvalue}",
//         counts_exp.dim()
//     );
//
//     pvalue
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        dice::{DieDistr, DieKind},
        prop::small_rng,
    };
    use approx::assert_relative_eq;
    use claim::{assert_gt, assert_lt};
    use ndarray::{array, Array1};
    use rand::distributions::Distribution;
    use tabular::{row, Table};

    /// Given `X_1` and `X_2` independent random variables defined by the same CDF
    /// `cdf`, returns the `Pr[X_1 <= X_2]`.
    fn p_rv_le_itself(cdf: ArrayView1<f64>) -> f64 {
        // p = ∑_{x_i} Pr[X_1 = x_i] * Pr[X_2 >= x_i]
        let p = 0.0;

        // c_i1 = Pr[X <= prev(x_i)] = cdf[i - 1]
        let c_i1 = 0.0;

        let (p, _c_i1) = cdf.into_iter().fold((p, c_i1), |(p, c_i1), &c_i| {
            // c_i = cdf[i]

            // p_1 = Pr[X_1 = x_i]
            //     = Pr[X_1 <= x_i] - Pr[X_1 <= prev(x_i)]
            //     = cdf[i] - cdf[i - 1]
            let p_1 = c_i - c_i1;

            // p_2 = Pr[X_2 >= x_i]
            //     = 1 - Pr[X_2 < x_i]
            //     = 1 - Pr[X_2 <= prev(x_i)]
            //     = 1 - cdf[i - 1]
            let p_2 = 1.0 - c_i1;

            (p + (p_1 * p_2), c_i)
        });

        p
    }

    #[test]
    fn test_p_rv_le_itself() {
        let cdf = Array1::from_vec(vec![0.1, 0.6, 1.0]);
        assert_relative_eq!(0.71, p_rv_le_itself(cdf.view()));
    }

    #[test]
    fn test_p_rv1_le_rv2() {
        let cdf1 = Array1::from_vec(vec![0.1, 0.6, 1.0]);
        let cdf2 = Array1::from_vec(vec![0.3, 0.7, 1.0]);

        assert_relative_eq!(
            p_rv_le_itself(cdf1.view()),
            p_rv1_le_rv2(cdf1.view(), cdf1.view())
        );

        assert_relative_eq!(
            p_rv_le_itself(cdf2.view()),
            p_rv1_le_rv2(cdf2.view(), cdf2.view())
        );

        assert_relative_eq!(0.57, p_rv1_le_rv2(cdf1.view(), cdf2.view()));
    }

    #[test]
    fn test_is_pmf_subset() {
        let p = array![0.0, 0.0, 1.0];
        let q = array![0.0, 1.0, 1.0];

        assert!(is_pmf_subset(p.view(), p.view()));
        assert!(is_pmf_subset(q.view(), q.view()));
        assert!(is_pmf_subset(p.view(), q.view()));
        assert!(!is_pmf_subset(q.view(), p.view()));
    }

    #[test]
    fn test_kl_divergence() {
        let p = array![0.1, 0.3, 0.6];
        let q = array![0.3, 0.3, 0.4];

        assert_relative_eq!(0.0_f64, kl_divergence(p.view(), p.view()));
        assert_relative_eq!(0.0_f64, kl_divergence(q.view(), q.view()));

        // D_KL(p || q) = (0.1 * ln(0.1 / 0.3))
        //              + (0.3 * ln(0.3 / 0.3))
        //              + (0.6 * ln(0.6 / 0.4))
        //              = -0.10986122886681096
        //              + 0.0
        //              + 0.24327906486489853
        //              = 0.13341783599808757
        assert_relative_eq!(0.13341783599808757_f64, kl_divergence(p.view(), q.view()));

        // D_KL(q || p) = (0.3 * ln(0.3 / 0.1))
        //              + (0.3 * ln(0.3 / 0.3))
        //              + (0.4 * ln(0.4 / 0.6))
        //              = 0.32958368660043286
        //              + 0.0
        //              + -0.16218604324326572
        //              = 0.16739764335716714
        assert_relative_eq!(0.16739764335716714_f64, kl_divergence(q.view(), p.view()));
    }

    #[test]
    fn test_multinomial_test() {
        let mut rng = small_rng(0xd15c0);
        let distr = DieKind::Lucky.die_distr();
        let p = distr.clone().into_pmf();
        let p = Array1::from_vec(p.to_vec());

        let p2 = Array1::from_vec(DieKind::Biased.die_distr().into_pmf().to_vec());

        for n in [20, 100, 1_000, 10_000] {
            let counts_exp = &p * (n as f64);
            let counts_exp2 = &p2 * (n as f64);

            let face_samples = distr.clone().sample_iter(&mut rng).take(n);

            let mut face_counts = [0_usize; 6];
            for face in face_samples {
                face_counts[(face as usize) - 1] += 1;
            }

            let counts_obs =
                Array1::from_vec(face_counts.clone().map(|count| count as f64).to_vec());

            let p_hat = face_counts.map(|count| (count as f64) / (n as f64));
            let p_hat = Array1::from_vec(p_hat.to_vec());

            let mut table = Table::new("{:>}  {:<}  {:<}  {:<}  {:<}  {:<}  {:<}").with_row(row!(
                "face", "p", "p_hat", "|diff|", "p2", "p_hat", "|diff2|"
            ));

            for face in 0..=5 {
                table.add_row(row!(
                    (face + 1),
                    p[face],
                    p_hat[face],
                    (p[face] - p_hat[face]).abs(),
                    p2[face],
                    p_hat[face],
                    (p2[face] - p_hat[face]).abs(),
                ));
            }

            println!("\nn = {n}");
            let pvalue_1 = multinomial_test(n, p.view(), p_hat.view());
            println!("p-value 1 = {pvalue_1}");
            let pvalue_2 = multinomial_test(n, p2.view(), p_hat.view());
            println!("p-value 2 = {pvalue_2}");

            let pvalue2_1 = multinomial_test2(n, counts_exp.view(), counts_obs.view());
            let pvalue2_2 = multinomial_test2(n, counts_exp2.view(), counts_obs.view());
            println!("p-value 2.1 = {pvalue2_1}");
            println!("p-value 2.2 = {pvalue2_2}");

            println!("{table}");

            assert_gt!(pvalue_1, 0.01);
            assert_lt!(pvalue_2, 0.01);

            assert_gt!(pvalue2_1, 0.01);
            assert_lt!(pvalue2_2, 0.01);
        }
    }
}
