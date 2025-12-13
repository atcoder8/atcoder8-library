//! 数論変換を用いて畳み込みを行います。
//! [modint2](https://github.com/atcoder8/atcoder8-library/blob/main/src/modint2.rs)に依存しています。

use crate::modint2::Modint998244353;

type Mint = Modint998244353;

/// バタフライ演算が可能な回数の最大値
/// (3^119)^(2^k) ≡ 1 (mod 998244353) を満たす最小の非負整数kと一致する
const MAX_EXP: usize = 23;

/// k番目(0-based)の要素: (3^119)^(2^k) mod 998244353 (998244353の原始根)
const PRIMITIVE_ROOTS: [u32; MAX_EXP] = [
    15311432, 267099868, 733596141, 565042129, 363395222, 996173970, 24514907, 629671588,
    968855178, 666702199, 350007156, 63912897, 584193783, 258648936, 166035806, 476477967,
    781712469, 922799308, 452798380, 929031873, 372528824, 911660635, 998244352,
];

/// k番目(0-based)の要素: (3^119)^(-2^k) mod 998244353 (998244353の原始根の逆元)
const INVERSE_PRIMITIVE_ROOTS: [u32; MAX_EXP] = [
    469870224, 382752275, 428961804, 950391366, 704923114, 121392023, 3707709, 283043518,
    708402881, 814576206, 358024708, 129292727, 335559352, 381598368, 685443576, 304459705,
    135236158, 609441965, 87557064, 337190230, 509520358, 86583718, 998244352,
];

/// `ceil(log2(x))`を計算します。
fn ceil_log2(x: usize) -> usize {
    let mut i = 0;
    while (1 << i) < x {
        i += 1;
    }
    i
}

/// 数論変換を行います。
/// `inverse`が`true`である場合は逆変換を行います。
fn ntt(a: &mut [Mint], inverse: bool) {
    let n = a.len();
    let h = ceil_log2(n);

    assert_eq!(n, 1 << h, "配列の長さは2の冪乗である必要があります。");
    assert!(
        h < MAX_EXP,
        "配列の長さは2^{MAX_EXP}以下である必要があります。"
    );

    // 配列の要素をバタフライ演算用に再配置
    for i in 0..a.len() {
        let j: usize = (0..h).map(|k| ((i >> k) & 1) << (h - 1 - k)).sum();
        if i < j {
            a.swap(i, j);
        }
    }

    // 998244353に対する原始根、またはその逆元を格納した配列を取得
    let roots = if inverse {
        INVERSE_PRIMITIVE_ROOTS
    } else {
        PRIMITIVE_ROOTS
    };

    // バタフライ演算
    for i in 0..h {
        let sub_size = 1 << i;
        let root = roots[MAX_EXP - 1 - i];
        let mut w = Mint::new(1);
        for j in 0..sub_size {
            for k in (0..n).step_by(2 * sub_size) {
                let s = a[k + j];
                let t = a[k + sub_size + j] * w;
                a[k + j] = s + t;
                a[k + sub_size + j] = s - t;
            }
            w *= root;
        }
    }

    // 逆変換である場合は各要素を`n`で除算
    if inverse {
        let inv_n = Mint::new(n).inv();
        a.iter_mut().for_each(|x| *x *= inv_n);
    }
}

/// 数論変換を用いて畳み込みを行います。
pub fn convolution(a: &[Mint], b: &[Mint]) -> Vec<Mint> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let s = a.len() + b.len() - 1;
    let n = 1 << ceil_log2(s);

    let mut a = a.to_vec();
    a.resize(n, Mint::new(0));
    ntt(&mut a, false);

    let mut b = b.to_vec();
    b.resize(n, Mint::new(0));
    ntt(&mut b, false);

    for i in 0..n {
        a[i] *= b[i];
    }

    ntt(&mut a, true);

    a.truncate(s);

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_expected_result(a: &[Mint], b: &[Mint]) -> Vec<Mint> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }

        let s = a.len() + b.len() - 1;
        let mut convolution_result = vec![Mint::new(0); s];
        for (i, &a) in a.iter().enumerate() {
            for (j, &b) in b.iter().enumerate() {
                convolution_result[i + j] += a * b;
            }
        }
        convolution_result
    }

    #[test]
    fn test_empty() {
        assert_eq!(convolution(&vec![], &vec![]), vec![]);
        assert_eq!(convolution(&vec![], &vec![Mint::new(3)]), vec![]);
        assert_eq!(convolution(&vec![Mint::new(3)], &vec![]), vec![]);
    }

    #[test]
    fn test_small() {
        let a = [3, 1, 4, -1, 5, -9, -2].map(Mint::new);
        let b = [6, 5, -3, 5].map(Mint::new);
        let result = convolution(&a, &b);
        let expected = compute_expected_result(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_large() {
        let a = [
            202079577, 351294914, 437885408, 733221123, 674063207, 208865746, 9194690, 632953215,
            414637094, 437432094,
        ]
        .map(Mint::new);
        let b = [
            379355117, 445654482, 367501804, 516023492, 731206818, 319973157, 979750425, 261899795,
            685065784, 701638389,
        ]
        .map(Mint::new);
        let result = convolution(&a, &b);
        let expected = compute_expected_result(&a, &b);
        assert_eq!(result, expected);
    }
}
