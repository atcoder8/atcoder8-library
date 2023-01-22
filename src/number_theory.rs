pub fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();

    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }

    a
}

pub fn lcm(a: i64, b: i64) -> i64 {
    a / gcd(a, b) * b
}

pub fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut s, mut t, mut u, mut v) = (1, 0, 0, 1);
    let (mut a, mut b) = (a, b);

    while b != 0 {
        let (q, r) = (a / b, a % b);

        std::mem::swap(&mut s, &mut t);
        t -= q * s;
        std::mem::swap(&mut u, &mut v);
        v -= q * u;

        a = b;
        b = r;
    }

    if a >= 0 {
        (s, u, a)
    } else {
        (-s, -u, -a)
    }
}

pub fn crt(rr: &Vec<i64>, mm: &Vec<i64>) -> Option<(i64, i64)> {
    assert_eq!(rr.len(), mm.len());
    assert!(!rr.is_empty());
    assert!(mm.iter().all(|&m| m >= 1));

    let mut acc_r = 0;
    let mut acc_m = 1;

    for (&r, &m) in rr.iter().zip(mm.iter()) {
        let diff_r = r - acc_r;

        let (x, _y, d) = ext_gcd(acc_m, m);

        if diff_r % d != 0 {
            return None;
        }

        let next_acc_m = acc_m / d * m;
        acc_r = (acc_m * diff_r / d * x + acc_r) % next_acc_m;
        acc_m = next_acc_m;
    }

    if acc_r < 0 {
        acc_r += acc_m;
    }

    Some((acc_r, acc_m))
}
