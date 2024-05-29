/// Module for testing
#[cfg(test)]
mod random_test {
    use rand::rngs::ThreadRng;

    /// Input data type.
    type Input = ();

    /// Output data type.
    type Output = ();

    /// Perform the specified number of tests.
    #[test]
    fn test() {
        const NUMBER_OF_TESTS: usize = 1000;

        let mut rng = rand::thread_rng();

        for test_case_index in 0..NUMBER_OF_TESTS {
            let input = generator(&mut rng);
            let expected_output = expected(input.clone());
            let actual_output = actual(input.clone());

            assert_eq!(
                expected_output, actual_output,
                "
Wrong Answer on Test #{}

[Input]
{:?}

[Expected output]
{:?}

[Actual output]
{:?}
",
                test_case_index, input, expected_output, actual_output
            );
        }
    }

    /// Generate test cases.
    pub fn generator(_rng: &mut ThreadRng) -> Input {
        todo!()
    }

    /// Return the expected answer.
    pub fn expected(input: Input) -> Output {
        let () = input;

        todo!()
    }

    /// Solution to be tested.
    pub fn actual(input: Input) -> Output {
        let () = input;

        todo!()
    }
}
