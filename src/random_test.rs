/// Module for random testing.
#[cfg(test)]
mod random_test {
    use rand::prelude::*;

    /// Input data type.
    type Input = ();

    /// Output data type.
    type Output = ();

    /// Performs the specified number of tests.
    #[test]
    fn test() {
        /// This value specifies the number of tests.
        const NUMBER_OF_TESTS: usize = 1000;

        let mut rng = rand::rng();

        for test_case_index in 0..NUMBER_OF_TESTS {
            let input = generator(&mut rng);
            let expected_output = expected(input.clone());
            let actual_output = actual(input.clone());

            // If an unexpected answer is returned, panic is triggered.
            assert_eq!(
                expected_output, actual_output,
                "
Unexpected answer was returned in test case #{}.

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

    /// Generates a test case.
    pub fn generator(_rng: &mut ThreadRng) -> Input {
        todo!()
    }

    /// Returns the expected answer.
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
