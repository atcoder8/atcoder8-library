/// Module for testing
#[cfg(test)]
mod random_test {
    /// Input data type
    type Input = ();

    /// Output data type
    type Output = ();

    /// Perform the specified number of tests.
    #[test]
    fn test() {
        const NUMBER_OF_TESTS: usize = 1000;

        for test_case_index in 0..NUMBER_OF_TESTS {
            let input = generator();
            let jury_output = jury(input.clone());
            let solve_output = solve(input.clone());

            assert_eq!(
                jury_output, solve_output,
                "
Wrong Answer on Test #{}

[Input]
{:?}

[Output(Jury)]
{:?}

[Output(Solve)]
{:?}
",
                test_case_index, input, jury_output, solve_output
            );
        }
    }

    /// Generate test cases.
    pub fn generator() -> Input {
        // let mut rng = rand::thread_rng();

        ()
    }

    /// Returns the correct answer.
    pub fn jury(input: Input) -> Output {
        let () = input;

        ()
    }

    /// Test this program.
    pub fn solve(input: Input) -> Output {
        let () = input;

        ()
    }
}
