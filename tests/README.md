# SandoSeer MEV Oracle Tests

This directory contains integration tests for the SandoSeer MEV Oracle.

## Test Structure

- **Unit Tests**: Located within the module files under `src/` with the `#[cfg(test)]` attribute
- **Integration Tests**: Located in this directory, these tests verify the interactions between components

## Test Categories

### MEV Strategy Execution Tests

Tests for verifying the correct functioning of the three MEV strategies:

1. **Arbitrage**: Tests the ability to identify and execute profitable token swaps across multiple DEXes
2. **Sandwich**: Tests front-running and back-running large transactions to profit from price impacts
3. **Token Snipe**: Tests rapidly buying newly listed tokens before market prices adjust

### Transaction Simulation Tests

Tests for verifying our transaction simulation system:

1. **Profitability Estimation**: Verifying that profit calculations are accurate
2. **Gas Estimation**: Ensuring gas costs are correctly estimated
3. **Safety Checks**: Verifying that safety checks prevent unauthorized transactions

### Opportunity Evaluation Tests

Tests for verifying our opportunity evaluation system:

1. **Scoring**: Tests that opportunity scoring correctly prioritizes the best opportunities
2. **Decision Making**: Tests that execution decisions are made properly based on thresholds

## Running Tests

Run all tests (both unit and integration):

```bash
cargo test
```

Run just integration tests:

```bash
cargo test --test '*'
```

Run a specific test:

```bash
cargo test --test integration_tests test_full_execution_pipeline
```

Run tests with logging enabled:

```bash
RUST_LOG=debug cargo test -- --nocapture
```

## Adding New Tests

When adding new tests, please follow these guidelines:

1. **Unit Tests**: Add to the respective module files
2. **Integration Tests**: Add to this directory
3. **Naming Convention**: Use descriptive test names with the `test_` prefix
4. **Documentation**: Add comments explaining the purpose of each test
5. **Independence**: Tests should be independent and not rely on the order of execution
6. **Mocking**: Use simulation mode for tests that would otherwise interact with the blockchain 