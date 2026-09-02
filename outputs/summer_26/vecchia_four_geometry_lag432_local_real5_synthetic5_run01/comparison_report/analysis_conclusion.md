# Local lag432 comparison: analysis conclusion

All 10 data sets and all 40 fits completed successfully.  The three
single-geometry methods use lag budgets 4/3/2.  The union is the exact
deduplicated union of those sets, with maxima 4/9/6.  Every fitted model fixes
the nugget at 0.0000.

## Data used

- Real GEMS TCO: 2024-07-03, 2024-07-05, 2024-07-07, 2024-07-15, 2024-07-22.
- Synthetic smooth-0.5: 2023-07-04, 2023-07-29, 2024-07-13, 2024-07-19,
  2025-07-06.
- The locally available synthetic generator has truth nugget 1.0000, while
  every fitted model fixes the nugget at 0.0000.  Therefore this is an
  intentionally misspecified experiment for all four methods.

## Mean results

| data | method | native NLL | union NLL | eigen D | fit seconds | advection error (grid cells) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| real | adapted | 1.2877 | 1.2875 | 10.5895 | 149.1079 | — |
| real | shifted | 1.2878 | 1.2875 | 10.5890 | 150.1470 | — |
| real | fixed | 1.2913 | 1.2897 | 11.0499 | 154.8293 | — |
| real | union | 1.2875 | 1.2875 | 10.5889 | 656.9671 | — |
| synthetic | adapted | 1.1340 | 1.1332 | 13.1417 | 165.9264 | 0.0562 |
| synthetic | shifted | 1.1339 | 1.1332 | 13.1435 | 160.2841 | 0.0497 |
| synthetic | fixed | 1.1412 | 1.1336 | 13.8274 | 154.8933 | 0.3604 |
| synthetic | union | 1.1332 | 1.1332 | 13.1044 | 708.0569 | 0.0546 |

Lower is better for NLL, eigen D, and truth error.

## Interpretation

1. Advection-aware conditioning is meaningfully better than fixed-center
   conditioning.  Fixed has worse native and union-reference likelihoods,
   larger eigen-curve departure, and a much larger synthetic advection error.
2. Adapted corridor and shifted-center are effectively tied at four-decimal
   likelihood resolution.  Adapted is marginally closer to the union fit on
   the five real dates.  Shifted-center has the smallest synthetic advection
   error on all five synthetic realizations.
3. The union remains a useful reference, but not an efficient default fit.  Its
   mean fitting time is about 4.3--4.4 times the adapted or shifted fit, while
   the union-reference NLL improvement over those two methods is below 0.0001
   on average.
4. On 2024-07-15 the initialized and fitted advection are nearly zero, and all
   four methods have the same NLL to four decimal places.  This is the expected
   negative-control behavior.
5. Do not over-interpret the aggregate seven-parameter truth error in this run:
   fixing the fitted nugget at 0.0000 when the synthetic truth is 1.0000 forces
   variance and range parameters to absorb the mismatch.  The likelihood,
   eigen diagnostic, and advection error are more informative for the requested
   geometry comparison.

The complete four-decimal table is `vecchia_comparison_summary.csv`, and the
parameter-level table is `vecchia_parameter_details.csv`.
