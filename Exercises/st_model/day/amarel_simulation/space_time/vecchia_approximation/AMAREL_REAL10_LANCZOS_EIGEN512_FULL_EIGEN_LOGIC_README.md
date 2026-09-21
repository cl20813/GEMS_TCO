# Amarel: full-eigen-logic diagnostic with 512 Lanczos eigenpairs

## 무엇을 유지하는가

이 실행은 기존 dense full eigendecomposition 진단의 통계량을 바꾸지 않는다.
Fitted Vecchia precision을

\[
\Omega u_j=\omega_j u_j
\]

라고 할 때 각 선택 eigenvector에서

\[
z_j=\sqrt{\omega_j}\,u_j^\top r,
\qquad
e_j=z_j^2=\omega_j(u_j^\top r)^2
\]

를 직접 계산한다. 정확한 Gaussian covariance와 알려진 parameter라면

\[
z_j\sim N(0,1),\qquad e_j\sim\chi_1^2.
\]

512개 방향을 precision eigenvalue가 작은 순서에서 큰 순서로 정렬하여

\[
x_k=\frac{k}{512},
\qquad
y_k=\frac1{512}\sum_{j=1}^k e_j
\]

를 그리고 `y=x`와 비교한다. 기존 dense 코드와 동일하게

\[
D=\frac{\max_k|\sum_{j\le k}e_j-k|}{\sqrt{2\cdot512}}
\]

도 저장한다.

## Lanczos와 SLQ의 제한된 역할

- `eigsh`의 implicitly restarted Lanczos가 실제 full-data eigenvector를 계산한다.
- SLQ는 full precision spectrum의 1/3, 1/2, 2/3 위치를 찾는 용도로만 쓴다.
- SLQ quadratic form으로 residual energy를 계산하지 않는다.
- random start는 eigensolver 초기값일 뿐이며 여러 start의 결과를 평균하지 않는다.
- eigenpair residual은 각 근사방향의 품질지표로 저장하며, 512개 전부에 대한 hard
  pass/fail 조건으로 사용하지 않는다.

선택하는 한 세트는 다음과 같다.

- low-frequency proxy: 가장 작은 precision eigenpair 170개
- middle-frequency proxy: SLQ spectral median에 가장 가까운 eigenpair 170개
- high-frequency proxy: 가장 큰 precision eigenpair 172개

일반 Lanczos는 spectrum 끝에서 먼저 수렴하므로 low는 \(\Omega^{-1}\)의 가장 큰
eigenpair로, high는 \(\Omega\)의 가장 큰 eigenpair로 계산한다. Middle은
\(-\left(\Omega-\sigma I\right)^2\)의 가장 큰 eigenpair를 구한 뒤 그 subspace 안에서
\(\Omega\)를 다시 diagonalize한다.

각 선택 pair에 대해

\[
\frac{\|\Omega\widehat u_j-\widehat\omega_j\widehat u_j\|_2}
{|\widehat\omega_j|}
\]

를 저장한다. `1e-3` 이하는 quality pass로 표시하지만 이를 넘는 mode가 있다고
job을 중단하지 않는다. CSV와 요약에는 median, p95, max residual 및 quality pass
비율이 기록된다.

전체 행렬은

\[
\max|\widehat U^\top\widehat U-I|\le10^{-5}
\]

를 만족해야 한다. NaN/음수 eigenvalue, band별 필요한 mode 수 미확보, 심한 직교성
붕괴는 진단 자체를 정의할 수 없으므로 여전히 중단 조건이다.

경량 테스트 설정은 다음과 같다.

- `eigsh tolerance = 1e-4`
- oversampling `32`
- `eigsh maxiter = 2000`
- SLQ `4 probes × 128 steps`
- ARPACK이 oversampling 후보 전체를 끝내지 못해도 band별 필요 개수를 반환하면 계속 진행

## 해석 범위

이것은 `full eigen decomposition 진단 로직`을 512개의 근사 full-data
eigen-direction에 적용한 것이다. 계산하지 않은 나머지 약 14만 개 방향까지 모두
누적한 all-\(n\) eigendecomposition은 아니다.

또한 low/middle/high는 불규칙한 space-time Vecchia precision에서 precision 크기로
정의한 spectral-frequency proxy이다. 정확한 Fourier frequency라는 뜻은 아니다.

현재 선택은 각 spectral third 전체에서 rank-uniform하게 170개를 뽑는 것이 아니라,
계산 가능한 low extreme, spectral median neighborhood, high extreme의 세 eigenpair
집합이다. Full-eigen score 식은 그대로 사용하지만 eigenpair가 근사이므로 residual
품질도 결과와 함께 해석해야 한다. 선택되지 않은 spectrum 내부 방향에 대한 주장은
하지 않는다.

## 날짜와 출력

각 연도에서 July 3, 5, 7, 12, 15를 사용한다.

- 2024: 5일
- 2025: 5일
- adapted/fixed: 각 날짜에서 모두 실행

주요 출력:

- `daily_subplots/*_nll_lanczos_eigen512_fulleigen_logic.png`
- `daily_band_segments/*_eigen512_exact_band_segments.png`
- `2024_07_selected5_average_nll_lanczos_eigen512_fulleigen_logic.png`
- `2025_07_selected5_average_nll_lanczos_eigen512_fulleigen_logic.png`
- `daily_approximate_eigen512_curves.csv`
- `daily_nll_eigen512_metrics.csv`
- `daily_slq_boundaries_only.csv`

Band segment 그림은 별도 평균이나 endpoint 재정규화를 하지 않는다. 세 패널은 동일한
전체 누적곡선의 해당 부분을 전역 x/y 좌표 그대로 잘라 보여준다.

기본적으로 eigenvector 자체는 출력 용량을 줄이기 위해 저장하지 않는다. 실제 계산에는
명시적인 \(n\times512\) eigenvector 행렬이 사용된다. 저장이 필요하면 실행 옵션에
`--retain-eigenvectors`를 추가한다. float32 `.npy` 파일이 날짜·방법마다 약 0.3 GB일
수 있고 전체 다운로드가 커진다.

## 업로드와 제출

로컬에서:

```bash
cd '/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation'
bash scp_vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_lag643.sh
```

업로드가 끝나면:

```bash
ssh jl2815@amarel-new.hpc.rutgers.edu \
'cd /home/jl2815/tco/exercise_25/st_model/day/amarel_simulation/space_time/vecchia_approximation && sbatch slurm_vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_lag643.sh'
```

상태 확인:

```bash
ssh jl2815@amarel-new.hpc.rutgers.edu 'squeue -u jl2815'
```

완료 후 다운로드:

```bash
scp -r \
jl2815@amarel-new.hpc.rutgers.edu:/home/jl2815/tco/exercise_output/summer/vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_light_tol1e4_lag643 \
'/Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/amarel_simulation/space_time/vecchia_approximation/'
```

8시간 안에 끝나지 않으면 같은 SLURM 파일을 다시 제출한다. 완료된 date/method cache는
재사용된다. 이 경량 버전은 기존 엄격 실행과 다른 새 output root를 사용하므로 이전의
불완전 cache/configuration과 섞이지 않는다.
