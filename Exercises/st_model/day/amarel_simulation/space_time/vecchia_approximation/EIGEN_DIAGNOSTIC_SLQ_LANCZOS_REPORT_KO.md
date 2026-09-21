# Full-data eigen diagnostic: SLQ + Lanczos/Ritz

## 1. 목적

하루에 시간당 약 18,000개, 8시간이면 약 144,000개의 관측값이 있다.
이 크기의 covariance 또는 precision matrix를 dense matrix로 만들고 전체
eigendecomposition을 수행하는 것은 메모리와 계산량 측면에서 비현실적이다.

기존 `400 x 8` 진단은 선택된 3,200개 위치에서 exact dense eigenanalysis를
수행했지만, 표본의 공간 간격보다 짧은 고주파 구조를 충분히 대표하지 못한다.
따라서 최종 비교에서는 이 진단을 제외하고 다음 두 기준을 사용한다.

1. 전체 자료의 native Vecchia NLL
2. 전체 자료의 sparse precision operator를 이용한 SLQ + Lanczos/Ritz 진단

## 2. Vecchia precision을 matrix-free로 적용

Vecchia 조건부분포를 쌓으면 sparse whitener `B`를 구성할 수 있다. 전체
precision matrix를 직접 저장하지 않고 다음 연산만 수행한다.

```text
Omega v = B' (B v)
```

따라서 약 `144,000 x 144,000` dense matrix, 모든 eigenvector 저장 및
`O(n^3)` full eigendecomposition이 필요하지 않다.

## 3. SLQ의 역할: 전체 spectrum과 band 경계

SLQ(stochastic Lanczos quadrature)는 Rademacher random probe를 사용하여
precision spectrum의 empirical CDF, 즉 eigenvalue threshold 이하에 전체
mode가 몇 개 있는지를 추정한다.

SLQ의 역할은 eigenvector를 구하는 것이 아니다. 전체 mode-count spectrum의
`1/3` 및 `2/3` quantile을 구하여 다음 세 band의 경계를 정하는 것이다.

- low: 작은 precision eigenvalue의 하위 1/3
- middle: 중간 1/3
- high: 큰 precision eigenvalue의 상위 1/3

Precision eigenvalue를 `omega`, covariance eigenvalue를 `lambda`라 하면
`omega=1/lambda`이다. 따라서 precision을 작은 값에서 큰 값으로 정렬하는
것은 covariance eigenvalue를 큰 값에서 작은 값으로 정렬하는 것과 같다.
Smooth covariance model에서는 이를 저주파에서 고주파로 가는 spectral proxy로
해석한다.

다만 불규칙한 space-time Vecchia graph에서 이 구분은 정확한 Fourier
frequency가 아니다. 물리적인 공간 파장이라고 주장하려면 eigenvector의 graph
roughness 또는 별도의 spatial wavenumber를 확인해야 한다.

## 4. Lanczos/Ritz의 역할: 대표 eigen-direction 근사

SLQ가 정한 세 band를 기준으로 별도의 random-start Lanczos run을 수행한다.
`m=1536` Krylov candidate space에서 spectral quantile에 골고루 위치한 implicit
Ritz mode를 선택한다.

```text
low:    170
middle: 170
high:   172
total:  512
```

이 512개는 전체 약 144,000개 eigenvector 중 spectral rank가 골고루 퍼진
대표 근사 방향이다. Exact eigenvector라고 주장하지 않으며, 각 Ritz pair의
tail residual을 저장하여 근사 품질을 평가한다.

## 5. Residual energy

Fitted mean을 제거한 residual을 `r`이라 하자. Covariance eigenpair가

```text
Sigma u_j = lambda_j u_j
```

이면 해당 eigen-direction의 whitened score는

```text
z_j = (u_j' r) / sqrt(lambda_j)
```

이고 진단에 사용하는 energy는 그 제곱이다.

```text
e_j = z_j^2 = (u_j' r)^2 / lambda_j
```

구현에서는 precision Ritz eigenvalue `omega_j=1/lambda_j`를 사용하므로
동일한 값을 다음과 같이 계산한다.

```text
e_j = omega_j * (u_j' r)^2
```

`u_j' r`은 eigenvector와 residual의 element-wise 곱이 아니라 scalar inner
product, 즉 residual을 eigenvector 방향으로 projection한 값이다. 올바른
모델과 exact whitening 아래에서는 `E[e_j]=1`이다.

## 6. 누적 y=x 진단

각 band는 SLQ가 추정한 전체 mode의 정확히 1/3을 대표한다. 따라서 선택된
Ritz mode의 가중치는 다음과 같다.

```text
low/middle: 1 / (3 * 170)
high:       1 / (3 * 172)
```

Precision eigenvalue가 작은 순서에서 큰 순서로 정렬한 뒤 다음을 계산한다.

```text
x_k = sum_{j<=k} w_j
y_k = sum_{j<=k} w_j e_j
```

올바른 spectral calibration에서는 `E[y_k]=x_k`이므로 누적곡선을 대각선
`y=x`와 비교한다. 곡선의 마지막 값을 다시 1로 정규화하지 않는다. 마지막
값은 전체 standardized residual energy의 평균이며 다음 정보를 보존한다.

- endpoint > 1: 모델이 residual variance를 전체적으로 작게 평가
- endpoint < 1: 모델이 residual variance를 전체적으로 크게 평가
- `max|y-x|`: 누적곡선의 최대 spectral calibration 이탈

Low/middle/high band-reset 그림은 각 band의 누적곡선을 다시 `(0,0)`에서
시작한다. 이 그림은 앞 band에서 생긴 누적 오차를 다음 band로 전달하지 않아
어느 spectral range에서 mismatch가 발생하는지 구분한다.

## 7. Random-start 변동을 통제하는 방법

한 번의 random-start Lanczos에서 선택한 512개 mode의 평균 energy는 seed에
따라 변한다. Exact independent mode라는 이상화 아래 하루 endpoint의 표준편차는
대략 다음과 같다.

```text
sqrt(2/512) = 0.0625
```

따라서 selected mode를 1024개로 바로 늘리는 대신 다음 paired replicate
설계를 사용한다.

1. Adapted와 fixed에 동일한 8개 SLQ Rademacher probe를 사용한다.
2. 각 날짜에서 동일한 네 개 Lanczos start를 adapted와 fixed에 사용한다.
3. 각 start에서 512-mode 누적곡선을 계산한다.
4. 네 곡선의 평균과 start 간 standard error를 보고한다.

네 start 평균의 이상화된 endpoint 표준편차는 약 `0.0625/sqrt(4)=0.03125`로
감소한다. 동일한 난수를 두 방법에 사용하므로 adapted-fixed 차이에 random
probe 차이가 섞이지 않는다. 이는 common-random-number를 이용한 paired
comparison이다.

네 start는 plot 해상도를 2048 point로 늘리는 것이 아니다. Plot의 spectral
resolution은 여전히 512이고, 네 반복은 seed 안정성을 평가하고 평균하기 위한
것이다.

## 8. 보고 지표와 해석 우선순위

### Native NLL

- 전체 Vecchia likelihood를 직접 비교한다.
- 낮을수록 좋다.
- Log determinant와 residual quadratic form을 함께 반영한다.

### Cumulative Ritz diagnostic

- 평균곡선이 `y=x`에 가까울수록 좋다.
- Endpoint가 1에 가까울수록 전체 variance calibration이 좋다.
- `max|y-x|`가 작을수록 spectrum 전반의 누적 calibration이 좋다.
- Low/middle/high reset curve로 mismatch 위치를 확인한다.

### Ritz approximation quality

- Relative tail residual이 tolerance 0.05 이하인 mode 비율을 보고한다.
- Quality pass가 낮은 날짜 또는 방법의 미세한 곡선 굴곡은 해석하지 않는다.
- Adapted와 fixed의 eigenbasis는 서로 다르므로 동일한 mode index의 peak를
  일대일로 비교하지 않는다. Spectral quantile curve와 band summary를 비교한다.

## 9. 테스트 설계

각 연도에서 7월의 다음 5일을 사용한다.

```text
July 3, 5, 7, 12, 15
```

총 10일이며 각 날짜에서 adapted와 fixed를 비교한다. 일별 결과와 함께 연도별
5일 평균을 작성한다.

사용 설정은 다음과 같다.

```text
SLQ:              8 paired probes x 256 steps
Lanczos:          4 paired starts
Candidate space:  1536 steps/start
Selected modes:   170/170/172 = 512/start
Ritz tolerance:   relative tail residual <= 0.05
```

## 10. 최종 결론의 형태

보고서에서는 한 지표만으로 adapted 또는 fixed가 우월하다고 결론 내리지 않는다.

1. Native NLL로 전체 probabilistic fit을 비교한다.
2. Full-data cumulative curve로 spectral calibration을 비교한다.
3. Band-reset curve로 차이가 저·중·고주파 중 어디에서 발생하는지 확인한다.
4. 결론을 내리기 전에 four-start variability와 Ritz quality를 확인한다.

따라서 이 분석은 `SLQ + Lanczos`를 결합한 full-data eigen diagnostic이다.
SLQ는 전체 spectrum의 분포와 band 경계를 제공하고, Lanczos/Ritz는 residual을
투영할 대표 eigen-directions를 근사한다.
