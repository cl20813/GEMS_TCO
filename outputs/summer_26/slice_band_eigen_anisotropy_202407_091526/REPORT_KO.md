# Latitude/longitude band eigenanalysis 결과

## 분석 설계

- 자료: 2024년 7월 13일, 19일, 25일의 시뮬레이션과 실제 자료. 날짜마다 8개 시간장을 사용했다.
- 위도 band: `[-3,-2)`, `[-2,-1)`, `[-1,0)`, `[0,1)`, `[1,2]`의 5개이다.
- 주 경도 band: `[122,123)`, `[124,125)`, `[126,127)`, `[128,129)`, `[130,131]`의 5개이다.
- 보조 경도 band: `[121,122)`, `[123,124)`, `[125,126)`, `[127,128)`, `[129,130)`도 분석하여 경도 band 시작점에 대한 민감도를 확인했다.
- 위도 band 안의 각 grid row는 E-W longitude profile을 만들고, 경도 band 안의 각 grid column은 N-S latitude profile을 만든다.
- E-W profile의 원래 길이는 10도이고 N-S profile은 5도이므로 raw eigenvalue를 직접 비교하지 않았다. E-W profile을 두 개의 5도 window로 나누고 양쪽 모두 동일한 80개 위치로 재표본화했다.
- 각 날짜와 band 안에서 covariance를 spatial lag별로 pooling하여 `80 x 80` stationary Toeplitz correlation matrix를 만든 뒤 eigen decomposition했다.
- 시뮬레이션은 알려진 생성 평균을 제거했다. 실제 자료는 intercept, source latitude, day fixed effect, hour-slot fixed effect를 적합하여 제거했다.

실제 사용한 regular-grid center 범위는 위도 `-2.972`부터 `2.000`, 경도 `121.046`부터 `131.000`이다. 설정상의 연구 영역은 `[-3,2] x [121,131]`이 맞으며, 차이는 cell center가 경계 안쪽에 있기 때문이다.

## 지표 해석

- `effective rank`가 작으면 variance가 소수의 smooth eigenmode에 더 집중되어 있다는 뜻이다.
- `lambda1 fraction`이 크면 첫 번째 큰-scale mode가 전체 correlation trace에서 차지하는 비율이 크다는 뜻이다.
- `modes90`이 작으면 전체 trace의 90%를 설명하는 데 필요한 mode 수가 적다는 뜻이다.
- 따라서 같은 길이와 같은 80-point 해상도에서 `E-W effective rank < N-S effective rank`이면 E-W 방향의 correlation이 더 길고 smooth하다는 방향성 증거이다.

## 시뮬레이션 결과

시뮬레이션 truth는 `range_lon=0.3`, `range_lat=0.2`이므로 E-W 방향이 N-S보다 더 smooth해야 한다. 분석이 이 순서를 세 날짜 모두 회복했다.

| 날짜 | E-W effective rank | N-S effective rank | E-W/N-S | E-W lambda1 | N-S lambda1 | E-W modes90 | N-S modes90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jul 13 | 33.754 | 40.746 | 0.828 | 0.0939 | 0.0786 | 37.6 | 42.4 |
| Jul 19 | 33.416 | 42.242 | 0.791 | 0.0934 | 0.0717 | 37.2 | 43.6 |
| Jul 25 | 32.168 | 41.859 | 0.768 | 0.1078 | 0.0793 | 36.2 | 43.6 |

평균 effective-rank ratio는 `0.796`이고 평균 lambda1 ratio는 `1.285`이다. 즉 알려진 longitude-longer anisotropy를 올바른 방향으로 검출했다. 보조 경도 band를 사용한 일별 ratio도 `0.793`, `0.806`, `0.801`로 거의 변하지 않았다.

## 실제 자료 결과

실제 자료도 세 날짜 모두 E-W 방향에 더 강한 eigenvalue concentration이 나타났다.

| 날짜 | E-W effective rank | N-S effective rank | E-W/N-S | E-W lambda1 | N-S lambda1 | E-W modes90 | N-S modes90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jul 13 | 21.398 | 33.034 | 0.648 | 0.2506 | 0.1229 | 31.4 | 37.4 |
| Jul 19 | 28.336 | 36.923 | 0.767 | 0.1866 | 0.1479 | 41.2 | 45.4 |
| Jul 25 | 16.590 | 33.730 | 0.492 | 0.3363 | 0.1939 | 27.8 | 44.4 |

평균 effective-rank ratio는 `0.636`, 평균 lambda1 ratio는 `1.678`이다. 방향 차이는 시뮬레이션보다 크지만 날짜와 위치에 따른 변동도 더 크다. 특히 Jul 25의 E-W spectrum이 매우 집중되어 있다. 보조 경도 band를 사용해도 일별 ratio는 `0.586`, `0.673`, `0.464`로 모두 1보다 작아 방향 순서는 유지되었다.

## 결론

이 slice eigenanalysis는 **E-W와 N-S의 axis anisotropy를 진단하는 데 유효하다.** 그 근거는 다음과 같다.

1. 알려진 anisotropic simulation에서 정확한 방향 순서를 세 날짜 모두 회복했다.
2. 실제 자료에서도 같은 방향 순서가 세 날짜와 두 가지 경도 band 선택 모두에서 유지되었다.
3. 실제 자료의 더 큰 day/band 변동은 단일 stationary anisotropy만이 아니라 공간적 비정상성, 잔여 mean structure, 구름·missing pattern 등이 함께 존재할 가능성을 보여준다.

단, raw eigenvalue 크기 자체를 비교하면 안 된다. 원래 E-W와 N-S 영역의 길이, grid spacing, dimension, residual variance가 다르기 때문이다. 동일 길이·동일 해상도의 correlation eigenvalue를 trace-normalize해서 비교해야 한다.

또한 이 분석은 같은 시간의 spatial covariance를 사용하므로 advection의 부호나 `h` 대 `-h` asymmetry를 검출하지 못한다. 그 목적에는 positive time lag의 oriented cross-variogram asymmetry 또는 odd-odd space-time contrast가 필요하다.

현재 결과는 서로 가까운 profile과 시간이 상관되어 있는 descriptive diagnostic이다. 공식적인 유의성 판단에는 fitted model 아래의 parametric bootstrap 또는 independent-day block bootstrap이 필요하다.
