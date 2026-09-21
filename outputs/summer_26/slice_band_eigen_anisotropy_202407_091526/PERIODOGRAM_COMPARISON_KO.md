# 방향별 periodogram과 eigenanalysis 비교

- 두 진단 모두 같은 5도 profile을 80개 등간격 위치로 선형보간한 자료를 사용한다.
- Eigenanalysis와 동일하게 각 위치에서 profile 평균을 제거한 뒤 periodogram을 평균했다.
- 각 periodogram과 eigenspectrum은 합이 1이 되도록 정규화했다.
- Low-frequency fraction은 1 cycle/degree 이하를 뜻한다. 이는 적도 부근에서 파장 1도, 약 111 km 이상에 해당한다.
- Ranked periodogram은 eigenvalue concentration과 비교하는 용도일 뿐이고, frequency 해석은 frequency-ordered plot을 사용해야 한다.

| 자료 | 방향 | low-frequency fraction | spectral centroid (cycles/degree) | median frequency (cycles/degree) | ranked TV distance | eigen effective rank |
|---|---|---:|---:|---:|---:|---:|
| real | east_west | 0.7368 (0.6355–0.8359) | 1.0155 | 0.2765 | 0.0843 | 22.108 |
| real | north_south | 0.6243 (0.5130–0.8015) | 1.4100 | 0.6715 | 0.0580 | 34.562 |
| simulation | east_west | 0.6635 (0.6285–0.6870) | 1.2862 | 0.6320 | 0.0314 | 33.113 |
| simulation | north_south | 0.5542 (0.5065–0.6255) | 1.5900 | 0.9217 | 0.0353 | 41.616 |

## 방향 차이

- real: E-W/N-S low-frequency fraction ratio=1.180; E-W/N-S spectral-centroid ratio=0.720.
- simulation: E-W/N-S low-frequency fraction ratio=1.197; E-W/N-S spectral-centroid ratio=0.809.

## 해석

- Lag-pooled autocovariance와 averaged periodogram은 Fourier transform 관계이므로 ranked periodogram과 eigenvalue 곡선이 비슷한 것이 정상이다.
- 유한 Toeplitz 행렬의 eigenvalue가 Fourier power와 정확히 같을 필요는 없다. 정확한 일치는 circulant covariance matrix에서 성립한다.
- Low-frequency fraction이 크고 spectral centroid가 작을수록 해당 방향의 구조가 더 부드럽고 장거리 scale에 집중되어 있다는 뜻이다.
- 이 same-time power 진단은 축별 anisotropy를 보여주지만 동쪽 또는 서쪽 이동의 부호는 식별하지 못한다.
