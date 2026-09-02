# Four-geometry lag432 Vecchia comparison

Report state: **COMPLETE** (10/10 data sets complete).

## Conditioning methods

- adapted: calibrated t-1 [0.5v,1.5v], t-2 [0,2v] corridors
- shifted: calibrated v and 2v nearest-block centers
- fixed: target-centered nearest blocks
- union: exact deduplicated union of the preceding three sets

## Run completeness

| data_source | date | data_kind | dataset_id | task_status | last_stage | failure_type | failure_message | fit_methods | missing_fit_methods | cross_rows | expected_cross_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_gems_tco | 2024-07-03 | real | real_20240703 | complete | complete |  |  | adapted,shifted,fixed,union |  | 16 | 16 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | complete | complete |  |  | adapted,shifted,fixed,union |  | 16 | 16 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | complete | complete |  |  | adapted,shifted,fixed,union |  | 16 | 16 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | complete | complete |  |  | adapted,shifted,fixed,union |  | 16 | 16 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | complete | complete |  |  | adapted,shifted,fixed,union |  | 16 | 16 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | complete | complete |  |  | adapted,shifted,fixed,union |  | 20 | 20 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | complete | complete |  |  | adapted,shifted,fixed,union |  | 20 | 20 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | complete | complete |  |  | adapted,shifted,fixed,union |  | 20 | 20 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | complete | complete |  |  | adapted,shifted,fixed,union |  | 20 | 20 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | complete | complete |  |  | adapted,shifted,fixed,union |  | 20 | 20 |

## Daily fitted performance

| data_source | date | data_kind | dataset_id | method | init_advec_lat | init_advec_lon | native_nll | union_nll | union_gap | eigen_D | precompute_seconds | fit_seconds | diagnostic_seconds | method_total_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_gems_tco | 2024-07-03 | real | real_20240703 | adapted | -0.0862 | -0.2540 | 1.3822 | 1.3818 | 0.0000 | 10.7452 | 1.3469 | 149.4388 | 2.8665 | 153.6522 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | shifted | -0.0862 | -0.2540 | 1.3828 | 1.3818 | 0.0000 | 10.7876 | 0.5318 | 154.0309 | 3.1561 | 157.7188 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | fixed | -0.0862 | -0.2540 | 1.3896 | 1.3855 | 0.0037 | 11.6521 | 0.3949 | 142.4466 | 3.0574 | 145.8989 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | union | -0.0862 | -0.2540 | 1.3818 | 1.3818 | 0.0000 | 10.6961 | 1.8733 | 709.2046 | 11.2067 | 722.2846 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | adapted | 0.0566 | -0.2220 | 1.3559 | 1.3556 | 0.0000 | 8.5235 | 1.2958 | 164.2168 | 3.2133 | 168.7259 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | shifted | 0.0566 | -0.2220 | 1.3559 | 1.3556 | 0.0000 | 8.5678 | 0.5567 | 152.8638 | 3.2041 | 156.6245 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | fixed | 0.0566 | -0.2220 | 1.3602 | 1.3576 | 0.0020 | 9.2116 | 0.4011 | 161.9069 | 3.1660 | 165.4739 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | union | 0.0566 | -0.2220 | 1.3556 | 1.3556 | 0.0000 | 8.5589 | 1.6977 | 529.2919 | 12.1073 | 543.0969 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | adapted | 0.0055 | -0.3655 | 1.3201 | 1.3200 | 0.0000 | 10.1492 | 1.3858 | 138.5790 | 3.1678 | 143.1326 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | shifted | 0.0055 | -0.3655 | 1.3201 | 1.3200 | 0.0000 | 10.1734 | 0.5349 | 118.9683 | 3.2530 | 122.7562 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | fixed | 0.0055 | -0.3655 | 1.3263 | 1.3250 | 0.0050 | 10.8878 | 0.3951 | 140.7461 | 3.1062 | 144.2474 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | union | 0.0055 | -0.3655 | 1.3200 | 1.3200 | 0.0000 | 10.1754 | 1.8575 | 675.5895 | 11.1425 | 688.5895 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | adapted | -0.0109 | -0.0025 | 1.1796 | 1.1796 | 0.0000 | 11.3644 | 1.2115 | 158.6028 | 3.2133 | 163.0276 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | shifted | -0.0109 | -0.0025 | 1.1796 | 1.1796 | 0.0000 | 11.3644 | 0.5380 | 164.3002 | 2.8898 | 167.7280 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | fixed | -0.0109 | -0.0025 | 1.1796 | 1.1796 | 0.0000 | 11.3762 | 0.3916 | 192.8837 | 2.8874 | 196.1626 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | union | -0.0109 | -0.0025 | 1.1796 | 1.1796 | 0.0000 | 11.3784 | 1.6225 | 704.8970 | 12.0234 | 718.5429 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | adapted | 0.0183 | -0.1131 | 1.2006 | 1.2005 | 0.0000 | 12.1652 | 1.1659 | 134.7022 | 2.8681 | 138.7363 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | shifted | 0.0183 | -0.1131 | 1.2004 | 1.2007 | 0.0001 | 12.0520 | 0.5155 | 160.5716 | 2.8463 | 163.9334 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | fixed | 0.0183 | -0.1131 | 1.2010 | 1.2007 | 0.0001 | 12.1216 | 0.3842 | 136.1634 | 2.8296 | 139.3772 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | union | 0.0183 | -0.1131 | 1.2005 | 1.2005 | 0.0000 | 12.1355 | 1.6011 | 665.8525 | 10.9264 | 678.3800 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | adapted | 0.0824 | -0.1971 | 1.1377 | 1.1369 | 0.0000 | 13.8338 | 1.2174 | 143.3469 | 2.9241 | 147.4883 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | shifted | 0.0824 | -0.1971 | 1.1376 | 1.1369 | 0.0000 | 13.8374 | 0.5302 | 161.1104 | 3.0048 | 164.6453 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | fixed | 0.0824 | -0.1971 | 1.1449 | 1.1374 | 0.0005 | 14.4424 | 0.3981 | 140.1154 | 3.0270 | 143.5405 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | union | 0.0824 | -0.1971 | 1.1369 | 1.1369 | 0.0000 | 13.7556 | 1.6386 | 768.3414 | 11.2720 | 781.2520 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | adapted | 0.0751 | -0.2047 | 1.1334 | 1.1326 | 0.0000 | 12.6862 | 1.2388 | 157.6403 | 3.0560 | 161.9351 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | shifted | 0.0751 | -0.2047 | 1.1333 | 1.1326 | 0.0000 | 12.6801 | 0.5327 | 159.1634 | 3.0816 | 162.7777 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | fixed | 0.0751 | -0.2047 | 1.1408 | 1.1331 | 0.0005 | 13.5372 | 0.3989 | 181.2109 | 3.1105 | 184.7203 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | union | 0.0751 | -0.2047 | 1.1326 | 1.1326 | 0.0000 | 12.6310 | 1.6825 | 726.7060 | 11.0256 | 739.4141 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | adapted | 0.0825 | -0.2042 | 1.1312 | 1.1303 | 0.0000 | 12.8539 | 1.2466 | 189.6881 | 3.0695 | 194.0042 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | shifted | 0.0825 | -0.2042 | 1.1310 | 1.1303 | 0.0000 | 12.8613 | 0.5411 | 156.3257 | 3.1215 | 159.9883 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | fixed | 0.0825 | -0.2042 | 1.1379 | 1.1305 | 0.0002 | 13.4844 | 0.3983 | 132.0032 | 3.0583 | 135.4599 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | union | 0.0825 | -0.2042 | 1.1303 | 1.1303 | 0.0000 | 12.8430 | 1.6881 | 734.5717 | 11.7001 | 747.9599 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | adapted | 0.0762 | -0.2005 | 1.1318 | 1.1310 | 0.0000 | 12.7438 | 1.2540 | 166.1037 | 3.0909 | 170.4485 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | shifted | 0.0762 | -0.2005 | 1.1316 | 1.1310 | 0.0000 | 12.7484 | 0.5432 | 157.1304 | 3.0809 | 160.7546 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | fixed | 0.0762 | -0.2005 | 1.1390 | 1.1315 | 0.0005 | 13.3613 | 0.4007 | 145.6731 | 3.1019 | 149.1757 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | union | 0.0762 | -0.2005 | 1.1310 | 1.1310 | 0.0000 | 12.7340 | 1.7223 | 692.9091 | 11.8099 | 706.4413 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | adapted | 0.0797 | -0.2010 | 1.1361 | 1.1354 | 0.0000 | 13.5905 | 1.2016 | 172.8529 | 2.6156 | 176.6700 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | shifted | 0.0797 | -0.2010 | 1.1360 | 1.1354 | 0.0000 | 13.5905 | 0.5197 | 167.6909 | 2.5756 | 170.7861 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | fixed | 0.0797 | -0.2010 | 1.1432 | 1.1357 | 0.0004 | 14.3117 | 0.3973 | 175.4637 | 2.5932 | 178.4542 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | union | 0.0797 | -0.2010 | 1.1354 | 1.1354 | 0.0000 | 13.5583 | 1.7097 | 617.7562 | 9.3427 | 628.8086 |

## Final fitted seven-parameter vectors

| data_source | date | data_kind | dataset_id | method | sigmasq_hat | range_lat_hat | range_lon_hat | range_time_hat | v_lat_hat | v_lon_hat | nugget_hat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_gems_tco | 2024-07-03 | real | real_20240703 | adapted | 13.5468 | 0.1056 | 0.1290 | 1.0228 | -0.0724 | -0.2850 | 0.0000 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | shifted | 13.6267 | 0.1064 | 0.1298 | 1.0236 | -0.0681 | -0.2787 | 0.0000 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | fixed | 14.1337 | 0.1112 | 0.1358 | 0.8829 | -0.0421 | -0.1711 | 0.0000 |
| real_gems_tco | 2024-07-03 | real | real_20240703 | union | 13.4791 | 0.1050 | 0.1283 | 1.0139 | -0.0708 | -0.2811 | 0.0000 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | adapted | 13.3087 | 0.1093 | 0.1361 | 1.0115 | 0.0521 | -0.2179 | 0.0000 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | shifted | 13.3573 | 0.1098 | 0.1367 | 1.0148 | 0.0530 | -0.2176 | 0.0000 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | fixed | 13.7866 | 0.1138 | 0.1423 | 0.9498 | 0.0506 | -0.1186 | 0.0000 |
| real_gems_tco | 2024-07-05 | real | real_20240705 | union | 13.3425 | 0.1096 | 0.1365 | 1.0147 | 0.0528 | -0.2144 | 0.0000 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | adapted | 14.5760 | 0.1263 | 0.1740 | 1.1436 | 0.0043 | -0.3852 | 0.0000 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | shifted | 14.6218 | 0.1267 | 0.1746 | 1.1492 | 0.0039 | -0.3855 | 0.0000 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | fixed | 15.1880 | 0.1323 | 0.1826 | 0.9073 | 0.0128 | -0.1288 | 0.0000 |
| real_gems_tco | 2024-07-07 | real | real_20240707 | union | 14.5815 | 0.1263 | 0.1742 | 1.1438 | 0.0036 | -0.3826 | 0.0000 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | adapted | 9.0096 | 0.1038 | 0.1324 | 0.9246 | 0.0000 | 0.0052 | 0.0000 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | shifted | 9.0096 | 0.1038 | 0.1324 | 0.9246 | 0.0000 | 0.0052 | 0.0000 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | fixed | 9.0071 | 0.1038 | 0.1323 | 0.9233 | 0.0002 | 0.0051 | 0.0000 |
| real_gems_tco | 2024-07-15 | real | real_20240715 | union | 8.9997 | 0.1037 | 0.1322 | 0.9205 | 0.0001 | 0.0053 | 0.0000 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | adapted | 9.3814 | 0.1049 | 0.1341 | 0.6892 | 0.0172 | -0.1047 | 0.0000 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | shifted | 9.3920 | 0.1050 | 0.1341 | 0.7110 | 0.0109 | -0.0498 | 0.0000 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | fixed | 9.4278 | 0.1054 | 0.1348 | 0.7030 | 0.0155 | -0.0480 | 0.0000 |
| real_gems_tco | 2024-07-22 | real | real_20240722 | union | 9.3623 | 0.1047 | 0.1338 | 0.6840 | 0.0172 | -0.1315 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | adapted | 9.3133 | 0.1095 | 0.1632 | 1.2521 | 0.0794 | -0.1952 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | shifted | 9.3231 | 0.1096 | 0.1633 | 1.2544 | 0.0798 | -0.1957 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | fixed | 9.6706 | 0.1145 | 0.1704 | 1.2680 | 0.0765 | -0.1705 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-04 | synthetic | synthetic_20230704 | union | 9.2734 | 0.1089 | 0.1623 | 1.2478 | 0.0802 | -0.1934 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | adapted | 8.8287 | 0.1030 | 0.1548 | 1.1942 | 0.0756 | -0.2012 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | shifted | 8.8404 | 0.1031 | 0.1550 | 1.1977 | 0.0758 | -0.2009 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | fixed | 9.1994 | 0.1080 | 0.1626 | 1.2127 | 0.0787 | -0.1769 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2023-07-29 | synthetic | synthetic_20230729 | union | 8.8117 | 0.1027 | 0.1545 | 1.1950 | 0.0771 | -0.1986 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | adapted | 8.7043 | 0.1030 | 0.1528 | 1.1590 | 0.0805 | -0.2034 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | shifted | 8.7188 | 0.1032 | 0.1531 | 1.1640 | 0.0802 | -0.2029 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | fixed | 9.0120 | 0.1073 | 0.1592 | 1.2077 | 0.0798 | -0.1846 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-13 | synthetic | synthetic_20240713 | union | 8.6868 | 0.1027 | 0.1525 | 1.1604 | 0.0817 | -0.2006 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | adapted | 8.7464 | 0.1035 | 0.1527 | 1.1629 | 0.0786 | -0.2008 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | shifted | 8.7543 | 0.1036 | 0.1528 | 1.1672 | 0.0789 | -0.2002 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | fixed | 9.0259 | 0.1075 | 0.1586 | 1.1409 | 0.0774 | -0.1768 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2024-07-19 | synthetic | synthetic_20240719 | union | 8.7420 | 0.1034 | 0.1526 | 1.1678 | 0.0794 | -0.1988 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | adapted | 8.8038 | 0.1028 | 0.1535 | 1.1825 | 0.0799 | -0.1992 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | shifted | 8.8153 | 0.1029 | 0.1537 | 1.1863 | 0.0798 | -0.1993 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | fixed | 9.1865 | 0.1080 | 0.1614 | 1.2298 | 0.0796 | -0.1784 | 0.0000 |
| synthetic_dgp_smooth0p5_nugget1_fit_nugget0 | 2025-07-06 | synthetic | synthetic_20250706 | union | 8.7952 | 0.1026 | 0.1533 | 1.1830 | 0.0804 | -0.1978 | 0.0000 |

## Mean performance by data type and method

| data_kind | method | native_nll | union_nll | union_gap | precompute_seconds | fit_seconds | diagnostic_seconds | method_total_seconds | eigen_D | eigen_mean_y2 | advection_error_cells | combined_parameter_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real | adapted | 1.2877 | 1.2875 | 0.0000 | 1.2812 | 149.1079 | 3.0658 | 153.4549 | 10.5895 | 1.0001 | — | — |
| real | fixed | 1.2913 | 1.2897 | 0.0022 | 0.3934 | 154.8293 | 3.0093 | 158.2320 | 11.0499 | 1.0001 | — | — |
| real | shifted | 1.2878 | 1.2875 | 0.0000 | 0.5354 | 150.1470 | 3.0699 | 153.7522 | 10.5890 | 1.0001 | — | — |
| real | union | 1.2875 | 1.2875 | 0.0000 | 1.7304 | 656.9671 | 11.4812 | 670.1788 | 10.5889 | 1.0001 | — | — |
| synthetic | adapted | 1.1340 | 1.1332 | 0.0000 | 1.2316 | 165.9264 | 2.9512 | 170.1092 | 13.1417 | 1.0001 | 0.0562 | 1.0702 |
| synthetic | fixed | 1.1412 | 1.1336 | 0.0004 | 0.3986 | 154.8933 | 2.9782 | 158.2701 | 13.8274 | 1.0001 | 0.3604 | 1.0680 |
| synthetic | shifted | 1.1339 | 1.1332 | 0.0000 | 0.5334 | 160.2841 | 2.9729 | 163.7904 | 13.1435 | 1.0001 | 0.0497 | 1.0666 |
| synthetic | union | 1.1332 | 1.1332 | 0.0000 | 1.6882 | 708.0569 | 11.0301 | 720.7752 | 13.1044 | 1.0001 | 0.0546 | 1.0730 |

## Synthetic truth performance

| date | method | v_lat_hat | v_lon_hat | advection_error_cells | combined_parameter_error |
| --- | --- | --- | --- | --- | --- |
| 2023-07-04 | adapted | 0.0794 | -0.1952 | 0.0768 | 0.9817 |
| 2023-07-04 | shifted | 0.0798 | -0.1957 | 0.0689 | 0.9788 |
| 2023-07-04 | fixed | 0.0765 | -0.1705 | 0.4748 | 1.0322 |
| 2023-07-04 | union | 0.0802 | -0.1934 | 0.1055 | 0.9928 |
| 2023-07-29 | adapted | 0.0756 | -0.2012 | 0.1016 | 1.0817 |
| 2023-07-29 | shifted | 0.0758 | -0.2009 | 0.0970 | 1.0781 |
| 2023-07-29 | fixed | 0.0787 | -0.1769 | 0.3673 | 1.0707 |
| 2023-07-29 | union | 0.0771 | -0.1986 | 0.0685 | 1.0819 |
| 2024-07-13 | adapted | 0.0805 | -0.2034 | 0.0554 | 1.1025 |
| 2024-07-13 | shifted | 0.0802 | -0.2029 | 0.0456 | 1.0975 |
| 2024-07-13 | fixed | 0.0798 | -0.1846 | 0.2437 | 1.0553 |
| 2024-07-13 | union | 0.0817 | -0.2006 | 0.0394 | 1.1042 |
| 2024-07-19 | adapted | 0.0786 | -0.2008 | 0.0350 | 1.0966 |
| 2024-07-19 | shifted | 0.0789 | -0.2002 | 0.0256 | 1.0933 |
| 2024-07-19 | fixed | 0.0774 | -0.1768 | 0.3731 | 1.1210 |
| 2024-07-19 | union | 0.0794 | -0.1988 | 0.0236 | 1.0952 |
| 2025-07-06 | adapted | 0.0799 | -0.1992 | 0.0123 | 1.0885 |
| 2025-07-06 | shifted | 0.0798 | -0.1993 | 0.0115 | 1.0852 |
| 2025-07-06 | fixed | 0.0796 | -0.1784 | 0.3430 | 1.0607 |
| 2025-07-06 | union | 0.0804 | -0.1978 | 0.0363 | 1.0907 |

All displayed numeric values are rounded to four decimal places. The flat
comparison table is `vecchia_comparison_summary.csv`; parameter-level truth and
fit values are in `vecchia_parameter_details.csv`.
